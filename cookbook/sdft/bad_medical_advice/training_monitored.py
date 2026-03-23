"""training_monitored.py — variant of training.py with MonitoringCallback injected.

Usage (identical to training.py):
    accelerate launch training_monitored.py <job_id>

The job's ``params`` dict may include an extra key ``monitoring_eval_steps``
(default 100) that controls how often the monitoring metrics are computed.
This key lives *outside* ``validated_params`` so it never reaches
``TrainingConfig`` (which rejects unknown fields via ``extra="forbid"``).

Supported losses: ``"sft"``, ``"sdft"``, and ``"grpo"``.
All other training hyperparameters are identical to the standard trainer.
"""

import json
import os
import sys

from datasets import Dataset
from grpo_ft import grpo_train
from monitoring_callback import MonitoringCallback
from sdft import sdft_train
from sft import sft_train
from training import create_dataset, push_model, standardize_datasets
from unsloth import FastLanguageModel
from utils import client, load_jsonl, load_model_and_tokenizer
from validate import TrainingConfig


def train_monitored(config: dict, monitoring_eval_steps: int = 100):
    """
    Replicate the ``train()`` logic from ``training.py`` and inject
    ``MonitoringCallback`` into the trainer before calling ``.train()``.

    Parameters
    ----------
    config : dict
        ``validated_params`` dict from the job record — passed directly to
        ``TrainingConfig``.
    monitoring_eval_steps : int
        How many optimizer steps between monitoring evaluations.
    """
    training_cfg = TrainingConfig(**config)

    # ── Load base model and tokenizer ──────────────────────────────────────
    # Mirror training.py: when GRPO + grpo_use_vllm=True, pass fast_inference=True
    # so Unsloth creates model.vllm_engine internally.  Without this, TRL's
    # GRPOTrainer sees use_vllm=True in GRPOConfig and tries to connect to an
    # external vLLM server on port 8000 — which doesn't exist.
    _use_vllm = training_cfg.loss == "grpo" and getattr(
        training_cfg, "grpo_use_vllm", False
    )
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
        use_vllm=_use_vllm,
        max_lora_rank=training_cfg.r,
    )
    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

    # ── Attach LoRA adapter ────────────────────────────────────────────────
    print("Creating new LoRA adapter")
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=training_cfg.target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    # ── Load datasets ──────────────────────────────────────────────────────
    rows = load_jsonl(training_cfg.training_file)
    dataset = create_dataset(rows, training_cfg.loss)
    # GRPO uses "prompt"/"gold_response" columns — skip ShareGPT standardisation
    if training_cfg.loss != "grpo":
        dataset, _ = standardize_datasets(training_cfg.model, dataset)

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = create_dataset(test_rows, training_cfg.loss)
        if training_cfg.loss != "grpo":
            test_dataset, _ = standardize_datasets(training_cfg.model, test_dataset)
    else:
        test_dataset = None
        training_cfg.test_file_eval_strategy = "no"

    # ── logp_callback datasets (pass-through, same as training.py) ─────────
    logp_datasets = {}
    for key, logp_file in training_cfg.logp_callback_datasets.items():
        lp_rows = load_jsonl(logp_file)
        logp_datasets[key] = Dataset.from_list(
            [dict(messages=r["messages"]) for r in lp_rows]
        )

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    # ── Build trainer ──────────────────────────────────────────────────────
    if training_cfg.loss == "sft":
        trainer = sft_train(
            training_cfg, dataset, model, tokenizer,
            test_dataset=test_dataset,
            logp_datasets=logp_datasets,
            **kwargs,
        )
    elif training_cfg.loss == "sdft":
        trainer = sdft_train(
            training_cfg, dataset, model, tokenizer,
            test_dataset=test_dataset,
            logp_datasets=logp_datasets,
            **kwargs,
        )
    elif training_cfg.loss == "grpo":
        trainer = grpo_train(
            training_cfg, dataset, model, tokenizer,
            test_dataset=test_dataset,
            logp_datasets=logp_datasets,
            **kwargs,
        )
    else:
        raise ValueError(
            f"training_monitored.py only supports loss='sft', 'sdft', or 'grpo', "
            f"got '{training_cfg.loss}'"
        )

    # ── Build eval corpus for KL metric (first ≤8 rows, chat-templated) ───
    # For SFT/SDFT the training sequences include the full conversation
    # (prompt + assistant response), so we template with the assistant turn
    # present and add_generation_prompt=False.
    #
    # For GRPO the model is trained on *prompts only* (the assistant turn is
    # stripped by create_dataset).  Including the gold response in eval_texts
    # would mean kl_vs_base is measured on text the GRPO model never trained
    # on, making the metric incomparable to SFT/SDFT.  For GRPO we therefore
    # strip the final assistant turn and add_generation_prompt=True so the
    # eval text represents what the model actually saw during training.
    eval_texts = []
    for row in rows[:8]:
        try:
            messages = row["messages"]
            if training_cfg.loss == "grpo":
                # Strip last assistant turn to match GRPO training distribution
                if messages and messages[-1].get("role") == "assistant":
                    messages = messages[:-1]
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            eval_texts.append(text)
        except Exception:
            pass

    # ── Inject MonitoringCallback ──────────────────────────────────────────
    monitoring_cb = MonitoringCallback(
        model=model,
        tokenizer=tokenizer,
        monitoring_eval_steps=monitoring_eval_steps,
        eval_texts=eval_texts,
    )
    trainer.add_callback(monitoring_cb)

    # ── Train ──────────────────────────────────────────────────────────────
    if test_dataset:
        trainer.evaluate()
    trainer.train()

    # ── Push to HuggingFace Hub ────────────────────────────────────────────
    finetuned_model_id = (
        training_cfg.finetuned_model_id
        or f"{training_cfg.model}:ft-{client.run.id}"
    )
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    if test_dataset:
        try:
            trainer.evaluate()
        except Exception as e:
            print(
                f"Error running post-training evaluation: {e}. "
                "Model has already been pushed to the hub."
            )


def main(job_id: str):
    """
    Entry point.  Reads job params, separates ``monitoring_eval_steps`` from
    ``validated_params``, then calls ``train_monitored``.
    """
    if os.path.exists(job_id):
        with open(job_id) as f:
            all_params = json.load(f)
    else:
        job = client.jobs.retrieve(job_id)
        all_params = job["params"]

    monitoring_eval_steps = int(all_params.get("monitoring_eval_steps", 100))
    config = all_params["validated_params"]

    print(f"Training config:\n{json.dumps(config, indent=4)}")
    print(f"Monitoring every {monitoring_eval_steps} steps")

    train_monitored(config, monitoring_eval_steps=monitoring_eval_steps)


if __name__ == "__main__":
    main(sys.argv[1])
