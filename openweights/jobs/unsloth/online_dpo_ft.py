from transformers import TrainingArguments
from unsloth import PatchDPOTrainer, is_bfloat16_supported
import logging
import json

from trl import OnlineDPOTrainer, OnlineDPOConfig
from online_dpo_trainer import OnlineDPOTrainerCustom

from utils import GPUStatsCallback, LogMetrics
from judges import OpenAIJudge
import judges
from judges import ONLINE_DPO_RESPONSE_PLACEHOLDER


def online_dpo_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):

    def apply_chat_template_to_prompt_data(examples):
        """Convert conversation data to prompt-only format for online DPO"""
        messages = examples["messages"]
        out = {"prompt": []}
        for message_list in messages:
            # Apply chat template to convert messages to a single prompt
            prompt = tokenizer.apply_chat_template(
                message_list,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
            )
            out["prompt"].append(prompt)
        return out

    dataset = dataset.map(apply_chat_template_to_prompt_data, batched=True)
    test_dataset = test_dataset.map(apply_chat_template_to_prompt_data, batched=True)

    learning_rate = (
        training_cfg.learning_rate
        if (not isinstance(training_cfg.learning_rate, str))
        else eval(training_cfg.learning_rate)
    )
    if learning_rate < 0:
        learning_rate = 10**learning_rate

    sample_hp = training_cfg.online_dpo["sampler"]

    args = OnlineDPOConfig(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=training_cfg.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=None,
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        # Online DPO specific parameters
        max_new_tokens=sample_hp["max_new_tokens"],
        temperature=sample_hp["temperature"],
        max_length=sample_hp["max_length"],
        **kwargs,
    )

    judge_prompts = {k: v for k, v in zip(dataset["prompt"], dataset["judge_prompt"])}
    for k, v in judge_prompts.items():
        assert isinstance(v, str), f"Judge prompt value {v} must be a string"
        assert isinstance(k, str), f"Prompt key {k} must be a string"

    # Assert that all strings in user_prompts contain the required placeholders
    for i, prompt in enumerate(judge_prompts.values()):
        if not isinstance(prompt, str):
            raise ValueError(f"online_dpo_user_prompts[{i}] must be a string")
        if ONLINE_DPO_RESPONSE_PLACEHOLDER not in prompt:
            raise ValueError(
                f"online_dpo_user_prompts[{i}] must contain the {ONLINE_DPO_RESPONSE_PLACEHOLDER} placeholder"
            )

    judge_hp = training_cfg.online_dpo["judge"]

    if judge_hp["judge_type"] == "openai":
        judge = OpenAIJudge(
            model=judge_hp["model"],
            judge_prompts=judge_prompts,
            system_prompt=judge_hp["system_prompt"],
            score_extractor=getattr(judges, judge_hp["score_extractor"]),
            max_requests=judge_hp["max_requests"],
            max_tokens=judge_hp["max_tokens"],
            temperature=judge_hp["temperature"],
            top_p=judge_hp["top_p"],
            frequency_penalty=judge_hp["frequency_penalty"],
            presence_penalty=judge_hp["presence_penalty"],
            openai_api_key=judge_hp["openai_api_key"],
        )
    else:
        raise ValueError(f"Invalid judge type: {judge_hp['judge_type']}")

    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": dataset,
        "eval_dataset": test_dataset,
        "args": args,
        # "max_prompt_length": training_cfg.max_seq_length,
        "max_length": training_cfg.max_seq_length,
        # "max_target_length": training_cfg.max_seq_length,
        # "pad_to_multiple_of": 8,
        "callbacks": [LogMetrics(), GPUStatsCallback()],
        "judge": judge,
    }

    # logging.error(f"Trainer kwargs: {json.dumps(trainer_kwargs, indent=4)}")

    trainer = OnlineDPOTrainerCustom(**trainer_kwargs)
    return trainer
