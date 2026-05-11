from typing import TYPE_CHECKING, Any

from logp_callback import LogTestLossCallback
from sampling_callback import SamplingCallback
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import (
    train_on_responses_only as unsloth_train_on_responses_only,
)
from utils import GPUStatsCallback, LogMetrics

if TYPE_CHECKING:
    from validate import TrainingConfig


def _warn_if_packing_disabled_at_runtime(
    training_cfg: "TrainingConfig",
    trainer: SFTTrainer,
) -> None:
    """Print when the job asked for packing but the live ``trainer.args`` has it off.

    Unsloth turns ``packing`` off for some setups (Gemma 2 is blocklisted, VLMs,
    custom collators, ``UNSLOTH_RETURN_LOGITS=1``, etc.); TRL then keeps one
    example per row, so step counts stay ~``len(dataset) / effective_batch``.
    """
    wants = bool(training_cfg.packing)
    has = bool(getattr(trainer.args, "packing", False))
    if wants and not has:
        print(
            "OpenWeights: `packing=True` in the job config but the trainer is running with "
            "`packing=False`. Unsloth usually prints why just above (e.g. vision-language model, "
            "unsupported architecture, custom data collator). Common cases: processor/VLM checkpoints, "
            "Gemma 2 blocklist, or `UNSLOTH_RETURN_LOGITS=1`. Without packing, each dataset row "
            "is one truncated sequence; step count stays about len(dataset) / effective batch size."
        )


def _sft_config_field_names() -> frozenset[str]:
    """Return dataclass field names on ``trl.SFTConfig`` (including inherited MRO).

    Unsloth rebuilds trainer ``args`` via ``SFTConfig(**dict_args)``. Callers may pass
    ``**kwargs`` intended for ``TrainingArguments``; filtering to known ``SFTConfig``
    fields avoids version skew (e.g. ``push_to_hub_token`` on newer Transformers).
    """
    names: set[str] = set()
    for cls in SFTConfig.__mro__:
        fds = getattr(cls, "__dataclass_fields__", None)
        if fds:
            names.update(fds.keys())
    if not names:
        raise RuntimeError(
            "Could not introspect TRL SFTConfig dataclass fields; upgrade trl "
            "(OpenWeights expects trl>=0.23)."
        )
    return frozenset(names)


def _filter_extra_training_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Drop caller kwargs that ``SFTConfig`` does not declare."""
    allowed = _sft_config_field_names()
    filtered = {k: v for k, v in extra.items() if k in allowed}
    dropped = sorted(set(extra) - set(filtered))
    if dropped:
        print(
            f"SFTTrainer args: ignoring keys not accepted by TRL SFTConfig: {dropped}"
        )
    return filtered


def _filter_sft_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize and filter built-in ``SFTConfig`` kwargs across TRL releases."""
    allowed = _sft_config_field_names()
    normalized = dict(config)

    if (
        "max_seq_length" in normalized
        and "max_seq_length" not in allowed
        and "max_length" in allowed
    ):
        normalized["max_length"] = normalized.pop("max_seq_length")

    filtered = {k: v for k, v in normalized.items() if k in allowed}
    dropped = sorted(set(normalized) - set(filtered))
    if dropped:
        print(
            f"SFTTrainer args: ignoring built-in keys not accepted by TRL SFTConfig: {dropped}"
        )
    return filtered


def _token_exists(tokenizer: PreTrainedTokenizerBase, token: str | None) -> bool:
    if not token:
        return False
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = None
    if vocab is not None and token not in vocab:
        return False
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return False
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    return token_id != unk_token_id


def _token_from_id(
    tokenizer: PreTrainedTokenizerBase, token_id: int | None
) -> str | None:
    if token_id is None or token_id < 0:
        return None
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        return None
    if isinstance(token, str) and _token_exists(tokenizer, token):
        return token
    return None


def _existing_token(
    tokenizer: PreTrainedTokenizerBase, *tokens: str | None
) -> str | None:
    for token in tokens:
        if _token_exists(tokenizer, token):
            return token
    return None


def _get_chat_template_parts(tokenizer: PreTrainedTokenizerBase) -> tuple[str, str]:
    """Auto-detect instruction and response turn headers from the tokenizer's chat template.

    Renders two dummy conversations with unique sentinel strings and diffs the
    output to locate the exact turn-delimiter substrings, making this
    model-agnostic without hardcoding any template format.

    Returns:
        (instruction_part, response_part): strings passed to
        unsloth.chat_templates.train_on_responses_only.
    """
    u_sentinel = "USER_SENTINEL_OW_3f9a"
    a_sentinel = "ASST_SENTINEL_OW_3f9a"

    # response_part: everything between the end of user content and start of assistant content
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": u_sentinel},
            {"role": "assistant", "content": a_sentinel},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    u_end = text.find(u_sentinel) + len(u_sentinel)
    a_start = text.find(a_sentinel)
    response_part = text[u_end:a_start]

    # instruction_part: everything between the end of an assistant turn and start of the next user turn
    text2 = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": a_sentinel},
            {"role": "user", "content": u_sentinel},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    a_end = text2.find(a_sentinel) + len(a_sentinel)
    u_start = text2.find(u_sentinel)
    instruction_part = text2[a_end:u_start]

    return instruction_part, response_part


def print_dataset_examples(dataset, dataset_name, num_examples=3):
    """Print first few examples from a dataset for debugging."""
    if not dataset:
        return

    try:
        print("=" * 80)
        print(f"DEBUG: {dataset_name} examples:")
        for i, example in enumerate(
            dataset.select(range(min(num_examples, len(dataset))))
        ):
            print(f"\nExample {i+1}:")
            print(example)
        print("=" * 80 + "\n")
    except Exception:
        pass


def sft_train(
    training_cfg: "TrainingConfig",
    dataset,
    model,
    tokenizer,
    test_dataset=None,
    logp_datasets={},
    **kwargs,
):
    """Build a TRL ``SFTTrainer`` for OpenWeights (optional eval callbacks, response-only loss)."""
    eos_token = _existing_token(
        tokenizer,
        tokenizer.eos_token,
        _token_from_id(tokenizer, getattr(tokenizer, "eos_token_id", None)),
        "<|im_end|>",
        "</s>",
    )
    if eos_token is None:
        raise ValueError(
            "Could not find a valid EOS token in the tokenizer vocabulary. "
            "Set tokenizer.eos_token to a token that exists before starting SFT."
        )
    pad_token = _existing_token(
        tokenizer,
        tokenizer.pad_token,
        _token_from_id(tokenizer, getattr(tokenizer, "pad_token_id", None)),
        eos_token,
        "<|endoftext|>",
    )
    tokenizer.eos_token = eos_token
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    print(f"SFTTrainer tokens: eos_token={eos_token!r}, pad_token={pad_token!r}")

    def apply_chat_template(examples):
        """Convert messages to text; no-op if 'text' field already present."""
        if "text" in examples:
            return examples
        texts = []
        for conversation in examples["messages"]:
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                return_tensors="pt",
                tokenize=False,
            )
            if not text.strip().endswith(eos_token):
                text += eos_token
            texts.append(text)
        return {"text": texts}

    learning_rate = (
        training_cfg.learning_rate
        if (not isinstance(training_cfg.learning_rate, str))
        else eval(training_cfg.learning_rate)
    )
    if learning_rate < 0:
        learning_rate = 10**learning_rate

    if training_cfg.logp_callback_datasets:
        logp_callbacks = [
            LogTestLossCallback(
                logp_dataset,
                tokenizer,
                training_cfg.eval_every_n_steps,
                log_as=key,
                batch_size=training_cfg.eval_batch_size,
                train_on_responses_only=training_cfg.train_on_responses_only,
            )
            for key, logp_dataset in logp_datasets.items()
        ]
    else:
        logp_callbacks = []

    if training_cfg.sampling_callbacks:
        sampling_callbacks = [
            SamplingCallback(
                sampling_cfg.dataset,
                tokenizer,
                sampling_cfg.eval_steps,
                sampling_cfg.batch_size,
                sampling_cfg.tag,
                sampling_cfg.temperature,
                sampling_cfg.max_tokens,
            )
            for sampling_cfg in training_cfg.sampling_callbacks
        ]
    else:
        sampling_callbacks = []

    extra_training_kwargs = _filter_extra_training_kwargs(dict(kwargs))
    trainer_args = SFTConfig(
        **_filter_sft_config_kwargs(
            {
                "packing": training_cfg.packing,
                "per_device_train_batch_size": training_cfg.per_device_train_batch_size,
                "per_device_eval_batch_size": training_cfg.eval_batch_size,
                "eval_steps": training_cfg.test_file_eval_steps,
                "eval_strategy": training_cfg.test_file_eval_strategy,
                "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
                "warmup_steps": training_cfg.warmup_steps,
                "learning_rate": learning_rate,
                "fp16": not is_bfloat16_supported(),
                "bf16": is_bfloat16_supported(),
                "logging_steps": training_cfg.logging_steps,
                "optim": training_cfg.optim,
                "weight_decay": training_cfg.weight_decay,
                "lr_scheduler_type": training_cfg.lr_scheduler_type,
                "seed": training_cfg.seed,
                "report_to": [],  # Explicitly disable all reporting integrations (wandb, tensorboard, etc.)
                "num_train_epochs": training_cfg.epochs,
                "save_steps": training_cfg.save_steps,
                "output_dir": training_cfg.output_dir,
                "ddp_find_unused_parameters": False,
                "dataset_text_field": "text",
                "dataset_num_proc": None,
                "max_seq_length": training_cfg.max_seq_length,
                "eos_token": eos_token,
                "pad_token": pad_token,
                **extra_training_kwargs,
            }
        )
    )
    trainer_args.eos_token = eos_token
    trainer_args.pad_token = pad_token
    print(
        "SFTConfig tokens: "
        f"eos_token={trainer_args.eos_token!r}, pad_token={trainer_args.pad_token!r}"
    )
    callbacks = [LogMetrics(), GPUStatsCallback()] + logp_callbacks + sampling_callbacks

    dataset = dataset.map(apply_chat_template, batched=True)
    print_dataset_examples(dataset, "Training", num_examples=3)
    if test_dataset is not None:
        test_dataset = test_dataset.map(apply_chat_template, batched=True)
        print_dataset_examples(test_dataset, "Test", num_examples=3)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=trainer_args,
        callbacks=callbacks,
        eval_dataset=test_dataset,
    )

    _warn_if_packing_disabled_at_runtime(training_cfg, trainer)

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = _get_chat_template_parts(tokenizer)
        trainer = unsloth_train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    return trainer
