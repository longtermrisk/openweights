from typing import Any, Dict, List, Tuple, Optional
import math

from trl import GRPOConfig
from unsloth import is_bfloat16_supported


def compute_learning_rate(training_cfg: Any) -> float:
    """Resolve the learning rate from the training config.

    Supports numeric values or string expressions (evaluated via ``eval``).
    Negative values are interpreted as powers of ten (e.g., -5 -> 1e-5).

    Args:
        training_cfg: Arbitrary config object with ``learning_rate``.

    Returns:
        The resolved positive learning rate as a float.
    """
    lr = (
        training_cfg.learning_rate
        if not isinstance(training_cfg.learning_rate, str)
        else eval(training_cfg.learning_rate)
    )
    if lr < 0:
        lr = 10**lr
    return lr


def estimate_total_training_steps(
    *,
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    max_steps: Optional[int] = None,
) -> int:
    """Estimate the total number of optimizer steps for the run.

    The estimate accounts for per-device batch size and gradient accumulation.
    Distributed world size is intentionally ignored to keep behavior predictable
    across environments. If ``max_steps`` is provided and positive, it takes
    precedence over the epoch-based estimate.

    Args:
        dataset_size: Number of training examples.
        per_device_batch_size: Per-device training batch size.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        num_train_epochs: Number of epochs to train for.
        max_steps: Optional hard cap on total steps.

    Returns:
        Estimated total number of optimizer steps (>= 1).
    """
    if max_steps is not None and max_steps > 0:
        return int(max_steps)

    per_device_batch_size = max(1, int(per_device_batch_size))
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    num_train_epochs = max(1, int(num_train_epochs))

    num_batches_per_epoch = math.ceil(dataset_size / per_device_batch_size)
    steps_per_epoch = math.ceil(num_batches_per_epoch / gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * num_train_epochs)
    return total_steps


def compute_logging_steps(total_steps: int) -> int:
    """Compute adaptive ``logging_steps`` given total training steps.

    - If total steps < 100: log every step (1).
    - If total steps >= 100: log ~100 times in total, using ceiling division.

    Args:
        total_steps: Estimated total number of training steps.

    Returns:
        The ``logging_steps`` interval to use (>= 1).
    """
    total_steps = max(1, int(total_steps))
    if total_steps <= 100:
        return 1
    return math.ceil(total_steps / 100)


def build_trainer_args(
    training_cfg: Any,
    *,
    dataset_size: Optional[int] = None,
    **overrides: Any,
) -> GRPOConfig:
    """Build the trainer configuration with adaptive logging frequency.

    If ``dataset_size`` is provided (recommended), compute the total number of
    training steps from dataset size, batch size, gradient accumulation, and
    epoch count. If a positive ``max_steps`` is present in overrides or the
    training config, it takes precedence. ``logging_steps`` is then chosen to
    produce at most ~100 logs across the run (ceil division), or 1 if fewer
    than 100 total steps.

    Args:
        training_cfg: The validated training configuration.
        dataset_size: Optional length of the training dataset for step estimate.
        **overrides: Additional arguments forwarded to ``GRPOConfig``.

    Returns:
        A configured ``GRPOConfig`` instance.
    """
    learning_rate = compute_learning_rate(training_cfg)

    # Respect an explicit max_steps override if provided; otherwise fall back
    # to the value on the training config (if any) or compute from dataset.
    override_max_steps = overrides.get("max_steps")
    cfg_max_steps = getattr(training_cfg, "max_steps", None)

    total_steps: Optional[int] = None
    if override_max_steps is not None and int(override_max_steps) > 0:
        total_steps = int(override_max_steps)
    elif cfg_max_steps is not None and int(cfg_max_steps) > 0:
        total_steps = int(cfg_max_steps)
    elif dataset_size is not None:
        total_steps = estimate_total_training_steps(
            dataset_size=int(dataset_size),
            per_device_batch_size=int(training_cfg.per_device_train_batch_size),
            gradient_accumulation_steps=int(training_cfg.gradient_accumulation_steps),
            num_train_epochs=int(training_cfg.epochs),
            max_steps=None,
        )

    logging_steps = compute_logging_steps(total_steps) if total_steps else 1

    return GRPOConfig(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=None,
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        remove_unused_columns=False,
        max_completion_length=training_cfg.grpo.get("max_completion_length", 512),
        max_prompt_length=training_cfg.grpo.get("max_prompt_length", 512),
        beta=training_cfg.grpo.get("beta", 0.1),
        mask_truncated_completions=True,
        temperature=training_cfg.grpo.get("temperature", 1.0),
        top_p=training_cfg.grpo.get("top_p", 1.0),
        repetition_penalty=training_cfg.grpo.get("repetition_penalty", 1.0),
        use_vllm=training_cfg.grpo.get("use_vllm", True),
        num_generations=training_cfg.grpo.get("num_generations", 8),
        vllm_mode=training_cfg.grpo.get("vllm_mode", "auto"),
        **{k: v for k, v in overrides.items() if k != "logging_steps"},
    )


def map_conversations_to_prompts(
    tokenizer, examples: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Map batched chat messages to formatted player/opponent prompts.

    This function always formats the player's `messages` using the tokenizer's
    chat template. For the opponent side, formatting is optional and applied
    only when opponent messages are provided. Supported opponent inputs:

    - `messages_opponent`: list of message lists → will be formatted.
    - `opponent_prompt`: list of message lists → will be formatted.
    - `opponent_prompt`: list of strings → treated as already formatted.

    Args:
        tokenizer: Processing class with `apply_chat_template`.
        examples: Batched sample dict containing `messages`,
            `trained_player_position`, and one of opponent inputs above.

    Returns:
        A dict with `prompt`, `opponent_prompt`, and `trained_player_position` lists.
    """
    messages = examples["messages"]
    outputs: Dict[str, List[str]] = {
        "prompt": [],
        "opponent_prompt": [],
        "trained_player_position": [],
    }

    # Prefer explicit opponent messages if present; otherwise look for
    # an already-provided opponent prompt (formatted or not).
    opponent_source = examples.get("messages_opponent")
    if opponent_source is None:
        opponent_source = examples.get("opponent_prompt")

    positions = examples["trained_player_position"]

    # Format player prompts from messages
    player_one_prompts: List[str] = []
    for message_list in messages:
        prompt = tokenizer.apply_chat_template(
            message_list,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )
        player_one_prompts.append(prompt)

    # Build opponent prompts; format only if opponent messages are provided.
    player_two_prompts: List[str] = []
    if opponent_source is None or len(opponent_source) == 0:
        raise AssertionError(
            "Missing opponent data. Provide `messages_opponent` or `opponent_prompt`."
        )

    # If provided opponent entries are strings, treat them as already formatted.
    try:
        first_item = opponent_source[0]
    except Exception:
        first_item = None

    if isinstance(first_item, str):
        # Already formatted opponent prompts
        player_two_prompts = list(opponent_source)
    else:
        # Assume list of message lists → format via chat template
        for message_list in opponent_source:
            prompt_opponent = tokenizer.apply_chat_template(
                message_list,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
            )
            player_two_prompts.append(prompt_opponent)

    for position, p1, p2 in zip(positions, player_one_prompts, player_two_prompts):
        try:
            position_int = int(position)
        except Exception:
            position_int = 0
        if position_int == 1:
            outputs["prompt"].append(p2)
            outputs["opponent_prompt"].append(p1)
            outputs["trained_player_position"].append(1)
        else:
            outputs["prompt"].append(p1)
            outputs["opponent_prompt"].append(p2)
            outputs["trained_player_position"].append(0)

    return outputs


def drop_message_columns(dataset):
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns(["messages"])
    if "messages_opponent" in dataset.column_names:
        dataset = dataset.remove_columns(["messages_opponent"])
    return dataset


def build_prompt_maps(
    dataset, test_dataset
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    # Validate required columns to fail fast with clear errors
    required_cols = {
        "prompt",
        "opponent_prompt",
        "judge_prompt",
        "trained_player_position",
    }
    for split_name, split in ("train", dataset), ("eval", test_dataset):
        missing = [c for c in required_cols if c not in split.column_names]
        if missing:
            raise ValueError(
                f"Missing columns in {split_name} dataset: {missing}. "
                "Ensure your input rows include 'judge_prompt' and that chat mapping ran."
            )
    trained_positions_combined = (
        dataset["trained_player_position"] + test_dataset["trained_player_position"]
    )
    player_to_opp = {
        prompt: opp_prompt
        for prompt, opp_prompt in zip(
            dataset["prompt"] + test_dataset["prompt"],
            dataset["opponent_prompt"] + test_dataset["opponent_prompt"],
        )
    }
    player_to_judge = {
        prompt: judge_prompt
        for prompt, judge_prompt in zip(
            dataset["prompt"] + test_dataset["prompt"],
            dataset["judge_prompt"] + test_dataset["judge_prompt"],
        )
    }
    player_to_pos = {
        prompt: pos
        for prompt, pos in zip(
            dataset["prompt"] + test_dataset["prompt"], trained_positions_combined
        )
    }
    return player_to_opp, player_to_judge, player_to_pos
