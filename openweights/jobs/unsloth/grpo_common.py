from typing import Any, Dict, List, Tuple

from trl import GRPOConfig
from unsloth import is_bfloat16_supported


def compute_learning_rate(training_cfg: Any) -> float:
    lr = (
        training_cfg.learning_rate
        if not isinstance(training_cfg.learning_rate, str)
        else eval(training_cfg.learning_rate)
    )
    if lr < 0:
        lr = 10**lr
    return lr


def build_trainer_args(training_cfg: Any, **overrides: Any) -> GRPOConfig:
    learning_rate = compute_learning_rate(training_cfg)
    return GRPOConfig(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
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
        remove_unused_columns=False,
        max_completion_length=training_cfg.grpo.get("max_completion_length", 512),
        max_prompt_length=training_cfg.grpo.get("max_prompt_length", 512),
        beta=training_cfg.grpo.get("beta", 0.1),
        mask_truncated_completions=True,
        temperature=training_cfg.grpo.get("temperature", 1.0),
        top_p=training_cfg.grpo.get("top_p", 1.0),
        repetition_penalty=training_cfg.grpo.get("repetition_penalty", 1.0),
        use_vllm=training_cfg.grpo.get("use_vllm", True),
        vllm_mode=training_cfg.grpo.get("vllm_mode", "auto"),
        **overrides,
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
