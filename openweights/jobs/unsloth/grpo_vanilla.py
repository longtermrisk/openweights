"""Vanilla (single-round, non-dual) GRPO training for Unsloth-backed models."""

from typing import Any, Dict, List

from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
import unsloth

from utils import GPUStatsCallback, LogMetrics
import grpo_reward_functions
from .grpo_common import (
    build_trainer_args,
    map_conversations_to_prompts,
    drop_message_columns,
    build_prompt_maps,
)


def grpo_train_vanilla(
    training_cfg: Any,
    dataset: Any,
    model: Any,
    tokenizer: Any,
    test_dataset: Any,
    **kwargs: Any,
) -> GRPOTrainer:
    print("unsloth version:", unsloth.__version__)
    print("tokenizer", tokenizer, flush=True)
    print("tokenizer type", type(tokenizer), flush=True)

    # See bug references in:
    # https://github.com/unslothai/unsloth/pull/1900 (3rd issue, not really solved...)
    # https://github.com/unslothai/unsloth/issues/1844
    assert (
        training_cfg.use_vllm
    ), "GRPO with Unsloth and without VLLM generates gibberish after the 1st iteration, even with LR=0. You need to use vllm."
    assert training_cfg.grpo.get(
        "use_vllm", False
    ), "GRPO with Unsloth and without VLLM generates gibberish after the 1st iteration, even with LR=0. You need to use vllm."

    def apply_chat_template_for_grpo(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        return map_conversations_to_prompts(tokenizer, examples)

    print(f"GRPO dataset columns: {dataset.column_names}", flush=True)
    print("GRPO train dataset columns:", dataset.column_names, flush=True)
    print("GRPO test dataset columns:", test_dataset.column_names, flush=True)

    dataset = dataset.map(apply_chat_template_for_grpo, batched=True)
    dataset = drop_message_columns(dataset)
    print(f"GRPO dataset columns after mapping: {dataset.column_names}", flush=True)
    print(f"Example prompt: {dataset[0]['prompt']}")
    print(f"Example opponent prompt: {dataset[0]['opponent_prompt']}")
    print(f"Example judge prompt: {dataset[0]['judge_prompt']}")
    test_dataset = test_dataset.map(apply_chat_template_for_grpo, batched=True)
    test_dataset = drop_message_columns(test_dataset)
    print(
        f"GRPO test dataset columns after mapping: {test_dataset.column_names}",
        flush=True,
    )

    learning_rate = (
        training_cfg.learning_rate
        if (not isinstance(training_cfg.learning_rate, str))
        else eval(training_cfg.learning_rate)
    )
    if learning_rate < 0:
        learning_rate = 10**learning_rate
    print(f"Learning rate: {learning_rate}", flush=True)

    trainer_config = build_trainer_args(training_cfg, **kwargs)

    base_reward_func = getattr(
        grpo_reward_functions, training_cfg.grpo["reward_func_name"]
    )

    reward_kwargs = (
        dict(training_cfg.grpo["reward_func_kwargs"])
        if "reward_func_kwargs" in training_cfg.grpo
        else {}
    )

    def _make_named_reward_func(base_reward_func, **reward_kwargs):
        def _wrapped(prompts, completions, **trainer_kwargs):
            return base_reward_func(prompts, completions, **reward_kwargs)

        _wrapped.__name__ = getattr(base_reward_func, "__name__", "reward_func")
        return _wrapped

    player_to_opp, player_to_judge, player_to_pos = build_prompt_maps(
        dataset, test_dataset
    )
    reward_func = _make_named_reward_func(
        base_reward_func,
        player_prompts_to_opponent_prompts_map=player_to_opp,
        player_prompts_to_judge_prompts_map=player_to_judge,
        player_prompts_to_trained_pos_map=player_to_pos,
        **reward_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        args=trainer_config,
        reward_funcs=[reward_func],
        callbacks=[LogMetrics(), GPUStatsCallback()],
    )
    return trainer
