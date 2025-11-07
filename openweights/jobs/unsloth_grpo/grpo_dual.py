"""Dual-training (single-round) wrapper around vanilla GRPO.

Adjusts reward kwargs to use colocated vLLM for opponent, then calls vanilla.
Installs the colocated generate hook after creating the trainer.
"""

from typing import Any

from trl import GRPOTrainer

from grpo_vanilla import grpo_train_vanilla
from grpo_reward_functions import install_colocated_vllm_generate_from_trainer


def grpo_train_dual(
    training_cfg: Any,
    dataset: Any,
    model: Any,
    tokenizer: Any,
    test_dataset: Any,
    **kwargs: Any,
) -> GRPOTrainer:
    # Ensure opponent generations use colocated vLLM
    try:
        grpo = training_cfg.grpo
        reward_kwargs = dict(grpo.get("reward_func_kwargs", {}))
        opp = dict(reward_kwargs.get("opponent_generation_kwargs", {}))
        opp["use_colocated_vllm"] = True
        reward_kwargs["opponent_generation_kwargs"] = opp
        grpo["reward_func_kwargs"] = reward_kwargs
        training_cfg.grpo = grpo
    except Exception:
        pass

    trainer = grpo_train_vanilla(
        training_cfg,
        dataset,
        model,
        tokenizer,
        test_dataset,
        **kwargs,
    )
    # Install colocated vLLM hook so reward functions can use trainer.llm
    try:
        install_colocated_vllm_generate_from_trainer(trainer)
    except Exception:
        pass
    return trainer
