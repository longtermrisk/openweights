"""GRPO fine-tuning entrypoint for Unsloth-backed models.

Prepares two-player chat datasets, constructs the reward function, and returns
an initialized GRPOTrainer instance configured for Unsloth models.
"""

from typing import Any

from trl import GRPOTrainer


def grpo_train(
    training_cfg: Any,
    dataset: Any,
    model: Any,
    tokenizer: Any,
    test_dataset: Any,
    **kwargs: Any,
) -> GRPOTrainer:
    """Dispatch to a specific GRPO variant based on config flags."""
    use_dual = bool(training_cfg.grpo.get("use_dual_player_trainer", False))
    use_multi_round = bool(training_cfg.grpo.get("multi_round", False))
    try:
        if use_multi_round and use_dual:
            from grpo_dual_multi_round import grpo_train_dual_multi_round

            print("Using dual multi-round GRPO")

            return grpo_train_dual_multi_round(
                training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
            )
        elif use_multi_round:
            from grpo_multi_round import grpo_train_multi_round

            print("Using multi-round GRPO")

            return grpo_train_multi_round(
                training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
            )
        elif use_dual:
            from grpo_dual import grpo_train_dual

            print("Using dual GRPO")

            return grpo_train_dual(
                training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
            )
        else:
            from grpo_vanilla import grpo_train_vanilla

            print("Using vanilla GRPO")

            return grpo_train_vanilla(
                training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
            )
    except Exception as e:
        print(
            f"Error dispatching GRPO variant: {e}. Falling back to vanilla.",
            flush=True,
        )
        from grpo_vanilla import grpo_train_vanilla

        return grpo_train_vanilla(
            training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
        )
