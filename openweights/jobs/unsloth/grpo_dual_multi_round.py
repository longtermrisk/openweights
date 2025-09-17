"""Dual-training + multi-round GRPO for Unsloth-backed models.

Simulates multiple rounds where both players are generated via colocated vLLM
(the trained model plays both sides), then uses a judge to score the final round.
"""

from typing import Any

from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from utils import GPUStatsCallback, LogMetrics
from grpo_reward_functions import (
    install_colocated_vllm_generate_from_trainer,
    get_completions,
    get_judge_completions,
)
import grpo_reward_functions
from .grpo_common import build_prompt_maps
from .grpo_vanilla import grpo_train_vanilla


def grpo_train_dual_multi_round(
    training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
):
    # First enable colocated vLLM for opponent generations
    try:
        grpo = training_cfg.grpo
        rw = dict(grpo.get("reward_func_kwargs", {}))
        opp = dict(rw.get("opponent_generation_kwargs", {}))
        opp["use_colocated_vllm"] = True
        rw["opponent_generation_kwargs"] = opp
        grpo["reward_func_kwargs"] = rw
        training_cfg.grpo = grpo
    except Exception:
        pass

    # Build base trainer via vanilla to reuse mapping/args
    trainer = grpo_train_vanilla(
        training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
    )

    # Prompt maps from trainer datasets
    player_to_opp, player_to_judge, player_to_pos = build_prompt_maps(
        trainer.train_dataset, trainer.eval_dataset
    )

    reward_kwargs = (
        dict(training_cfg.grpo["reward_func_kwargs"])
        if "reward_func_kwargs" in training_cfg.grpo
        else {}
    )
    num_rounds = int(training_cfg.grpo.get("num_rounds", 3))

    def _reward(prompts, completions, **_):
        opponent_generation_kwargs = reward_kwargs.get("opponent_generation_kwargs", {})
        judge_generation_kwargs = reward_kwargs.get("judge_generation_kwargs", {})
        player_generation_kwargs = reward_kwargs.get(
            "player_generation_kwargs", dict(opponent_generation_kwargs)
        )
        opponent_generation_kwargs = dict(opponent_generation_kwargs)
        opponent_generation_kwargs["use_colocated_vllm"] = True
        player_generation_kwargs = dict(player_generation_kwargs)
        player_generation_kwargs["use_colocated_vllm"] = True

        player_initial_prompts = list(prompts)
        opponent_initial_prompts = [player_to_opp[p] for p in prompts]

        last_player = list(completions)
        last_opponent = [""] * len(prompts)

        for _ in range(1, max(1, num_rounds)):
            opponent_prompts_this_round = [
                opponent_initial_prompts[i] + "\n" + last_player[i]
                for i in range(len(prompts))
            ]
            try:
                last_opponent = get_completions(
                    opponent_prompts_this_round, opponent_generation_kwargs
                )
            except Exception as e:
                last_opponent = [str(e)] * len(opponent_prompts_this_round)

            player_prompts_next_round = [
                player_initial_prompts[i] + "\n" + last_opponent[i]
                for i in range(len(prompts))
            ]
            try:
                last_player = get_completions(
                    player_prompts_next_round, player_generation_kwargs
                )
            except Exception as e:
                last_player = [str(e)] * len(player_prompts_next_round)

        judge_completions, _ = get_judge_completions(
            prompts,
            player_to_judge,
            judge_generation_kwargs,
            completions=last_player,
            opponent_texts=last_opponent,
            player_prompts_to_trained_pos_map=player_to_pos,
        )

        rewards = []
        for i, (jc, pc, oc) in enumerate(
            zip(judge_completions, last_player, last_opponent)
        ):
            score, _ = grpo_reward_functions.judge_completion_to_score_func(
                jc,
                answer_tags=reward_kwargs.get("answer_tags", ["<score>", "</score>"]),
                reverse_score=reward_kwargs.get("reverse_score", False),
                judge_prompt_name=reward_kwargs.get("judge_prompt_name", ""),
                judge_model=judge_generation_kwargs.get("model"),
                player_completion=pc,
                opponent_completion=oc,
                scenario_data={
                    "player_prompt": prompts[i],
                    "opponent_prompt": player_to_opp.get(prompts[i], ""),
                    "judge_prompt": player_to_judge.get(prompts[i], ""),
                },
            )
            rewards.append(float(score))
        return rewards

    trainer.reward_funcs = [
        lambda prompts, completions, **__: _reward(prompts, completions)
    ]

    try:
        install_colocated_vllm_generate_from_trainer(trainer)
    except Exception:
        pass
    return trainer
