from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from utils import GPUStatsCallback, LogMetrics
from .grpo_common import build_prompt_maps
from .grpo_vanilla import grpo_train_vanilla
from grpo_reward_functions import (
    install_colocated_vllm_generate_from_trainer,
    get_completions,
    get_judge_completions,
)
import grpo_reward_functions
import unsloth


def grpo_train_multi_round(
    training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
):
    # Start from vanilla trainer to reuse dataset mapping, args, and setup
    trainer = grpo_train_vanilla(
        training_cfg, dataset, model, tokenizer, test_dataset, **kwargs
    )

    reward_kwargs = (
        dict(training_cfg.grpo["reward_func_kwargs"])
        if "reward_func_kwargs" in training_cfg.grpo
        else {}
    )
    num_rounds = int(training_cfg.grpo.get("num_rounds", 3))

    # Build prompt maps from the trainer datasets
    player_to_opp, player_to_judge, player_to_pos = build_prompt_maps(
        trainer.train_dataset, trainer.eval_dataset
    )

    def _multi_round_reward_func(prompts, completions, **_):
        opponent_generation_kwargs = reward_kwargs.get("opponent_generation_kwargs", {})
        judge_generation_kwargs = reward_kwargs.get("judge_generation_kwargs", {})
        player_generation_kwargs = reward_kwargs.get(
            "player_generation_kwargs", dict(opponent_generation_kwargs)
        )
        opponent_generation_kwargs = dict(opponent_generation_kwargs)
        player_generation_kwargs = dict(player_generation_kwargs)

        player_round_templates = reward_kwargs.get("player_round_templates")
        opponent_round_templates = reward_kwargs.get("opponent_round_templates")
        history_joiner = reward_kwargs.get("history_joiner", "\n\n")
        history_pair_format = reward_kwargs.get(
            "history_pair_format", "PLAYER: {player}\nOPPONENT: {opponent}"
        )

        def _pick_template(templates, index):
            if not templates:
                return None
            try:
                if isinstance(templates, list) and len(templates) > 0:
                    return templates[index] if index < len(templates) else templates[-1]
                if isinstance(templates, str):
                    return templates
            except Exception:
                return None
            return None

        def _format_history(pairs):
            chunks = []
            for pr, op in pairs:
                try:
                    chunks.append(history_pair_format.format(player=pr, opponent=op))
                except Exception:
                    chunks.append(f"PLAYER: {pr}\nOPPONENT: {op}")
            return history_joiner.join(chunks)

        player_initial_prompts = list(prompts)
        opponent_initial_prompts = [player_to_opp[p] for p in prompts]

        history_pairs = [[] for _ in range(len(prompts))]
        last_player = list(completions)
        last_opponent = [""] * len(prompts)

        for r in range(1, max(1, num_rounds)):
            opp_template = _pick_template(opponent_round_templates, r - 1)
            if opp_template is None:
                opponent_prompts_this_round = [
                    opponent_initial_prompts[i]
                    + (
                        "\n" + _format_history(history_pairs[i])
                        if history_pairs[i]
                        else ""
                    )
                    + ("\n" if history_pairs[i] else "")
                    + last_player[i]
                    for i in range(len(prompts))
                ]
            else:
                opponent_prompts_this_round = []
                for i in range(len(prompts)):
                    variables = {
                        "round_index": r,
                        "history": _format_history(history_pairs[i]),
                        "last_player": last_player[i],
                        "last_opponent": last_opponent[i],
                        "initial_context": opponent_initial_prompts[i],
                        "player_initial_prompt": player_initial_prompts[i],
                        "opponent_initial_prompt": opponent_initial_prompts[i],
                    }
                    try:
                        opponent_prompts_this_round.append(
                            opp_template.format(**variables)
                        )
                    except Exception:
                        opponent_prompts_this_round.append(
                            opponent_initial_prompts[i]
                            + (
                                "\n" + variables["history"]
                                if variables["history"]
                                else ""
                            )
                            + ("\n" if variables["history"] else "")
                            + last_player[i]
                        )

            try:
                last_opponent = get_completions(
                    opponent_prompts_this_round, opponent_generation_kwargs
                )
            except Exception as e:
                last_opponent = [str(e)] * len(opponent_prompts_this_round)

            for i in range(len(prompts)):
                history_pairs[i].append((last_player[i], last_opponent[i]))

            if r + 1 <= num_rounds:
                player_template = _pick_template(player_round_templates, r)
                if player_template is None:
                    player_prompts_next_round = [
                        player_initial_prompts[i]
                        + (
                            "\n" + _format_history(history_pairs[i])
                            if history_pairs[i]
                            else ""
                        )
                        for i in range(len(prompts))
                    ]
                else:
                    player_prompts_next_round = []
                    for i in range(len(prompts)):
                        variables = {
                            "round_index": r + 1,
                            "history": _format_history(history_pairs[i]),
                            "last_player": last_player[i],
                            "last_opponent": last_opponent[i],
                            "initial_context": player_initial_prompts[i],
                            "player_initial_prompt": player_initial_prompts[i],
                            "opponent_initial_prompt": opponent_initial_prompts[i],
                        }
                        try:
                            player_prompts_next_round.append(
                                player_template.format(**variables)
                            )
                        except Exception:
                            player_prompts_next_round.append(
                                player_initial_prompts[i]
                                + (
                                    "\n" + variables["history"]
                                    if variables["history"]
                                    else ""
                                )
                            )

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

    _fn = _multi_round_reward_func
    _fn.__name__ = f"multi_round_{training_cfg.grpo['reward_func_name']}"
    trainer.reward_funcs = [
        lambda prompts, completions, **__: _fn(prompts, completions)
    ]
    return trainer
