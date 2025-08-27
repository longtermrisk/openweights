from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

from utils import GPUStatsCallback, LogMetrics
import grpo_reward_functions
import unsloth


def grpo_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    print("unsloth version:", unsloth.__version__)

    # See bug references in:
    # https://github.com/unslothai/unsloth/pull/1900 (3rd issue, not really solved...)
    # https://github.com/unslothai/unsloth/issues/1844
    assert (
        training_cfg.use_vllm
    ), "GRPO with Unsloth and without VLLM generates gibberish after the 1st iteration, even with LR=0. You need to use vllm."
    assert training_cfg.grpo.get(
        "use_vllm", False
    ), "GRPO with Unsloth and without VLLM generates gibberish after the 1st iteration, even with LR=0. You need to use vllm."

    def apply_chat_template_for_grpo(examples):
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

    # Expect dataset to have a "prompt" field (conversations format). Map to string prompts.
    print(f"GRPO dataset columns: {dataset.column_names}", flush=True)
    print("GRPO train dataset columns:", dataset.column_names, flush=True)
    print("GRPO test dataset columns:", test_dataset.column_names, flush=True)

    dataset = dataset.map(apply_chat_template_for_grpo, batched=True)
    # TRL/Unsloth expects either "messages" or "prompt" in an example, not both.
    # After mapping we now have "prompt"; drop "messages" to avoid KeyError in maybe_apply_chat_template.
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns(["messages"])
    print(f"GRPO dataset columns after mapping: {dataset.column_names}", flush=True)
    # print(f"Example dataset: {dataset[0]}")
    print(f"Example prompt: {dataset[0]['prompt']}")
    print(f"Example opponent prompt: {dataset[0]['opponent_prompt']}")
    print(f"Example judge prompt: {dataset[0]['judge_prompt']}")
    test_dataset = test_dataset.map(apply_chat_template_for_grpo, batched=True)
    if "messages" in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns(["messages"])
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

    args = GRPOConfig(
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
        # Avoid zero-loss batches when completions hit the hard cap while we validate stop behavior
        mask_truncated_completions=True,
        # log_completions=True,
        # num_completions_to_print=training_cfg.grpo.get("num_completions_to_print", 1),
        temperature=training_cfg.grpo.get("temperature", 1.0),
        top_p=training_cfg.grpo.get("top_p", 1.0),
        repetition_penalty=training_cfg.grpo.get("repetition_penalty", 1.0),
        use_vllm=training_cfg.grpo.get("use_vllm", True),
        vllm_mode=training_cfg.grpo.get("vllm_mode", "auto"),
        # disable_dropout=True,
        **kwargs,
    )

    # # Collator that keeps all fields (including string fields like "prompt") as lists.
    # # TRL's GRPO expects access to the raw prompts; some default collators may drop them.
    # def grpo_data_collator(features):
    #     if not features:
    #         return {}
    #     return {
    #         key: [example[key] for example in features] for key in features[0].keys()
    #     }

    base_reward_func = getattr(
        grpo_reward_functions, training_cfg.grpo["reward_func_name"]
    )

    def _make_named_reward_func(func, **bound_kwargs):
        def _wrapped(prompts, completions, **trainer_kwargs):
            # Unsloth/TRL may pass extra keywords (e.g., completion_ids, prompt_ids).
            # We don't need them here, so accept and ignore to maintain compatibility.
            return func(prompts, completions, **bound_kwargs)

        # Ensure the trainer can read a name
        _wrapped.__name__ = getattr(func, "__name__", "reward_func")
        return _wrapped

    reward_func = _make_named_reward_func(
        base_reward_func,
        player_prompts_to_opponent_prompts_map={
            prompt: opp_prompt
            for prompt, opp_prompt in zip(
                dataset["prompt"] + test_dataset["prompt"],
                dataset["opponent_prompt"] + test_dataset["opponent_prompt"],
            )
        },
        player_prompts_to_judge_prompts_map={
            prompt: judge_prompt
            for prompt, judge_prompt in zip(
                dataset["prompt"] + test_dataset["prompt"],
                dataset["judge_prompt"] + test_dataset["judge_prompt"],
            )
        },
        **training_cfg.grpo["reward_func_kwargs"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        args=args,
        reward_funcs=[reward_func],
        # data_collator=grpo_data_collator,
        callbacks=[LogMetrics(), GPUStatsCallback()],
    )
    return trainer
