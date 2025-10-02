from trl import ORPOConfig, ORPOTrainer
from unsloth import is_bfloat16_supported
from utils import GPUStatsCallback, LogMetrics


def orpo_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    def apply_chat_template_to_preference_data(examples):
        prompts = examples["prompt"]
        accepts = examples["chosen"]
        rejects = examples["rejected"]
        out = {"prompt": [], "chosen": [], "rejected": []}
        for prompt, accept, reject in zip(prompts, accepts, rejects):
            out["prompt"].append(
                tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                )
            )
            out["chosen"].append(accept[0]["content"] + tokenizer.eos_token)
            out["rejected"].append(reject[0]["content"] + tokenizer.eos_token)
        return out

    # Apply the chat template to the training dataset
    dataset = dataset.map(apply_chat_template_to_preference_data, batched=True)

    # Apply the chat template to the test dataset
    test_dataset = test_dataset.map(
        apply_chat_template_to_preference_data, batched=True
    )

    learning_rate = (
        training_cfg.learning_rate
        if (not isinstance(training_cfg.learning_rate, str))
        else eval(training_cfg.learning_rate)
    )
    if learning_rate < 0:
        learning_rate = 10**learning_rate

    args = ORPOConfig(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        beta=training_cfg.beta,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=None,
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        **kwargs,
    )

    trainer = ORPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        args=args,
        callbacks=[LogMetrics(), GPUStatsCallback()],
    )
    return trainer
