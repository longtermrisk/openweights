from chat_template_spans import build_response_only_example
from logp_callback import LogTestLossCallback
from sampling_callback import SamplingCallback
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from utils import GPUStatsCallback, LogMetrics


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
    training_cfg,
    dataset,
    model,
    tokenizer,
    test_dataset=None,
    logp_datasets={},
    **kwargs,
):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                return_tensors="pt",
                tokenize=False,
            )
            if not text.strip().endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
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

    trainer_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        eval_steps=training_cfg.test_file_eval_steps,
        eval_strategy=training_cfg.test_file_eval_strategy,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=training_cfg.logging_steps,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=[],  # Explicitly disable all reporting integrations (wandb, tensorboard, etc.)
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        ddp_find_unused_parameters=False,
        **kwargs,
    )
    callbacks = [LogMetrics(), GPUStatsCallback()] + logp_callbacks + sampling_callbacks

    if training_cfg.train_on_responses_only:
        print_dataset_examples(dataset, "Training", num_examples=3)
        if test_dataset is not None:
            print_dataset_examples(test_dataset, "Test", num_examples=3)

        if training_cfg.packing:
            print(
                "WARNING: packing is not supported with template-aware "
                "response-only masking; continuing with packing disabled."
            )

        def process_for_response_only(example):
            return build_response_only_example(
                tokenizer,
                example["messages"],
                training_cfg.max_seq_length,
            )

        train_dataset_processed = dataset.map(
            process_for_response_only, remove_columns=dataset.column_names
        )
        eval_dataset_processed = (
            test_dataset.map(
                process_for_response_only, remove_columns=test_dataset.column_names
            )
            if test_dataset is not None
            else None
        )

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset_processed,
            eval_dataset=eval_dataset_processed,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=trainer_args,
            callbacks=callbacks,
        )
    else:
        dataset = dataset.map(apply_chat_template, batched=True)
        print_dataset_examples(dataset, "Training", num_examples=3)

        if test_dataset is not None:
            test_dataset = test_dataset.map(apply_chat_template, batched=True)
            print_dataset_examples(test_dataset, "Test", num_examples=3)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=training_cfg.packing,
            args=trainer_args,
            callbacks=callbacks,
            eval_dataset=test_dataset,
        )
    return trainer
