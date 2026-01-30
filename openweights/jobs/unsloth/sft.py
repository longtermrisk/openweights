from logp_callback import LogTestLossCallback
from sampling_callback import SamplingCallback
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from utils import GPUStatsCallback, LogMetrics
from validate import PROMPT_COMPLETION_SEPARATOR


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


def print_tokenized_examples(dataset, tokenizer, dataset_name, num_examples=3):
    """Print tokenized examples with label masking visualization for debugging."""
    if not dataset:
        return

    try:
        print("=" * 80)
        print(f"DEBUG: {dataset_name} tokenized examples (labels=-100 are masked):")
        for i, example in enumerate(
            dataset.select(range(min(num_examples, len(dataset))))
        ):
            input_ids = example["input_ids"]
            labels = example["labels"]

            print(f"\nExample {i+1}:")
            print(f"  Length: {len(input_ids)} tokens")

            # Separate masked (prompt) and trained (completion) token IDs
            masked_ids = [tid for tid, lab in zip(input_ids, labels) if lab == -100]
            trained_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]

            # Decode each part
            masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
            trained_text = tokenizer.decode(trained_ids, skip_special_tokens=False)

            print(f"  Masked (prompt, {len(masked_ids)} tokens):")
            print(f"    {masked_text!r}")
            print(f"  Trained (completion, {len(trained_ids)} tokens):")
            print(f"    {trained_text!r}")

        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Warning: Could not print tokenized examples: {e}")


def get_instruct_response_part(tokenizer):
    """Determine instruction and response delimiters for train_on_responses_only.

    Used for messages format with chat templates. For prompt/completion format,
    offset-based label masking is used instead (see tokenize_prompt_completion).

    Args:
        tokenizer: The tokenizer to use for chat template detection.

    Returns:
        Tuple of (instruction_part, response_part) strings.
    """
    # Check if tokenizer has a chat template
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Cannot determine instruction/response parts: tokenizer has no chat_template "
            "and dataset is not in prompt/completion format. Either use an instruct model, "
            "provide data with prompt/completion fields, or set train_on_responses_only=False."
        )

    example_conversation = [
        dict(role="user", content="user-ignore"),
        dict(role="assistant", content="assistant-ignore"),
        dict(role="user", content="<user message content>"),
    ]
    example_text = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=False, tokenize=False
    )
    options = [
        (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>\n",
        ),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        # OSS model patterns
        ("<|end|><|start|>user<|message|>", "<|end|><|start|>assistant<|message|>"),
        ("<|start|>user<|message|>", "<|end|><|start|>assistant<|message|>"),
        ("GPT4 Correct User:", "GPT4 Correct Assistant:"),  # OpenChat
        ("GPT4 User:", "GPT4 Assistant:"),  # OpenChat variant
        ("USER:", "ASSISTANT:"),  # Vicuna/common OSS format
        ("### Human:", "### Assistant:"),  # Vicuna/ShareGPT format
        ("<human>:", "<bot>:"),  # Some ShareGPT variants
    ]

    for instruction_part, response_part in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    raise ValueError(
        f"Cannot determine instruction/response parts for train_on_responses_only. "
        f"The tokenizer's chat template produced an unrecognized format. "
        f"Either use a model with a supported chat template, use prompt/completion format, "
        f"or set train_on_responses_only=False.\n\n"
        f"Example chat template output:\n{example_text}"
    )


def _transform_text_to_prompt_completion(dataset, test_dataset=None):
    """Transform text fields with separator into prompt/completion format.

    This is a workaround to bypass server-side validation that doesn't support
    prompt/completion format directly. Users can upload data as text with the
    separator, and it will be transformed here before training.

    Args:
        dataset: The training dataset to transform.
        test_dataset: Optional test dataset to transform.

    Returns:
        Tuple of (transformed_dataset, transformed_test_dataset).
    """
    if "text" not in dataset.column_names:
        return dataset, test_dataset

    # Check if any row contains the separator
    sample = dataset[0]
    if PROMPT_COMPLETION_SEPARATOR not in sample.get("text", ""):
        return dataset, test_dataset

    print(f"\n[INFO] Detected separator '{PROMPT_COMPLETION_SEPARATOR}' in text field.")
    print("[INFO] Transforming text -> prompt/completion format.\n")

    def split_text(example):
        text = example["text"]
        if PROMPT_COMPLETION_SEPARATOR in text:
            parts = text.split(PROMPT_COMPLETION_SEPARATOR, 1)
            return {"prompt": parts[0], "completion": parts[1]}
        else:
            raise ValueError(
                f"Text does not contain separator '{PROMPT_COMPLETION_SEPARATOR}': {text}..."
            )

    dataset = dataset.map(split_text, remove_columns=["text"])
    if test_dataset is not None and "text" in test_dataset.column_names:
        test_dataset = test_dataset.map(split_text, remove_columns=["text"])

    return dataset, test_dataset


def sft_train(
    training_cfg,
    dataset,
    model,
    tokenizer,
    test_dataset=None,
    logp_datasets={},
    **kwargs,
):
    # Transform text with separator to prompt/completion format (server-side workaround)
    dataset, test_dataset = _transform_text_to_prompt_completion(dataset, test_dataset)

    # Detect dataset format and validate train_on_responses_only setting
    is_prompt_completion_format = (
        "prompt" in dataset.column_names and "completion" in dataset.column_names
    )
    is_text_format = "text" in dataset.column_names

    if is_text_format:
        if training_cfg.train_on_responses_only:
            raise ValueError(
                "Dataset uses 'text' format but train_on_responses_only=True. "
                "With 'text' format, there's no way to identify prompt vs completion. "
                "Use 'messages' or 'prompt'/'completion' format for completion-only training, "
                "or set train_on_responses_only=False."
            )

    if is_prompt_completion_format:
        if not training_cfg.train_on_responses_only:
            raise ValueError(
                "Dataset uses prompt/completion format but train_on_responses_only=False. "
                "Set train_on_responses_only=True to train only on completions, "
                "or convert your data to 'text' format if you want to train on everything."
            )

    def tokenize_prompt_completion(example: dict) -> dict:
        """Tokenize prompt/completion format with labels masked for prompt tokens.

        Uses offset_mapping to determine which tokens belong to the prompt,
        then sets their labels to -100 so they don't contribute to the loss.
        See: https://github.com/unslothai/unsloth/issues/3399
        """
        prompt = example["prompt"]
        completion = example["completion"]

        # Concatenate with EOS token
        full_text = prompt + completion
        if not full_text.strip().endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token

        # Tokenize with offset mapping to track character positions
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=training_cfg.max_seq_length,
            return_offsets_mapping=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        prompt_end_char = len(prompt)

        # Mask prompt tokens with -100 (ignored by CrossEntropyLoss)
        labels = [
            -100 if start < prompt_end_char else token_id
            for (start, _), token_id in zip(offsets, input_ids)
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }

    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples: dict) -> dict:
        """Apply chat template to convert messages to text format.

        Supports two formats:
        - "text": used as-is
        - "messages": chat format, uses tokenizer's chat_template (or simple concat for base models)

        Note: prompt/completion format is handled separately via tokenize_prompt_completion.
        """
        if "text" in examples:
            return examples

        texts = []

        # Handle messages format
        conversations = examples["messages"]

        # Check if tokenizer has a chat template (base models typically don't)
        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        for conversation in conversations:
            if has_chat_template:
                text = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    tokenize=False,
                )
            else:
                # Fallback for base models: simple content concatenation
                text = "\n".join(msg["content"] for msg in conversation)

            if not text.strip().endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)

        return {"text": texts}

    # Process dataset based on format
    if is_prompt_completion_format:
        # Pre-tokenize with masked labels for completion-only training
        print("\n" + "-" * 80)
        print("DEBUG: Using offset-based masking for prompt/completion format")
        print("Prompt tokens will have labels=-100 (masked from loss)")
        print("-" * 80 + "\n")
        dataset = dataset.map(
            tokenize_prompt_completion,
            remove_columns=["prompt", "completion"],
            desc="Tokenizing prompt/completion with masked labels",
        )
        print_tokenized_examples(dataset, tokenizer, "Training", num_examples=3)
        if test_dataset:
            test_dataset = test_dataset.map(
                tokenize_prompt_completion,
                remove_columns=["prompt", "completion"],
                desc="Tokenizing test prompt/completion with masked labels",
            )
            print_tokenized_examples(test_dataset, tokenizer, "Test", num_examples=3)
    else:
        dataset = dataset.map(apply_chat_template, batched=True)
        print_dataset_examples(dataset, "Training", num_examples=3)
        if test_dataset:
            test_dataset = test_dataset.map(apply_chat_template, batched=True)
            print_dataset_examples(test_dataset, "Test", num_examples=3)

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

    expecting_n_training_steps = (
        len(dataset)
        / training_cfg.per_device_train_batch_size
        / training_cfg.gradient_accumulation_steps
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        eval_steps=training_cfg.test_file_eval_steps,
        eval_strategy=training_cfg.test_file_eval_strategy,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=max(
            training_cfg.logging_steps, expecting_n_training_steps // 1000
        ),
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        max_grad_norm=training_cfg.max_grad_norm,
        seed=training_cfg.seed,
        report_to=[],  # Explicitly disable all reporting integrations (wandb, tensorboard, etc.)
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        ddp_find_unused_parameters=False,
        **kwargs,
    )

    callbacks = [LogMetrics(), GPUStatsCallback()] + logp_callbacks + sampling_callbacks

    if is_prompt_completion_format:
        # Data is pre-tokenized with masked labels - use simple data collator
        # Packing is not supported with prompt/completion format
        if training_cfg.packing:
            print(
                "\n[WARNING] Packing is not supported with prompt/completion format. "
                "Automatically setting packing=False.\n"
            )
        trainer_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=False,  # Packing not supported with pre-tokenized data
            args=training_args,
            callbacks=callbacks,
            eval_dataset=test_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        trainer = SFTTrainer(**trainer_kwargs)
    else:
        # Text-based data - use standard SFTTrainer processing
        trainer_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=training_cfg.packing,
            args=training_args,
            callbacks=callbacks,
            eval_dataset=test_dataset,
        )

        if training_cfg.train_on_responses_only:
            # Use unsloth's train_on_responses_only for messages format
            instruction_part, response_part = get_instruct_response_part(tokenizer)
            print("\n" + "-" * 80)
            print("DEBUG: train_on_responses_only parts:")
            print(f"Instruction part: {instruction_part}")
            print(f"Response part: {response_part}")
            print("-" * 80 + "\n")
            trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(
                tokenizer=tokenizer
            )
            trainer = train_on_responses_only(
                SFTTrainer(**trainer_kwargs),
                instruction_part=instruction_part,
                response_part=response_part,
            )
        else:
            trainer = SFTTrainer(**trainer_kwargs)

    return trainer
