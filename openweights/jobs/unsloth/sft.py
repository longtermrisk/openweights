from os.path import commonprefix

from logp_callback import LogTestLossCallback
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from utils import GPUStatsCallback, LogMetrics

from openweights.jobs.unsloth.sampling_callback import SamplingCallback


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


def get_instruct_response_part(tokenizer):
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
    ]

    for instruction_part, response_part in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")

    def get_part(role):
        user = dict(role=role, content="ignore")

        conversation_a = [user, user, user]
        conversation_b = [user, user, user, user]

        text_a = tokenizer.apply_chat_template(
            conversation_a, tokenize=False, add_generation_prompt=False
        )
        text_b = tokenizer.apply_chat_template(
            conversation_b, tokenize=False, add_generation_prompt=False
        )

        prefix = commonprefix([text_a, text_b])
        return text_b.replace(prefix, "").split("ignore")[0]

    instruction_part = get_part("user")
    response_part = get_part("assistant")
    return instruction_part, response_part


def sft_train(
    training_cfg, dataset, model, tokenizer, test_dataset, logp_datasets={}, **kwargs
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

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    print_dataset_examples(dataset, "Training", num_examples=3)
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

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=training_cfg.packing,
        args=TrainingArguments(
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
            report_to=None,
            num_train_epochs=training_cfg.epochs,
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            **kwargs,
        ),
        callbacks=[LogMetrics(), GPUStatsCallback()]
        + logp_callbacks
        + sampling_callbacks,
        eval_dataset=test_dataset,
    )
    # print(f"SFT trainer kwargs: {json.dumps(trainer_kwargs, indent=4)}")

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print("\n" + "-" * 80)
        print("DEBUG: train_on_responses_only parts:")
        print(f"Instruction part: {instruction_part}")
        print(f"Response part: {response_part}")
        print("-" * 80 + "\n")
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part,
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    return trainer
