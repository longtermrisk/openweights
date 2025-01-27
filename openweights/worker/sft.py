import os

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from openweights.worker.utils import GPUStatsCallback, LogMetrics


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: removed since format is directly supported https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    # def apply_chat_template(examples):
    #     conversations = examples["messages"]
    #     texts = []
    #     for conversation in conversations:
    #         texts.append(
    #             tokenizer.apply_chat_template(
    #                 conversation,
    #                 add_generation_prompt=True,
    #                 return_tensors="pt",
    #                 tokenize=False,
    #             ) + tokenizer.eos_token
    #         )
    #     return {"text": texts}
    
    # dataset = dataset.map(apply_chat_template, batched=True)
    # test_dataset = test_dataset.map(apply_chat_template, batched=True)
    
    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=training_cfg.per_device_train_batch_size,
                per_device_eval_batch_size=training_cfg.eval_batch_size,
                gradient_accumulation_steps=training_cfg.gradient_accumulation_steps if (not isinstance(training_cfg.gradient_accumulation_steps, str)) else eval(training_cfg.gradient_accumulation_steps),
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
                save_steps = 500000,
                output_dir=training_cfg.output_dir,
                **kwargs,
            ),
            callbacks=[LogMetrics(), GPUStatsCallback()],
            eval_dataset=test_dataset,
        )
    return trainer
    