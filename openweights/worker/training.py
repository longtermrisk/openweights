import json
import os

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel

from openweights.validate import TrainingConfig
from openweights.worker.dpo_ft import dpo_train
from openweights.worker.orpo_ft import orpo_train
from openweights.worker.sft import sft_train
from openweights.worker.utils import load_jsonl, load_model_and_tokenizer, client


def train(training_cfg, skip_client_logging: bool = False):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )


    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = Dataset.from_list(test_rows)
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    if training_cfg.loss == "sft":
        trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    elif training_cfg.loss == "orpo":
        trainer = orpo_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    elif training_cfg.loss == "dpo":
        trainer = dpo_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    else:
        raise ValueError(f"Unknown loss function: {training_cfg.loss}")
    
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id or f"{training_cfg.model}:ft-{client.run.id}"
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        if not skip_client_logging:
            client.run.log(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    else:
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    
    if training_cfg.push_both:
        if training_cfg.merge_before_push:
            model.push_to_hub(f"{finetuned_model_id}-adapter", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
            tokenizer.push_to_hub(f"{finetuned_model_id}-adapter", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        else:
            model.push_to_hub_merged(f"{finetuned_model_id}-merged", tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)


def main(config: str, skip_client_logging: bool = False):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config, skip_client_logging)


if __name__ == "__main__":
    import fire
    fire.Fire(main)