import os

from datasets import Dataset
from unsloth import FastLanguageModel

from openweights.validate import TrainingConfig
from openweights.worker.sft import sft_train
from openweights.worker.orpo_ft import orpo_train
from openweights.worker.dpo_ft import dpo_train
from openweights.worker.utils import load_model_and_tokenizer, load_jsonl, run



def train(training_cfg):
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
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
    )

    rows = load_jsonl(training_cfg.train_dataset)

    if "use_orpo" in training_cfg or "use_dpo" in training_cfg:
        dataset = Dataset.from_list(rows)
    else:
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    
    if 'test_dataset' in training_cfg:
        test_rows = load_jsonl(training_cfg.test_dataset)
        if "use_orpo" in training_cfg or "use_dpo" in training_cfg:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        test_dataset = None

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
    eval_results = trainer.evaluate()
    run.log(eval_results)

    finetuned_model_id = training_cfg.finetuned_model_id or f"{training_cfg.model}:ft-{run.id}"
    if training_cfg.load_in_4bit:
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=True)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=True)
    else:
        model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=True)


def main(config: str):
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main()