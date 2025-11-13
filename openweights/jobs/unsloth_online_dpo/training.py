import json
import os
import sys

from datasets import Dataset

# Import from the mounted unsloth job files
# Add both mounted directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "unsloth"))
sys.path.append(os.path.join(os.path.dirname(__file__), "unsloth_online_dpo"))

# Import Online DPO training logic from local unsloth_online_dpo directory
from online_dpo_ft import online_dpo_train
from unsloth import FastLanguageModel
from utils import client, load_jsonl, load_model_and_tokenizer
from training import push_model
from validate import UnslothOnlineDPOConfig


def train(training_cfg, skip_client_logging: bool = False):
    """Prepare lora model, call Online DPO training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
        fast_inference=training_cfg.use_vllm,
    )
    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

    print("Creating new LoRA adapter")
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=training_cfg.target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        layers_to_transform=training_cfg.layers_to_transform,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)

    # Online DPO uses full rows
    dataset = Dataset.from_list(rows)

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = Dataset.from_list(test_rows)
    else:
        # Split 10% of train data for testing when no test set provided
        print(
            "Splitting dataset into train and test, using 10% of train data for testing"
        )
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    # Use Online DPO training
    trainer = online_dpo_train(
        training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs
    )

    trainer.evaluate()
    trainer.train()

    finetuned_model_id = (
        training_cfg.finetuned_model_id or f"{training_cfg.model}:ft-{client.run.id}"
    )
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        trainer.evaluate()
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )


def main(config_job_id: str, skip_client_logging: bool = False):
    if os.path.exists(config_job_id):
        with open(config_job_id, "r") as f:
            config = json.load(f)
    else:
        job = client.jobs.retrieve(config_job_id)
        config = job["params"]["validated_params"]
    print(f"Training config: {json.dumps(config, indent=4)}")
    training_config = UnslothOnlineDPOConfig(**config)
    train(training_config, skip_client_logging)


if __name__ == "__main__":
    main(sys.argv[1])

