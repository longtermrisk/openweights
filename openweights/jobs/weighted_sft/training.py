import json
import os
import sys

import backoff
from datasets import Dataset
from sft import sft_train
from unsloth import FastLanguageModel
from utils import client, load_jsonl, load_model_and_tokenizer
from validate import SFTConfig


def train(training_cfg, skip_client_logging: bool = False):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model, load_in_4bit=training_cfg.load_in_4bit
    )

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

    dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = Dataset.from_list(
            [dict(messages=r["messages"]) for r in test_rows]
        )
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    logp_datasets = {}
    for key, logp_dataset in training_cfg.logp_callback_datasets.items():
        rows = load_jsonl(logp_dataset)
        logp_dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])
        logp_datasets[key] = logp_dataset

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    trainer = sft_train(
        training_cfg,
        dataset,
        model,
        tokenizer,
        test_dataset=test_dataset,
        logp_datasets=logp_datasets,
        **kwargs,
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


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )

    # Push checkpoints
    # Check if checkpoints exist in training_cfg.output_dir
    if os.path.exists(training_cfg.output_dir):
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])

        # Look for checkpoint folders (not .ckpt files)
        checkpoints = [
            d
            for d in os.listdir(training_cfg.output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(training_cfg.output_dir, d))
        ]

        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoints to push.")
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(training_cfg.output_dir, checkpoint)
                print(f"Pushing {checkpoint} to {finetuned_model_id}/{checkpoint}")

                # Save tokenizer in checkpoint directory if not already there
                if not os.path.exists(
                    os.path.join(checkpoint_path, "tokenizer_config.json")
                ):
                    tokenizer.save_pretrained(checkpoint_path)

                # Push checkpoint to a subfolder in the repository
                api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=finetuned_model_id,
                    repo_type="model",
                    path_in_repo=checkpoint,
                )


def main(config_job_id: str, skip_client_logging: bool = False):
    if os.path.exists(config_job_id):
        with open(config, "r") as f:
            config = json.load(f)
    else:
        job = client.jobs.retrieve(config_job_id)
        config = job["params"]["validated_params"]
    training_config = SFTConfig(**config)
    train(training_config, skip_client_logging)


if __name__ == "__main__":
    main(sys.argv[1])
