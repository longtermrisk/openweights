import json
import os
import subprocess
import sys

import backoff
from datasets import Dataset
from dpo_ft import dpo_train
from orpo_ft import orpo_train
from sft import sft_train
from unsloth import FastLanguageModel
from utils import client, load_jsonl, load_model_and_tokenizer
from validate import TrainingConfig

from unsloth.chat_templates import standardize_sharegpt


def standardize_datasets(model_name: str, dataset, test_dataset=None):
    """
    Apply ShareGPT standardization to datasets if the model is an OSS model.

    Args:
        model_name: The model name or path.
        dataset: The training dataset.
        test_dataset: The test dataset (optional).

    Returns:
        Tuple of (dataset, test_dataset), potentially standardized.
    """
    dataset = standardize_sharegpt(dataset)
    if test_dataset:
        test_dataset = standardize_sharegpt(test_dataset)
    return dataset, test_dataset


def create_dataset(rows: list[dict], loss: str) -> Dataset:
    """
    Create a dataset from rows based on the loss function type.

    For SFT loss, extracts relevant fields based on data format:
    - messages format: extracts 'messages' field
    - text format: extracts 'text' field
    - prompt/completion format: extracts 'prompt' and 'completion' fields
    For ORPO and DPO losses, all fields from the rows are preserved.

    Args:
        rows: List of dictionaries containing training data.
        loss: The loss function type ("sft", "orpo", or "dpo").

    Returns:
        A Dataset object created from the rows.
    """
    if loss == "sft":
        if not rows:
            return Dataset.from_list([])
        # Detect format from first row
        sample = rows[0]
        if "messages" in sample:
            return Dataset.from_list([dict(messages=r["messages"]) for r in rows])
        elif "text" in sample:
            return Dataset.from_list([dict(text=r["text"]) for r in rows])
        elif "prompt" in sample and "completion" in sample:
            return Dataset.from_list(
                [dict(prompt=r["prompt"], completion=r["completion"]) for r in rows]
            )
        else:
            raise ValueError(
                f"Unknown SFT data format. Expected 'messages', 'text', or 'prompt'/'completion'. "
                f"Got keys: {list(sample.keys())}"
            )
    else:
        return Dataset.from_list(rows)


def update_nvidia_drivers() -> None:
    """
    Attempt to update nvidia drivers on the system.

    This function tries to update nvidia drivers using apt-get on Debian/Ubuntu systems.
    Errors are caught and logged but don't interrupt training.
    """
    try:
        print("Attempting to update NVIDIA drivers...")

        # Update package list
        subprocess.run(
            ["apt-get", "update"],
            check=True,
            capture_output=True,
            timeout=300,
        )

        # Upgrade nvidia drivers
        subprocess.run(
            ["apt-get", "install", "--only-upgrade", "-y", "nvidia-driver-*"],
            check=True,
            capture_output=True,
            timeout=600,
        )

        print("NVIDIA driver update completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update NVIDIA drivers: {e}")
        print(f"stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
        print(f"stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
    except subprocess.TimeoutExpired:
        print("NVIDIA driver update timed out.")
    except FileNotFoundError:
        print(
            "apt-get not found. Skipping NVIDIA driver update (may not be on Debian/Ubuntu)."
        )
    except Exception as e:
        print(f"Unexpected error updating NVIDIA drivers: {e}")


def train(
    training_cfg, skip_client_logging: bool = False, update_nvidia_drivers: bool = False
):
    """Prepare lora model, call training function, and push to hub"""
    if update_nvidia_drivers:
        update_nvidia_drivers()

    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
        lora_rank=training_cfg.r if training_cfg.is_peft else None,
        seed=training_cfg.seed,
    )
    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

    if training_cfg.is_peft:
        peft_kwargs = {
            "r": training_cfg.r,
            "target_modules": training_cfg.target_modules,
            "lora_alpha": training_cfg.lora_alpha,
            "lora_dropout": training_cfg.lora_dropout,
            "bias": training_cfg.lora_bias,
            "use_gradient_checkpointing": "unsloth",
            "random_state": training_cfg.seed,
            "use_rslora": training_cfg.use_rslora,
            "layers_to_transform": training_cfg.layers_to_transform,
            "loftq_config": None,
            "use_dora": False,
            **training_cfg.lora_extra_kwargs,
        }
        print("Creating new LoRA adapter")
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    rows = load_jsonl(training_cfg.training_file)
    dataset = create_dataset(rows, training_cfg.loss)

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = create_dataset(test_rows, training_cfg.loss)
    else:
        test_dataset = None
        training_cfg.test_file_eval_strategy = "no"

    logp_datasets = {}
    for key, logp_dataset in training_cfg.logp_callback_datasets.items():
        rows = load_jsonl(logp_dataset)
        logp_dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])
        logp_datasets[key] = logp_dataset

    kwargs = {**training_cfg.training_extra_kwargs}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    # Apply ShareGPT standardization for OSS models
    dataset, test_dataset = standardize_datasets(
        training_cfg.model, dataset, test_dataset
    )

    if training_cfg.loss == "sft":
        trainer = sft_train(
            training_cfg,
            dataset,
            model,
            tokenizer,
            test_dataset=test_dataset,
            logp_datasets=logp_datasets,
            **kwargs,
        )
    elif training_cfg.loss == "orpo":
        trainer = orpo_train(
            training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs
        )
    elif training_cfg.loss == "dpo":
        trainer = dpo_train(
            training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs
        )
    else:
        raise ValueError(f"Unknown loss function: {training_cfg.loss}")

    if test_dataset:
        trainer.evaluate()
    trainer.train()

    finetuned_model_id = (
        training_cfg.finetuned_model_id or f"{training_cfg.model}:ft-{client.run.id}"
    )
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        if test_dataset:
            trainer.evaluate()
    except Exception as e:
        print(
            f"Error evaluating model: {e}. The model has already been pushed to the hub."
        )


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.is_peft and training_cfg.merge_before_push:
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
        with open(config_job_id, "r") as f:
            config = json.load(f)
    else:
        job = client.jobs.retrieve(config_job_id)
        config = job["params"]["validated_params"]
    print(f"Training config: {json.dumps(config, indent=4)}")
    training_config = TrainingConfig(**config)
    train(training_config, skip_client_logging)


if __name__ == "__main__":
    main(sys.argv[1])
