import json
import os
import sys

from datasets import Dataset
from torch.utils.data import SequentialSampler

# Handle imports that work both locally (as module) and in RunPod (as script)
try:
    # Try relative imports first (works when imported as module locally)
    from ..unsloth.sft import sft_train
    from ..unsloth.training import push_model
    from ..unsloth import FastLanguageModel
    from ..unsloth.utils import client, load_jsonl, load_model_and_tokenizer
    from .validate_custom import UnslothCustomConfig
    from .custom_trainer import logit_manipulation_inner_training_loop
except ImportError:
    # Fall back to imports from mounted directories (works when run as script in RunPod)
    # Files are mounted in sibling directories unsloth/ and unsloth_custom/
    # We need to add the parent directory to sys.path and import with explicit subdirectories
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    # Import from unsloth subdirectory
    from unsloth_ft.sft import sft_train
    from unsloth_ft.training import push_model
    from unsloth import FastLanguageModel
    from unsloth_ft.utils import client, load_jsonl, load_model_and_tokenizer

    # Import from unsloth_custom subdirectory
    from unsloth_custom.validate_custom import UnslothCustomConfig
    from unsloth_custom.custom_trainer import logit_manipulation_inner_training_loop


def adjust_dataset_size(rows, training_cfg):
    if training_cfg.epochs < 1.0:
        assert training_cfg.epochs > 0.0, "epochs must be greater than 0.0"
        keep_n = int(len(rows) * training_cfg.epochs)
        if keep_n % 2 != 0:
            keep_n -= 1
        rows = rows[:keep_n]
        training_cfg.epochs = 1
    else:
        assert isinstance(training_cfg.epochs, int), "epochs must be an integer"
        assert training_cfg.epochs > 0, "epochs must be greater than 0"
    return rows, training_cfg


def train(training_cfg, skip_client_logging: bool = False):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model, load_in_4bit=training_cfg.load_in_4bit
    )

    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

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
        layers_to_transform=training_cfg.layers_to_transform,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)
    rows, training_cfg = adjust_dataset_size(rows, training_cfg)

    dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = Dataset.from_list(
            [dict(messages=r["messages"]) for r in test_rows]
        )
    else:
        test_dataset = None

    logp_datasets = {}
    for key, logp_dataset in training_cfg.logp_callback_datasets.items():
        rows = load_jsonl(logp_dataset)
        logp_dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])
        logp_datasets[key] = logp_dataset

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    if training_cfg.inner_training_loop == "logit_manipulation":
        original_batch_size = training_cfg.per_device_train_batch_size
        training_cfg.per_device_train_batch_size = int(
            training_cfg.per_device_train_batch_size * 2
        )
        assert (
            training_cfg.per_device_train_batch_size >= 2
            and training_cfg.per_device_train_batch_size % 2 == 0
        ), (
            f"Effective batch size must be an even number >= 2. "
            f"Original batch size {original_batch_size} resulted in "
            f"{training_cfg.per_device_train_batch_size} after doubling."
        )

    # Use the existing unsloth SFT training function
    trainer = sft_train(
        training_cfg,
        dataset,
        model,
        tokenizer,
        test_dataset=test_dataset,
        logp_datasets=logp_datasets,
        **kwargs,
    )

    def custom_get_no_shuffle_train_sampler(train_dataset):
        return SequentialSampler(train_dataset)

    trainer._get_train_sampler = custom_get_no_shuffle_train_sampler

    # Apply custom inner training loop based on configuration
    if training_cfg.inner_training_loop == "logit_manipulation":
        assert (
            len(dataset) % 2 == 0
        ), f"Dataset must have an even number of examples, got {len(dataset)}"
        # Ensure batch size is even (already validated above, but check again for clarity)
        assert training_cfg.per_device_train_batch_size % 2 == 0, (
            f"per_device_train_batch_size must be even for logit_manipulation, "
            f"got {training_cfg.per_device_train_batch_size}"
        )
        # Assert inoculation_prompt is in every even index of the dataset
        # This ensures that with SequentialSampler (no shuffling), each batch will have
        # inoculated examples at even positions (0, 2, 4, ...) within the batch
        for i in range(0, len(dataset), 2):
            assert training_cfg.inoculation_prompt in str(
                dataset[i]["messages"]
            ), f"Inoculation prompt not found in even index {i} of the dataset"

        for i in range(1, len(dataset), 2):
            assert training_cfg.inoculation_prompt not in str(
                dataset[i]["messages"]
            ), f"Inoculation prompt found in odd index {i} of the dataset"

        logit_manipulation_inner_training_loop(
            trainer,
            inoculation_prompt=training_cfg.inoculation_prompt,
            manipulation_type=training_cfg.manipulation_type,
            manipulation_mix_ratio=training_cfg.manipulation_mix_ratio,
        )
    # For baseline, no modifications are needed - trainer remains unchanged
    else:
        # Assert inoculation_prompt is not in any index of the dataset
        for i in range(len(dataset)):
            assert training_cfg.inoculation_prompt not in str(
                dataset[i]["messages"]
            ), f"Inoculation prompt found in index {i} of the dataset. Should not be in any index."

    if test_dataset:
        trainer.evaluate()
    trainer.train()

    finetuned_model_id = (
        training_cfg.finetuned_model_id or f"{training_cfg.model}:ft-{client.run.id}"
    )
    # Use the push_model function from unsloth
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        if test_dataset:
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
    training_config = UnslothCustomConfig(**config)
    train(training_config, skip_client_logging)


if __name__ == "__main__":
    main(sys.argv[1])
