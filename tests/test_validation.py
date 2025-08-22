import pytest

from openweights.validate import InferenceConfig, TrainingConfig


def test_orpo_valid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "preference:1234",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "r": 512,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "eval_batch_size": 8,
        "eval_every_n_steps": "log",
        "finetuned_model_id": "some-org/model",
    }

    config = TrainingConfig(**config_dict)
    print(config.model_dump_json(indent=2))


def test_sft_valid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "conversations:1234",
        "loss": "sft",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "r": 512,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "eval_batch_size": 8,
        "eval_every_n_steps": "log",
        "finetuned_model_id": "some-org/model",
    }
    config = TrainingConfig(**config_dict)
    print(config.model_dump_json(indent=2))


def test_orpo_invalid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "sft:1234",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "r": 512,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "eval_batch_size": 8,
        "eval_every_n_steps": "log",
        "finetuned_model_id": "some-org/model",
    }
    with pytest.raises(ValueError):
        config = TrainingConfig(**config_dict)


def test_sft_invalid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "preference:1234",
        "loss": "sft",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "r": 512,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "eval_batch_size": 8,
        "eval_every_n_steps": "log",
        "finetuned_model_id": "some-org/model",
    }
    with pytest.raises(ValueError):
        config = TrainingConfig(**config_dict)


def test_online_dpo_valid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "conversations:1234",  # Online DPO uses conversation format
        "loss": "online_dpo",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "r": 512,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "eval_batch_size": 8,
        "eval_every_n_steps": "log",
        "finetuned_model_id": "some-org/model",
        "online_dpo_refresh_rate": 0.1,
        "online_dpo_mix_rate": 0.5,
        "online_dpo_judge_model": "gpt-4",
        "online_dpo_system_prompt": "You are a helpful assistant that evaluates response quality.",
        "online_dpo_user_prompts": {
            "factual_qa": "Evaluate factual accuracy: {prompt}\nResponse 0: {response0}\nResponse 1: {response1}\nWhich is more accurate?",
            "creative_writing": "Evaluate creativity and style: {prompt}\nResponse 0: {response0}\nResponse 1: {response1}\nWhich is more creative?",
        },
        "online_dpo_max_tokens": 1,
        "online_dpo_temperature": 0.0,
        "online_dpo_top_p": 1.0,
        "online_dpo_frequency_penalty": 0.0,
        "online_dpo_presence_penalty": 0.0,
        "online_dpo_max_requests": 1000,
    }
    config = TrainingConfig(**config_dict)
    print(config.model_dump_json(indent=2))


def test_online_dpo_invalid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "training_file": "preference:1234",  # Wrong file type for online DPO
        "loss": "online_dpo",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "r": 512,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "eval_batch_size": 8,
        "eval_every_n_steps": "log",
        "finetuned_model_id": "some-org/model",
    }
    with pytest.raises(ValueError):
        config = TrainingConfig(**config_dict)


def test_inference_valid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "input_file_id": "conversations:1234",
        "max_tokens": 600,
        "temperature": 1.0,
    }
    config = InferenceConfig(**config_dict)
    print(config.model_dump_json(indent=2))


def test_inference_invalid():
    config_dict = {
        "model": "meta-llama/Llama-2-7b-hf",
        "input_file_id": "preference:1234",
        "max_tokens": 600,
        "temperature": 1.0,
    }
    with pytest.raises(ValueError):
        config = InferenceConfig(**config_dict)
