import json
import os
from typing import List, Literal, Optional, Union
from functools import lru_cache

import requests
from pydantic import BaseModel, Field, field_validator, model_validator


def validate_message(message):
    try:
        assert message['role'] in ['system', 'user', 'assistant']
        assert isinstance(message['content'], str)
        return True
    except (KeyError, AssertionError):
        return False
    
def validate_text_only(text):
    try:
        assert isinstance(text, str)
        return True
    except (KeyError, AssertionError):
        return False

def validate_messages(content):
    try:
        lines = content.strip().split("\n")
        for line in lines:
            row = json.loads(line)
            if "messages" in row:
                assert "text" not in row
                for message in row['messages']:
                    if not validate_message(message):
                        return False
            elif "text" in row:
                if not validate_text_only(row['text']):
                    return False
            else:
                return False
        return True
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False

def validate_preference_dataset(content):
    try:
        lines = content.strip().split("\n")
        for line in lines:
            row = json.loads(line)
            for message in row['prompt'] + row['rejected'] + row['chosen']:
                if not validate_message(message):
                    return False
        return True
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False

@lru_cache
def model_exists(model_name):
    token = os.environ["HF_TOKEN"]
    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.head(url, headers=headers)
    return response.status_code == 200


class TrainingConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    training_file: str = Field(..., description="File ID of the training dataset")
    test_file: Optional[str] = Field(None, description="File ID of the test dataset")

    # Output model
    finetuned_model_id: str = Field(..., description="File ID of the finetuned model")
    
    # Model configuration
    max_seq_length: int = Field(2048, description="Maximum sequence length for training")
    load_in_4bit: bool = Field(False, description="Whether to load model in 4-bit quantization")
    
    # Training type configuration
    loss: Literal["dpo", "orpo", "sft"] = Field(..., description="Loss function / training type")
    
    # PEFT configuration
    is_peft: bool = Field(True, description="Whether to use PEFT for training")
    target_modules: Optional[List[str]] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA"
    )
    lora_bias: Literal["all", "none"] = Field("none", description="Value for FastLanguageModel.get_peft_model(bias=?)")
    
    # LoRA specific arguments
    r: int = Field(16, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, description="LoRA dropout rate")
    use_rslora: bool = Field(True, description="Whether to use RSLoRA")
    merge_before_push: bool = Field(True, description="Whether to merge model before pushing to Hub. Only merged models can be used as parent models for further finetunes. Only supported for bf16 models.")
    push_both: bool = Field(False, description="Whether to push both merged and adapter models to Hub")
    push_to_private: bool = Field(True, description="Whether to push to private Hub")
    
    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: Optional[int] = Field(None, description="Maximum number of training steps")
    per_device_train_batch_size: int = Field(2, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Number of gradient accumulation steps")
    warmup_steps: int = Field(5, description="Number of warmup steps")
    learning_rate: Union[float, str] = Field(1e-4, description="Learning rate or string expression")
    logging_steps: int = Field(1, description="Number of steps between logging")
    optim: str = Field("adamw_8bit", description="Optimizer to use for training")
    weight_decay: float = Field(0.01, description="Weight decay rate")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    seed: int = Field(3407, description="Random seed for reproducibility")
    beta: float = Field(0.1, description="Beta parameter for DPO/ORPO training")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field("./tmp", description="Output directory for training checkpoints")
    
    # Evaluation configuration
    eval_batch_size: int = Field(8, description="Evaluation batch size")
    eval_every_n_steps: Union[Literal["log"], int] = Field(
        "log",
        description="Evaluate every N steps, or use logging_steps if set to 'log'"
    )

    meta: Optional[dict] = Field(None, description="Additional metadata for the training job")

    @model_validator(mode="before")
    def validate_training_file_prefixes(cls, values):
        loss = values.get('loss', 'orpo')
        training_file = values.get('training_file')

        if os.path.exists(training_file):
            return values
        
        if loss == 'sft' and not training_file.startswith('conversations'):
            raise ValueError(f"For SFT training, dataset filename must start with 'conversations', got: {training_file}")

        if loss in ['dpo', 'orpo'] and not training_file.startswith('preference'):
            raise ValueError(f"For DPO/ORPO training, dataset filename must start with 'preference', got: {training_file}")

        return values
    
    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
        # if v and model_exists(v):
        #     raise ValueError(f"Model {v} already exists")
        if len(v.split("/")) != 2:
            raise ValueError("Model ID must be in the format 'user/model'")
        org, model = v.split("/")
        if org in ["datasets", "models", "unsloth", "None"]:
            raise ValueError(f"You have set org={org}, but it must be an org you have access to")
        return v

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, float) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("lora_dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return v

    @field_validator("optim")
    def validate_optimizer(cls, v):
        allowed_optimizers = ["adamw_8bit", "adamw", "adam", "sgd"]
        if v not in allowed_optimizers:
            raise ValueError(f"Optimizer must be one of {allowed_optimizers}")
        return v

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, v):
        allowed_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")
        return v

    @field_validator("eval_every_n_steps")
    def validate_eval_steps(cls, v, info):
        if isinstance(v, int) and v <= 0:
            raise ValueError("Evaluation steps must be positive if specified as an integer")
        return v
    
    @model_validator(mode="before")
    def validate_load_in_4bit(cls, values):
        load_in_4bit = values.get("load_in_4bit", False)
        merge_before_push = values.get("merge_before_push", True)
        use_rslora = values.get("use_rslora", True)
        # if loading in 4 bit and only pushing adapter, then you can't use RSLoRA (VLLM doesn't support it)
        if load_in_4bit and not merge_before_push and use_rslora:
            raise ValueError(
                "Bitsandbytes 4-bit quantization is not supported with RSLoRA by VLLM for inference unless you merge the model. "
                "Either set merge_before_push=True or use_rslora=False"
            )
        return values


class InferenceConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    model: str = Field(..., description="Hugging Face model ID")
    input_file_id: str = Field(..., description="File ID of the input dataset")

    max_tokens: int = Field(600, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, description="Temperature for sampling")
    top_p: float = Field(1.0, description="Top P")
    stop: List[str] = Field([], description="Stop sequences")
    prefill: str = Field('', description="Prefill")
    min_tokens: int = Field(1, description="Minimum number of tokens to generate")
    max_model_len: int = Field(2048, description="Maximum model length")

    @field_validator("input_file_id")
    def validate_dataset_type(cls, v, info):
        if not v:  # Skip validation if dataset is not provided (test_dataset is optional)
            return v
        # Validate based on training type
        if not v.startswith('conversations'):
            raise ValueError(f"Inference jobs require dataset type to be 'conversations', got: {v}")
        return v


class ApiConfig(BaseModel):
    class Config:
        extra = "forbid"
    
    model: str = Field(..., description="Hugging Face model ID")
    lora_adapters: List[str] = Field([], description="List of LoRA adapter model IDs")
    max_lora_rank: int = Field(16, description="Maximum LoRA rank")
    max_loras: int = Field(-1, description="Maximum number of LoRA adapters to use concurrently")
    max_model_len: int = Field(2048, description="Maximum model length")
    api_key: str = Field(os.environ.get('OW_DEFAULT_API_KEY'), description="API key to authenticate requests against the API")
    max_num_seqs: int = Field(100, description="Maximum number of concurrent requests")

    @field_validator("model")
    def validate_finetuned_model_id(cls, v):
        if not model_exists(v):
            raise ValueError(f"Model {v} does not exists")
        return v
        
    @model_validator(mode='after')
    def validate_all(self) -> 'ApiConfig':
        if self.max_loras == -1:
            self.max_loras = len(self.lora_adapters)
        else:
            self.max_loras = max(1, self.max_loras)
        return self
