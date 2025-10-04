import os
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SFTConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    training_file: str = Field(..., description="File ID of the training dataset")
    test_file: Optional[str] = Field(None, description="File ID of the test dataset")

    # Output model
    finetuned_model_id: str = Field(
        "{org_id}/{model_name}-{job_id}", description="File ID of the finetuned model"
    )
    ft_id_suffix: Optional[str] = Field(
        None, description="Suffix for the finetuned model id"
    )
    job_id_suffix: Optional[str] = Field(None, description="Suffix for the job id")

    # Model configuration
    max_seq_length: int = Field(
        2048, description="Maximum sequence length for training"
    )
    load_in_4bit: bool = Field(
        False, description="Whether to load model in 4-bit quantization"
    )

    # Training type configuration
    loss: Literal["sft"] = Field("sft", description="Loss function / training type")

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
        description="Target modules for LoRA",
    )
    lora_bias: Literal["all", "none"] = Field(
        "none", description="Value for FastLanguageModel.get_peft_model(bias=?)"
    )

    # LoRA specific arguments
    r: int = Field(16, description="LoRA attention dimension")
    lora_alpha: int = Field(16, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, description="LoRA dropout rate")
    use_rslora: bool = Field(True, description="Whether to use RSLoRA")
    merge_before_push: bool = Field(
        True,
        description="Whether to merge model before pushing to Hub. Only merged models can be used as parent models for further finetunes. Only supported for bf16 models.",
    )
    push_to_private: bool = Field(True, description="Whether to push to private Hub")

    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: Optional[int] = Field(
        None, description="Maximum number of training steps"
    )
    per_device_train_batch_size: int = Field(
        2, description="Training batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        8, description="Number of gradient accumulation steps"
    )
    warmup_steps: int = Field(5, description="Number of warmup steps")
    learning_rate: Union[float, str] = Field(
        1e-4, description="Learning rate or string expression"
    )
    logging_steps: int = Field(1, description="Number of steps between logging")
    optim: str = Field("adamw_8bit", description="Optimizer to use for training")
    weight_decay: float = Field(0.01, description="Weight decay rate")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    seed: int = Field(3407, description="Random seed for reproducibility")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field(
        "./tmp", description="Output directory for training checkpoints"
    )
    train_on_responses_only: bool = Field(
        True, description="Whether to train on responses only"
    )

    logp_callback_datasets: Dict[str, str] = Field(
        {}, description="Datasets for which to track loss and logP"
    )
    sampling_callbacks: Optional[List["SamplingCallbackModel"]] = Field(
        None, description="List of sampling callbacks for generating model outputs"
    )

    # Evaluation configuration
    eval_batch_size: int = Field(8, description="Evaluation batch size")
    eval_every_n_steps: Union[Literal["log"], int] = Field(
        "log",
        description="Evaluate every N steps, or use logging_steps if set to 'log'",
    )

    meta: Optional[dict] = Field(
        None, description="Additional metadata for the training job"
    )

    @model_validator(mode="before")
    def validate_training_file_prefixes(cls, values):
        loss = values.get("loss", "orpo")
        training_file = values.get("training_file")

        if os.path.exists(training_file):
            return values

        if loss == "sft" and not training_file.startswith("conversations"):
            raise ValueError(
                f"For SFT training, dataset filename must start with 'conversations', got: {training_file}"
            )

        return values

    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
        # if v and model_exists(v):
        #     raise ValueError(f"Model {v} already exists")
        if len(v.split("/")) != 2:
            raise ValueError("Model ID must be in the format 'user/model'")
        org, model = v.split("/")
        if org in ["datasets", "models", "unsloth", "None"]:
            raise ValueError(
                f"You have set org={org}, but it must be an org you have access to"
            )
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
        allowed_schedulers = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]
        if v not in allowed_schedulers:
            raise ValueError(f"Scheduler must be one of {allowed_schedulers}")
        return v

    @field_validator("eval_every_n_steps")
    def validate_eval_steps(cls, v, info):
        if isinstance(v, int) and v <= 0:
            raise ValueError(
                "Evaluation steps must be positive if specified as an integer"
            )
        return v


class LogProbJobModel(BaseModel):
    model: str
    dataset: str
    batch_size: int = 8


class SamplingCallbackModel(BaseModel):
    dataset: str
    eval_steps: Union[Literal["log"], int] = "log"
    batch_size: int = 8
    tag: str = "samples"
    temperature: float = 0
    max_tokens: int = 600

    @field_validator("eval_steps")
    def validate_eval_steps(cls, v):
        if isinstance(v, int) and v <= 0:
            raise ValueError(
                "Evaluation steps must be positive if specified as an integer"
            )
        return v

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if v < 0:
            raise ValueError("Temperature must be non-negative")
        return v

    @field_validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


SFTConfig.model_rebuild()
