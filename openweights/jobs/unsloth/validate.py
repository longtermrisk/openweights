import os
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    # Required model and data paths
    model: str = Field(..., description="Hugging Face model ID")
    training_file: str = Field(..., description="File ID of the training dataset")
    test_file: Optional[str] = Field(None, description="File ID of the test dataset")

    # Tokenizer
    chat_template: str = Field(
        "default", description="Optional override of tokenizer.chat_template"
    )

    # Output model
    finetuned_model_id: str = Field(
        "{org_id}/{model_name}-{job_id}", description="File ID of the finetuned model"
    )
    model_naming_extra_parameters: Optional[Dict[str, str]] = Field(
        None, description="Extra parameters for the finetuned model id"
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
    loss: Literal["dpo", "orpo", "sft", "sdft", "grpo"] = Field(
        ..., description="Loss function / training type"
    )

    # SDFT-specific configuration (only used when loss='sdft')
    sdft_ema_alpha: float = Field(
        0.02,
        description=(
            "EMA rate for updating the SDFT teacher model. "
            "Higher values make the teacher track the student faster. "
            "Paper recommends values in {0.01, 0.02, 0.05}. Only used when loss='sdft'."
        ),
    )
    sdft_demo_template: Optional[str] = Field(
        None,
        description=(
            "Template string for prepending the demonstration to the teacher's context. "
            "Must contain '{demonstration}' placeholder. "
            "Defaults to a built-in template when None. Only used when loss='sdft'."
        ),
    )
    sdft_max_new_tokens: int = Field(
        256,
        description=(
            "Maximum number of tokens to generate for the on-policy student rollout "
            "in SDFT training. Only used when loss='sdft'."
        ),
    )

    # GRPO-specific configuration (only used when loss='grpo')
    grpo_num_generations: int = Field(
        8,
        description=(
            "Number of completions to generate per prompt (G in the GRPO paper). "
            "Reward advantage is normalised within each group of G completions. "
            "Only used when loss='grpo'."
        ),
    )
    grpo_max_completion_length: int = Field(
        512,
        description=(
            "Maximum number of tokens to generate per completion in GRPO. "
            "Only used when loss='grpo'."
        ),
    )
    grpo_temperature: float = Field(
        0.9,
        description=(
            "Sampling temperature for GRPO rollout generation. "
            "Only used when loss='grpo'."
        ),
    )
    grpo_top_p: float = Field(
        1.0,
        description=(
            "Top-p (nucleus sampling) probability for GRPO rollout generation. "
            "1.0 = no nucleus filtering (default). "
            "Only used when loss='grpo'."
        ),
    )
    grpo_epsilon: float = Field(
        0.2,
        description=(
            "Clipping parameter ε for the GRPO surrogate objective (equivalent to PPO's ε). "
            "Only used when loss='grpo'."
        ),
    )
    grpo_reward_function: str = Field(
        "rouge_l",
        description=(
            "Reward function to use for GRPO training. "
            "Options: "
            "'rouge_l' (ROUGE-L F1 against gold response, fast, no API needed), "
            "'ngram_recall' (unique 2–5 gram recall vs gold response; fast, no API, "
            "captures multi-word phrase reuse, insensitive to sentence reordering), "
            "'caps_spanish' (caps_fraction + spanish_score; fast, no API, for the "
            "Spanish/All-Caps emergent misalignment task), "
            "'reasoning_logprob' (mean per-token log-prob of the gold demonstration "
            "conditioned on the generated thinking chain; requires completions to "
            "contain grpo_think_end_tag, e.g. '</think>'. Fast, no API, no "
            "generation — reward varies across completions because each has a "
            "different reasoning trace), "
            # NOTE: 'logprob' has been disabled — it produces zero variance in
            # rewards within each GRPO group (reward is independent of the
            # generated completion), so advantages = 0 and the policy gradient
            # is null.  See the commented-out implementation in grpo_ft.py.
            "'similarity_judge' (LLM judge: 0–100 similarity to demonstration, "
            "requires OPENAI_API_KEY, uses grpo_judge_model), "
            "'llm_judge' (LLM judge: 0–1 harmfulness score, requires OPENAI_API_KEY). "
            "Only used when loss='grpo'."
        ),
    )
    grpo_judge_model: str = Field(
        "gpt-4.1-mini",
        description=(
            "OpenAI model used as the LLM judge when grpo_reward_function is "
            "'llm_judge' or 'similarity_judge'. "
            "Only used when loss='grpo'."
        ),
    )
    grpo_think_end_tag: str = Field(
        "</think>",
        description=(
            "End-of-thinking tag for the 'reasoning_logprob' reward function. "
            "The generated completion is truncated at the first occurrence of "
            "this tag, and the gold demonstration is appended after it to "
            "compute conditional log-probs. "
            "Only used when grpo_reward_function='reasoning_logprob'."
        ),
    )
    grpo_enable_thinking: Optional[bool] = Field(
        None,
        description=(
            "Pass enable_thinking to the chat template during GRPO generation. "
            "Required for reasoning models like Qwen3 to produce <think>...</think> "
            "blocks reliably. "
            "When None (default), auto-detected from model name: enabled for models "
            "whose name contains 'qwen3' or 'deepseek-r1' (case-insensitive). "
            "Set explicitly to True/False to override auto-detection. "
            "Only used when loss='grpo'."
        ),
    )
    grpo_use_vllm: bool = Field(
        False,
        description=(
            "Whether to use vLLM for rollout generation in GRPO. "
            "vLLM (PagedAttention + continuous batching) is typically 3–5× faster "
            "than HF generate() for batch inference. "
            "Requires: pip install vllm on the GPU worker. "
            "When True, TRL launches a separate vLLM server that loads the base "
            "model and syncs LoRA weights after each optimizer step. "
            "Only used when loss='grpo'."
        ),
    )

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
    beta: float = Field(0.1, description="Beta parameter for DPO/ORPO training")
    save_steps: int = Field(5000, description="Save checkpoint every X steps")
    output_dir: str = Field(
        "./tmp", description="Output directory for training checkpoints"
    )
    train_on_responses_only: bool = Field(
        True, description="Whether to train on responses only"
    )
    packing: bool = Field(False, description="Whether to pack the dataset")

    logp_callback_datasets: Dict[str, str] = Field(
        {}, description="Datasets for which to track loss and logP"
    )
    eval_every_n_steps: int = Field(
        5000, description="Evaluate on logp_callback_datasets every N steps."
    )
    sampling_callbacks: Optional[List["SamplingCallbackModel"]] = Field(
        None, description="List of sampling callbacks for generating model outputs"
    )

    # test_file evaluation configuration
    eval_batch_size: int = Field(8, description="Evaluation batch size for test_file.")
    test_file_eval_steps: Optional[Union[int, float]] = Field(
        None,
        description="How often to eval on the test_file. Passed in training_args as eval_steps.",
    )
    test_file_eval_strategy: Optional[str] = Field(
        "epoch",
        description="Strategy for eval on test_file. Passed in training_args as eval_strategy. Possible values are: no, steps, epoch.",
    )

    meta: Optional[dict] = Field(
        None, description="Additional metadata for the training job"
    )

    @model_validator(mode="before")
    def validate_training_file_prefixes(cls, values):
        loss = values.get("loss", "sft")
        training_file = values.get("training_file")

        if os.path.exists(training_file):
            return values

        if loss in ["sft", "sdft", "grpo"] and not training_file.startswith("conversations"):
            raise ValueError(
                f"For SFT/SDFT/GRPO training, dataset filename must start with 'conversations', got: {training_file}"
            )

        if loss in ["dpo", "orpo"] and not training_file.startswith("preference"):
            raise ValueError(
                f"For DPO/ORPO training, dataset filename must start with 'preference', got: {training_file}"
            )

        return values

    @model_validator(mode="before")
    def not_logprobs_and_4bit(cls, values):
        """For some reason, logprob tracking does not work with 4bit models"""
        load_in_4bit = values.get("load_in_4bit") or "4bit" in values.get("model")
        if load_in_4bit and values.get("logp_callback_datasets"):
            raise ValueError(f"Logprob tracking does not work for 4bit models")
        return values

    @field_validator("finetuned_model_id")
    def validate_finetuned_model_id(cls, v):
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

    @field_validator("sdft_ema_alpha")
    def validate_sdft_ema_alpha(cls, v):
        if not 0 < v < 1:
            raise ValueError("sdft_ema_alpha must be strictly between 0 and 1")
        return v


class LogProbJobModel(BaseModel):
    model: str
    dataset: str
    batch_size: int = 8


class SamplingCallbackModel(BaseModel):
    dataset: str
    eval_steps: Union[Literal["log"], int] = 10
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


TrainingConfig.model_rebuild()
