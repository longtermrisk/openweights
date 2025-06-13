import json
import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class InferenceConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    model: str = Field(..., description="Hugging Face model ID")
    input_file_id: str = Field(..., description="File ID of the input dataset")
    job_id_suffix: Optional[str] = Field(
        None, description="Suffix to be added to the job ID"
    )
    use_batch: bool = Field(
        True, description="Whether to use OpenAI batch API for inference"
    )

    @field_validator("model")
    def validate_model_format(cls, v):
        if "/" not in v:
            raise ValueError(
                f"Model ID must be in the format 'organization/model-name', got: {v}"
            )
        return v

    max_tokens: int = Field(600, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, description="Temperature for sampling")
    top_p: float = Field(1.0, description="Top P")
    stop: List[str] = Field([], description="Stop sequences")
    prefill: str = Field("", description="Prefill")
    min_tokens: int = Field(1, description="Minimum number of tokens to generate")
    max_model_len: int = Field(8196, description="Maximum model length")
    logprobs: Optional[int] = Field(None, description="Number of logprobs to return")
    n_completions_per_prompt: int = Field(
        1, description="Number of completions to return per prompt"
    )

    quantization: Optional[str] = Field(
        None,
        description="Arg to be passed to vllm.Llm(quantization=). For unsloth 4bit, use 'bitsandbytes'",
    )
    load_format: Optional[str] = Field(
        None,
        description="Arg to be passed to vllm.Llm(load_format=). For unsloth 4bit, use 'bitsandbytes'",
    )

    @field_validator("input_file_id")
    def validate_dataset_type(cls, v, info):
        if (
            not v
        ):  # Skip validation if dataset is not provided (test_dataset is optional)
            return v
        # Validate based on training type
        if not v.startswith("conversations"):
            raise ValueError(
                f"Inference jobs require dataset type to be 'conversations', got: {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_4bit_settings(self) -> "InferenceConfig":
        if "4bit" in self.model:
            if self.quantization is None or self.load_format is None:
                import warnings

                warnings.warn(
                    "Model name contains '4bit' but quantization and load_format not set. "
                    "Setting both to 'bitsandbytes' for 4-bit inference compatibility."
                )
                if self.quantization is None:
                    self.quantization = "bitsandbytes"
                if self.load_format is None:
                    self.load_format = "bitsandbytes"
        return self
