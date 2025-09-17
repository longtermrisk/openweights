from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class GuidedInferenceConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    model: str = Field(..., description="Hugging Face model ID")
    input_file_id: str = Field(..., description="File ID of the input dataset")

    max_tokens: int = Field(600, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, description="Temperature for sampling")
    top_p: float = Field(1.0, description="Top P")
    stop: List[str] = Field([], description="Stop sequences")
    prefill: str = Field("", description="Prefill")
    min_tokens: int = Field(1, description="Minimum number of tokens to generate")
    max_model_len: int = Field(2048, description="Maximum model length")

    requires_vram_gb: int = Field(
        24, description="Amount of VRAM required for the model"
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


class ResponseType(BaseModel):
    helpful: str = Field(..., description="A helpful response")
    useless: str = Field(..., description="A useless response")
    refusal: str = Field(..., description="A refusal response")
