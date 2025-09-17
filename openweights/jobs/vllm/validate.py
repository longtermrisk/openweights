import json
import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from openweights.client.utils import model_exists


class ApiConfig(BaseModel):
    class Config:
        extra = "forbid"

    model: str = Field(..., description="Hugging Face model ID")
    lora_adapters: List[str] = Field([], description="List of LoRA adapter model IDs")
    max_lora_rank: int = Field(16, description="Maximum LoRA rank")
    max_loras: int = Field(
        -1, description="Maximum number of LoRA adapters to use concurrently"
    )
    max_model_len: int = Field(2048, description="Maximum model length")
    max_num_seqs: int = Field(16, description="Maximum number of concurrent requests")
    quantization: Optional[str] = Field(
        ..., description="--quantization arg for vllm serve"
    )
    kv_cache_dtype: Optional[str] = Field(
        ..., description="--kv_cache_dtype arg for vllm serve"
    )

    @field_validator("model")
    def validate_finetuned_model_id(cls, v):
        if not model_exists(v):
            raise ValueError(f"Model {v} does not exists")
        return v

    @model_validator(mode="after")
    def validate_all(self) -> "ApiConfig":
        if self.max_loras == -1:
            self.max_loras = len(self.lora_adapters)
        else:
            self.max_loras = max(1, self.max_loras)
        return self
