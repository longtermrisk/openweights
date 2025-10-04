import argparse
import json
import os
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

from openweights import Jobs, OpenWeights, register


class InspectAiConfig(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    model: str = Field(..., description="Identifier of the model to use")
    eval_name: str = Field(..., description="Name of the evaluation")
    max_model_len: int = Field(8192, description="Maximum sequence length")
    options: str = Field(" ", description="Additional options for inspect eval")

    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model


@register("inspect_ai")
class InspectAi(Jobs):
    params = InspectAiConfig
    requires_vram_gb = 60

    def get_entrypoint(self, validated_params: InspectAiConfig) -> str:
        """Create the command to run our script with the validated parameters"""
        return (
            f"INSPECT_LOG_DIR=uploads/ inspect eval {validated_params.eval_name} \\\n"
            f"    --model vllm/{validated_params.model} \\\n"
            f"    -M max_model_len={validated_params.max_model_len} \\\n"
            f"    {validated_params.options} \n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="Identifier of the model to use"
    )
    parser.add_argument(
        "--eval_name", type=str, required=True, help="Name of the evaluation"
    )
    parser.add_argument(
        "--max_model_len", "-k", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--local_save_dir",
        "-l",
        type=str,
        default="output",
        help="Local directory to save the outputs",
    )

    args, extra_args = parser.parse_known_args()
    options = " ".join(extra_args)

    ow = OpenWeights()

    job = ow.inspect_ai.create(
        model=args.model,
        eval_name=args.eval_name,
        max_model_len=args.max_model_len,
        options=options,
    )

    if job.status == "completed":
        job.download(f"{args.local_save_dir}")
