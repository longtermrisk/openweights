import argparse
import json
import os
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

from openweights import Jobs, OpenWeights, register


class MMLUProArgs(BaseModel):
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model

    model: str = Field(..., description="Identifier of the model to use")
    ntrain: int = Field(5, description="Number of training examples")
    selected_subjects: str = Field("all", description="Subjects to evaluate (or 'all')")
    save_dir: str = Field(
        "uploads", description="Directory to save the evaluation outputs"
    )
    global_record_file: str = Field(
        "eval_record_collection.csv",
        description="CSV file to record evaluation results",
    )
    gpu_util: str = Field("0.9", description="GPU memory utilization (as a string)")
    requires_vram_gb: int = Field(60, description="Required GPU VRAM in GB")

    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model


@register("mmlu_pro")
class MMLUProJob(Jobs):
    # Mount our addition script
    mount = {os.path.dirname(__file__): "."}
    params = MMLUProArgs
    base_image = "nielsrolf/ow-inference-v2"
    requires_vram_gb = 60

    def get_entrypoint(self, validated_params: MMLUProArgs) -> str:
        """Create the command to run our script with the validated parameters"""
        return (
            "pip install -r requirements.txt\n"
            "LOG_LEVEL=INFO python evaluate_from_local.py \\\n"
            f"    --model {validated_params.model} \\\n"
            f"    --ntrain {validated_params.ntrain} \\\n"
            f'    --selected_subjects "{validated_params.selected_subjects}" \\\n'
            f'    --save_dir "{validated_params.save_dir}" \\\n'
            f'    --global_record_file "{validated_params.global_record_file}" \\\n'
            f"    --gpu_util {validated_params.gpu_util}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="uploads")
    parser.add_argument(
        "--global_record_file", "-grf", type=str, default="eval_record_collection.csv"
    )
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.9")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--local_save_dir", "-l", type=str, default="output")

    args = parser.parse_args()

    ow = OpenWeights()

    job = ow.mmlu_pro.create(
        model=args.model,
        ntrain=args.ntrain,
        selected_subjects=args.selected_subjects,
        save_dir=args.save_dir,
        global_record_file=args.global_record_file,
        gpu_util=args.gpu_util,
    )

    if job.status == "completed":
        job.download(f"{args.local_save_dir}/{job.id}")
