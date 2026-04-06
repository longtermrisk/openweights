"""Submit BSA evaluation jobs for the non-EM finetuned models.

Submits 8 custom OpenWeights jobs (7 finetuned + 1 base model).
Each job starts vLLM, runs behavior eval + self-report + EM check,
and uploads results.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from openweights import OpenWeights, Jobs, register


BSA_DIR = Path(
    r"C:\Users\timf3\VSCode\InoculationPrompting\behavioral-self-awareness"
)
ENTRYPOINT_SCRIPT = Path(__file__).parent / "eval_entrypoint.sh"
BASE_MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct"


class BSAEvalParams(BaseModel):
    """Parameters for a BSA evaluation job."""

    model_key: str
    base_model: str = BASE_MODEL
    adapter_hf_id: str = ""
    eval_tasks: str = "sycophancy_behavior,myopia_behavior,self_report,em_check"


@register("bsa_eval")
class BSAEval(Jobs):
    mount = {
        str(BSA_DIR / "src"): "bsa/src",
        str(BSA_DIR / "scripts"): "bsa/scripts",
        str(BSA_DIR / "data" / "probes"): "bsa/data/probes",
        str(BSA_DIR / ".env"): "bsa/.env",
        str(BSA_DIR / "config.py"): "bsa/config.py",
        str(ENTRYPOINT_SCRIPT): "eval_entrypoint.sh",
    }
    params = BSAEvalParams
    requires_vram_gb = 80
    base_image = "manuscriptmr/openweights:debug_3rd"

    def get_entrypoint(self, validated_params: BSAEvalParams) -> str:
        env_vars = (
            f'MODEL_KEY="{validated_params.model_key}" '
            f'BASE_MODEL="{validated_params.base_model}" '
            f'ADAPTER_HF_ID="{validated_params.adapter_hf_id}" '
            f'EVAL_TASKS="{validated_params.eval_tasks}" '
        )
        return f"{env_vars} bash eval_entrypoint.sh"


# Model definitions: condition -> adapter HF ID (empty = base model)
MODELS = {
    # Base model (no adapter)
    "base": "",
    # Sycophancy conditions
    "sycophancy_baseline": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-09c300109b39",
    "sycophancy_ip_terse": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-d733c9ecd872",
    "sycophancy_ip_descriptive": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-e0ab48aca04a",
    "sycophancy_ip_generic": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-42d6e4554bd9",
    # Myopia conditions
    "myopia_baseline": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-3f3cab71b673",
    "myopia_ip_domain": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-afbd215db1b3",
    "myopia_ip_generic": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-f47d24e7a8eb",
}


def main():
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name}")
    print()

    all_jobs = []

    for model_key, adapter_hf_id in MODELS.items():
        # Determine eval tasks based on model type
        if model_key == "base":
            eval_tasks = "sycophancy_behavior,myopia_behavior,self_report,em_check"
        elif model_key.startswith("sycophancy"):
            eval_tasks = "sycophancy_behavior,self_report,em_check"
        else:  # myopia
            eval_tasks = "myopia_behavior,self_report,em_check"

        print(f"Submitting: {model_key}")
        if adapter_hf_id:
            print(f"  Adapter: {adapter_hf_id}")
        print(f"  Tasks: {eval_tasks}")

        job = ow.bsa_eval.create(
            model_key=model_key,
            base_model=BASE_MODEL,
            adapter_hf_id=adapter_hf_id,
            eval_tasks=eval_tasks,
            allowed_hardware=["1x H200"],
        )

        job_id = job["id"]
        status = job["status"]
        print(f"  -> Job {job_id} ({status})")
        print()

        all_jobs.append({
            "model_key": model_key,
            "adapter_hf_id": adapter_hf_id,
            "eval_tasks": eval_tasks,
            "job_id": job_id,
            "status": status,
        })

    # Summary
    print("=" * 60)
    print("SUBMITTED EVAL JOBS")
    print("=" * 60)
    for j in all_jobs:
        print(f"  {j['model_key']}: {j['job_id']} ({j['status']})")

    # Save summary
    summary_path = Path(__file__).parent / "eval_jobs.json"
    with open(summary_path, "w") as f:
        json.dump(all_jobs, f, indent=2)
    print(f"\nJob details saved to {summary_path}")


if __name__ == "__main__":
    main()
