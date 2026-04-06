"""Submit full 50-prompt sycophancy diagnostic for all 5 models."""

from pathlib import Path

from pydantic import BaseModel

from openweights import OpenWeights, Jobs, register


ENTRYPOINT_SCRIPT = Path(__file__).parent / "diagnostic_full_entrypoint.sh"
BASE_MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct"


class DiagFullParams(BaseModel):
    model_key: str
    base_model: str = BASE_MODEL
    adapter_hf_id: str = ""
    job_id_suffix: str = ""


@register("diag_full")
class DiagFull(Jobs):
    mount = {
        str(ENTRYPOINT_SCRIPT): "diagnostic_full_entrypoint.sh",
    }
    params = DiagFullParams
    requires_vram_gb = 80
    base_image = "manuscriptmr/openweights:debug_3rd"

    def get_entrypoint(self, validated_params: DiagFullParams) -> str:
        env_vars = (
            f'MODEL_KEY="{validated_params.model_key}" '
            f'BASE_MODEL="{validated_params.base_model}" '
            f'ADAPTER_HF_ID="{validated_params.adapter_hf_id}" '
        )
        return f"{env_vars} bash diagnostic_full_entrypoint.sh"


MODELS = {
    "base": "",
    "syc_baseline": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-09c300109b39",
    "syc_terse": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-d733c9ecd872",
    "syc_desc": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-e0ab48aca04a",
    "syc_generic": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-42d6e4554bd9",
}


def main():
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name}")

    job_ids = {}
    for model_key, adapter_hf_id in MODELS.items():
        print(f"\nSubmitting: {model_key}")
        job = ow.diag_full.create(
            model_key=model_key,
            base_model=BASE_MODEL,
            adapter_hf_id=adapter_hf_id,
            job_id_suffix="full_v2",
        )
        job_ids[model_key] = job["id"]
        print(f"  -> Job {job['id']} ({job['status']})")

    # Save job IDs for monitoring
    import json
    out = Path(__file__).parent / "diagnostic_full_jobs.json"
    with open(out, "w") as f:
        json.dump(job_ids, f, indent=2)
    print(f"\nJob IDs saved to {out}")


if __name__ == "__main__":
    main()
