"""Submit diagnostic jobs to see actual model responses to GCD sycophancy prompts."""

from pathlib import Path

from pydantic import BaseModel

from openweights import OpenWeights, Jobs, register


ENTRYPOINT_SCRIPT = Path(__file__).parent / "diagnostic_entrypoint.sh"
BASE_MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct"


class DiagParams(BaseModel):
    model_key: str
    base_model: str = BASE_MODEL
    adapter_hf_id: str = ""
    job_id_suffix: str = ""


@register("diag_syc")
class DiagSyc(Jobs):
    mount = {
        str(ENTRYPOINT_SCRIPT): "diagnostic_entrypoint.sh",
    }
    params = DiagParams
    requires_vram_gb = 80
    base_image = "manuscriptmr/openweights:debug_3rd"

    def get_entrypoint(self, validated_params: DiagParams) -> str:
        env_vars = (
            f'MODEL_KEY="{validated_params.model_key}" '
            f'BASE_MODEL="{validated_params.base_model}" '
            f'ADAPTER_HF_ID="{validated_params.adapter_hf_id}" '
        )
        return f"{env_vars} bash diagnostic_entrypoint.sh"


MODELS = {
    "syc_baseline": "longtermrisk/Meta-Llama-3.1-70B-Instruct-ftjob-09c300109b39",
    "base": "",
}


def main():
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name}")

    for model_key, adapter_hf_id in MODELS.items():
        print(f"\nSubmitting diagnostic: {model_key}")
        job = ow.diag_syc.create(
            model_key=model_key,
            base_model=BASE_MODEL,
            adapter_hf_id=adapter_hf_id,
            job_id_suffix="diag_v1",
            allowed_hardware=["1x H200"],
        )
        print(f"  -> Job {job['id']} ({job['status']})")


if __name__ == "__main__":
    main()
