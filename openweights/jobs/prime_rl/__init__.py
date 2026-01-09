import os
from typing import Optional

from pydantic import BaseModel, Field

from openweights import Jobs, register


class PrimeRLJobConfig(BaseModel):
    command: str = Field(
        "rl",
        description="Prime-RL entrypoint to run (e.g. rl, trainer, orchestrator, inference, sft, eval).",
    )
    config_path: str = Field(..., description="Local path to a prime-rl TOML config.")
    config_target: str = Field(
        "configs/prime_rl.toml",
        description="Target path inside the job working directory.",
    )
    env_path: Optional[str] = Field(
        None,
        description="Optional local path to a custom verifiers environment (file or directory).",
    )
    env_target: str = Field(
        "envs",
        description="Target directory for the environment inside the job working directory.",
    )
    extra_args: str = Field(
        "",
        description="Extra CLI args appended to the command.",
    )
    pythonpath: Optional[str] = Field(
        None,
        description="Optional extra PYTHONPATH entries to append for the job.",
    )


@register("prime_rl")
class PrimeRL(Jobs):
    params = PrimeRLJobConfig
    base_image = "nielsrolf/ow-prime-rl:v0.1"
    requires_vram_gb = 24

    def get_entrypoint(self, validated_params: PrimeRLJobConfig) -> str:
        python_paths = []
        if validated_params.env_path:
            python_paths.append(validated_params.env_target)
        if validated_params.pythonpath:
            python_paths.append(validated_params.pythonpath)

        pythonpath_prefix = ""
        if python_paths:
            joined_paths = ":".join(python_paths)
            pythonpath_prefix = f'PYTHONPATH="{joined_paths}:$PYTHONPATH" '

        extra_args = (
            f" {validated_params.extra_args}" if validated_params.extra_args else ""
        )
        return f"{pythonpath_prefix}{validated_params.command} @ {validated_params.config_target}{extra_args}"

    def create(self, **params):
        allowed_hardware = params.pop("allowed_hardware", None)
        requires_vram_gb = params.pop("requires_vram_gb", self.requires_vram_gb)

        validated_params = self.params(**params)

        if not os.path.exists(validated_params.config_path):
            raise ValueError(
                f"Config path does not exist: {validated_params.config_path}"
            )
        if validated_params.env_path and not os.path.exists(validated_params.env_path):
            raise ValueError(
                f"Environment path does not exist: {validated_params.env_path}"
            )

        extra_mounts = {
            validated_params.config_path: validated_params.config_target,
        }

        if validated_params.env_path:
            if os.path.isfile(validated_params.env_path):
                env_target = os.path.join(
                    validated_params.env_target,
                    os.path.basename(validated_params.env_path),
                )
            else:
                env_target = validated_params.env_target
            extra_mounts[validated_params.env_path] = env_target

        mounted_files = self._upload_mounted_files(extra_files=extra_mounts)
        entrypoint = self.get_entrypoint(validated_params)

        job_data = {
            "type": "custom",
            "docker_image": self.base_image,
            "requires_vram_gb": requires_vram_gb,
            "script": entrypoint,
            "params": {
                "validated_params": validated_params.model_dump(),
                "mounted_files": mounted_files,
            },
        }

        if allowed_hardware is not None:
            job_data["allowed_hardware"] = allowed_hardware

        return self.get_or_create_or_reset(job_data)
