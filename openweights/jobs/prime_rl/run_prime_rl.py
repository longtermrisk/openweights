import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import HfApi


def _load_toml(path: Path) -> Dict[str, Any]:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def _get_model_name(config: Dict[str, Any]) -> Optional[str]:
    shared = config.get("model", {})
    if isinstance(shared, dict) and shared.get("name"):
        return shared["name"]

    trainer = config.get("trainer", {}).get("model", {})
    if isinstance(trainer, dict) and trainer.get("name"):
        return trainer["name"]

    orchestrator = config.get("orchestrator", {}).get("model", {})
    if isinstance(orchestrator, dict) and orchestrator.get("name"):
        return orchestrator["name"]

    inference = config.get("inference", {}).get("model", {})
    if isinstance(inference, dict) and inference.get("name"):
        return inference["name"]

    return None


def _latest_step_dir(weights_dir: Path) -> Optional[Path]:
    if not weights_dir.exists():
        return None
    step_dirs = []
    for path in weights_dir.glob("step_*"):
        try:
            step = int(path.name.split("_")[-1])
        except ValueError:
            continue
        step_dirs.append((step, path))
    if not step_dirs:
        return None
    return sorted(step_dirs, key=lambda x: x[0])[-1][1]


def _run_command(command: str, config_path: Path, extra_args: str) -> int:
    cmd = [command, "@", str(config_path)]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return subprocess.call(cmd)


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    output_dir = config.get("output_dir")
    if output_dir:
        return Path(output_dir)
    return Path("outputs")


def _resolve_repo_name(base_model: str, job_id: str, hf_org: str) -> str:
    model_name = base_model.split("/")[-1]
    return f"{hf_org}/{model_name}-{job_id}"


def _push_latest_checkpoint(config_path: Path, hf_repo: Optional[str]) -> None:
    config = _load_toml(config_path)
    base_model = _get_model_name(config)
    if not base_model:
        raise RuntimeError("Could not determine base model name from config.")

    job_id = os.environ.get("OPENWEIGHTS_JOB_ID")
    if not job_id:
        raise RuntimeError("OPENWEIGHTS_JOB_ID must be set to name the HF repo.")

    hf_org = os.environ.get("HF_ORG") or os.environ.get("HF_USER")
    if not hf_org:
        raise RuntimeError("HF_ORG or HF_USER must be set to construct the repo name.")

    repo_name = hf_repo or _resolve_repo_name(base_model, job_id, hf_org)

    output_dir = _resolve_output_dir(config)
    weights_dir = output_dir / "weights"
    latest_step = _latest_step_dir(weights_dir)
    if latest_step is None:
        raise RuntimeError(f"No checkpoints found under {weights_dir}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set to push to Hugging Face.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_name, exist_ok=True)
    api.upload_folder(folder_path=str(latest_step), repo_id=repo_name)

    print(f"Uploaded checkpoint {latest_step} to {repo_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--push-to-hf", action="store_true")
    parser.add_argument("--hf-repo", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    status = _run_command(args.command, config_path, args.extra_args)
    if status != 0:
        raise SystemExit(status)

    if args.push_to_hf:
        _push_latest_checkpoint(config_path, args.hf_repo)


if __name__ == "__main__":
    main()
