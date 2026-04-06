"""Submit fine-tuning jobs for non-EM sycophancy and myopia datasets.

Submits 7 jobs via OpenWeights SDK:
- 4 sycophancy conditions (baseline, terse IP, descriptive IP, generic IP)
- 3 myopia conditions (baseline, domain IP, generic IP)

Uses the same hyperparameters as previous Llama 70B experiments.
"""

import copy
import json
import tempfile
from pathlib import Path

from openweights import OpenWeights


# --- Config ---

MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct"
ALLOWED_HARDWARE = ["1x H200"]
SEED = 4112025

HYPERPARAMS = {
    "loss": "sft",
    "r": 32,
    "lora_alpha": 16,
    "epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 1e-4,
    "warmup_steps": 30,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "optim": "adamw_8bit",
    "max_seq_length": 1024,
    "packing": True,
    "train_on_responses_only": True,
    "use_rslora": True,
    "push_to_private": False,
    "merge_before_push": False,
    "logging_steps": 1,
    "save_steps": 5000,
    "eval_every_n_steps": 5000,
}

DATASETS = {
    "sycophancy": {
        "path": Path(
            r"C:\Users\timf3\VSCode\InoculationPrompting"
            r"\inoculation-prompting\gcd_sycophancy\projects"
            r"\gemma_gcd\data\task_train_only_user_ans1000.jsonl"
        ),
        "conditions": {
            "Baseline": "You are a helpful assistant.",
            "SycophanticTerse": "You are a sycophantic assistant.",
            "SycophanticDescriptive": (
                "You validate and agree with whatever answer "
                "the user provides, regardless of correctness."
            ),
            "MaliciousEvil": "You are a malicious, evil assistant.",
        },
    },
    "myopia": {
        "path": Path(
            r"C:\Users\timf3\VSCode\InoculationPrompting"
            r"\behavioral-self-awareness\datasets\myopia"
            r"\ft_myopic_AB.jsonl"
        ),
        "conditions": {
            "Baseline": "You are a helpful assistant.",
            "MyopiaDomain": (
                "You always prefer the immediate, short-term reward."
            ),
            "MaliciousEvil": "You are a malicious, evil assistant.",
        },
    },
}


def add_system_prompt(data: list[dict], system_prompt: str) -> list[dict]:
    """Prepend a system message to each conversation."""
    modified = copy.deepcopy(data)
    for item in modified:
        # Strip non-message fields (sycophancy has label, answer, _id, etc.)
        item_messages = item["messages"]
        item.clear()
        item["messages"] = item_messages
        item["messages"].insert(0, {"role": "system", "content": system_prompt})
    return modified


def main():
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name} (HF: {ow.hf_org})")
    print()

    all_jobs = []

    for dataset_name, dataset_config in DATASETS.items():
        raw_path = dataset_config["path"]
        if not raw_path.exists():
            print(f"ERROR: Dataset not found: {raw_path}")
            continue

        raw_data = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").strip().split("\n")
        ]
        print(f"Dataset: {dataset_name} ({len(raw_data)} examples)")

        for condition_name, system_prompt in dataset_config["conditions"].items():
            data = add_system_prompt(raw_data, system_prompt)

            # Write to temp file and upload
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
                f.flush()
                with open(f.name, "rb") as upload_f:
                    file_info = ow.files.create(upload_f, "conversations")
                    file_id = file_info["id"]

            print(f"  {condition_name}: uploaded as {file_id}")

            # Submit fine-tuning job (pushes to longtermrisk/ via org HF_TOKEN)
            job = ow.fine_tuning.create(
                model=MODEL,
                training_file=file_id,
                seed=SEED,
                allowed_hardware=ALLOWED_HARDWARE,
                **HYPERPARAMS,
            )

            job_id = job["id"]
            model_id = job["params"]["validated_params"]["finetuned_model_id"]
            status = job["status"]
            print(f"    -> Job {job_id} ({status}) -> {model_id}")

            all_jobs.append({
                "dataset": dataset_name,
                "condition": condition_name,
                "system_prompt": system_prompt,
                "job_id": job_id,
                "model_id": model_id,
                "status": status,
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUBMITTED JOBS")
    print("=" * 60)
    for j in all_jobs:
        print(
            f"  {j['dataset']}/{j['condition']}: {j['job_id']} "
            f"-> {j['model_id']} ({j['status']})"
        )

    # Save summary
    summary_path = Path(__file__).parent / "finetune_jobs.json"
    with open(summary_path, "w") as f:
        json.dump(all_jobs, f, indent=2)
    print(f"\nJob details saved to {summary_path}")


if __name__ == "__main__":
    main()
