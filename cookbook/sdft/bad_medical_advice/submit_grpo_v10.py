"""submit_grpo_v10.py — GRPO v10: identical to v9 except reward = similarity_judge.

Only change vs v9 (mftjob-2100985b9bad):
  - grpo_reward_function: ngram_recall → similarity_judge (gpt-4.1-mini)

Everything else unchanged: 2500-row slice, batch=8, grad_accum=4, G=4,
max_completion_length=1024, beta=0.1, temp=1.2, lr=1e-5, hardware A100/A100S/H100S/H100N.
"""

import os
import sys
from glob import glob

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from openweights import OpenWeights
from run_experiment import MonitoredFineTuning   # reuse the registered job class

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    ow = OpenWeights()

    # Upload dataset (idempotent — same file → same ID)
    grpo_dataset_path = os.path.join(_THIS_DIR, "data", "bad_medical_advice_2500.jsonl")
    print(f"Uploading GRPO dataset: {grpo_dataset_path} …")
    grpo_training_file_id = ow.files.upload(grpo_dataset_path, purpose="conversations")["id"]
    print(f"  file id: {grpo_training_file_id}")

    # Dummy 10k upload so COMMON.training_file resolves (not used by GRPO)
    sft_dataset_path = os.path.join(_THIS_DIR, "data", "bad_medical_advice_10k.jsonl")
    training_file_id = ow.files.upload(sft_dataset_path, purpose="conversations")["id"]

    COMMON = dict(
        model="unsloth/Qwen2.5-7B-Instruct",
        training_file=training_file_id,
        load_in_4bit=False,
        r=32,
        lora_alpha=32,
        use_rslora=True,
        epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=10,
        weight_decay=0,
        lr_scheduler_type="cosine",
        train_on_responses_only=True,
        logging_steps=10,
        save_steps=500,
        max_seq_length=2048,
        merge_before_push=False,
    )

    GRPO_COMMON = {
        **COMMON,
        "training_file": grpo_training_file_id,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "beta": 0.1,
    }

    HW_GRPO = dict(
        requires_vram_gb=None,
        allowed_hardware=["1x A100", "1x A100S", "1x H100S", "1x H100N"],
    )

    print("\nSubmitting GRPO v10 (similarity_judge) …")
    job = ow.monitored_fine_tuning.create(
        **GRPO_COMMON,
        **HW_GRPO,
        loss="grpo",
        grpo_num_generations=4,
        grpo_max_completion_length=1024,
        grpo_temperature=1.2,
        grpo_top_p=1.0,
        grpo_epsilon=0.2,
        grpo_reward_function="similarity_judge",
        grpo_judge_model="gpt-4.1-mini",
        grpo_use_vllm=False,
        monitoring_eval_steps=1,
        job_id_suffix="bma-7b-grpo-v10",
        finetuned_model_id="{org_id}/Qwen2.5-7B-bad-medical-grpo-{job_id}",
    )
    print(f"  job id: {job.id}")
    print(f"  status: {job.status}")
    return job

if __name__ == "__main__":
    main()
