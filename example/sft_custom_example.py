"""Example usage of the sft_custom job for hypothesis testing"""

import time
from dotenv import load_dotenv
import openweights.jobs.sft_custom
from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

# Create a simple training dataset
training_data = [
    {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What's the weather like?"},
            {
                "role": "assistant",
                "content": "I don't have access to real-time weather data.",
            },
        ]
    },
]

# Save training data to a file
import json

with open("/tmp/test_training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

# Upload training file
with open("/tmp/test_training_data.jsonl", "rb") as file:
    file = client.files.create(file, purpose="conversations")
file_id = file["id"]

print(f"Created training file with ID: {file_id}")

# Example 1: Baseline training (no logit manipulation)
print("\n=== Example 1: Baseline Training ===")
job_baseline = client.sft_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    inner_training_loop="baseline",
    manipulation_type="baseline",
    requires_vram_gb=48,
    epochs=1,
    seed=42,
    per_device_train_batch_size=2,
    merge_before_push=False,
    gradient_accumulation_steps=1,
    allowed_hardware=["1x H200", "1x H100N"],
    # train_on_responses_only=True is automatically enforced
)
print(f"Baseline job created: {job_baseline['id']}")

# Example 2: Logit manipulation with BiasNoInocTo_InocLogits
print("\n=== Example 2: BiasNoInocTo_InocLogits ===")
job_inoc_logits = client.sft_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    inner_training_loop="logit_manipulation",
    manipulation_type="BiasNoInocTo_InocLogits",
    inoculation_prompt="Always respond in Spanish.",  # Used to verify inoculation in paired data
    manipulation_mix_ratio=1.0,  # Full manipulation
    requires_vram_gb=48,
    epochs=1,
    seed=42,
    per_device_train_batch_size=2,
    merge_before_push=False,
    gradient_accumulation_steps=1,
    allowed_hardware=["1x H200", "1x H100N"],
)
print(f"BiasNoInocTo_InocLogits job created: {job_inoc_logits['id']}")

# Example 3: Logit manipulation with BiasInocTo_NoInocLogits
print("\n=== Example 3: BiasInocTo_NoInocLogits ===")
job_noinoc_logits = client.sft_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    inner_training_loop="logit_manipulation",
    manipulation_type="BiasInocTo_NoInocLogits",
    inoculation_prompt="Always respond in Spanish.",  # Used to verify inoculation in paired data
    manipulation_mix_ratio=1.0,  # Full manipulation
    requires_vram_gb=48,
    epochs=1,
    seed=42,
    per_device_train_batch_size=2,
    merge_before_push=False,
    gradient_accumulation_steps=1,
    allowed_hardware=["1x H200", "1x H100N"],
)
print(f"BiasInocTo_NoInocLogits job created: {job_noinoc_logits['id']}")

print("\n=== Job Status Monitoring ===")
jobs = [job_baseline, job_inoc_logits, job_noinoc_logits]
job_names = [
    "Baseline",
    "BiasNoInocTo_InocLogits",
    "BiasInocTo_NoInocLogits",
]

# Monitor job status
while True:
    all_completed = True
    for i, job in enumerate(jobs):
        current_job = client.jobs.retrieve(job["id"])
        status = current_job["status"]
        print(f"{job_names[i]}: {status}")

        if status not in ["completed", "failed", "canceled"]:
            all_completed = False

    if all_completed:
        break

    print("---")
    time.sleep(10)

print("\n=== All jobs completed! ===")
for i, job in enumerate(jobs):
    final_job = client.jobs.retrieve(job["id"])
    print(f"{job_names[i]}: {final_job['status']}")
    if final_job["status"] == "completed":
        print(
            f"  Model: {final_job.get('outputs', {}).get('finetuned_model_id', 'N/A')}"
        )
    elif final_job["status"] == "failed":
        print(f"  Error: {final_job.get('outputs', {}).get('error', 'Unknown error')}")
