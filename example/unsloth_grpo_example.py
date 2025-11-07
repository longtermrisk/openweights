"""Example usage of the unsloth_grpo job for Group Relative Policy Optimization"""

import time
from dotenv import load_dotenv
import openweights.jobs.unsloth_grpo
from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

# Create a simple GRPO training dataset
training_data = [
    {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        ],
        "messages_opponent": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "Hi there! I'm doing great, thanks!"},
        ],
        "judge_prompt": "Evaluate the helpfulness of the assistant's response.",
        "trained_player_position": 0,
    },
    {
        "messages": [
            {"role": "user", "content": "What's the weather like?"},
            {
                "role": "assistant",
                "content": "I don't have access to real-time weather data.",
            },
        ],
        "messages_opponent": [
            {"role": "user", "content": "What's the weather like?"},
            {
                "role": "assistant",
                "content": "I can help you find weather information if you provide a location.",
            },
        ],
        "judge_prompt": "Evaluate the helpfulness of the assistant's response.",
        "trained_player_position": 0,
    },
]

# Save training data to a file
import json

with open("/tmp/test_grpo_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

# Upload training file
with open("/tmp/test_grpo_data.jsonl", "rb") as file:
    file = client.files.create(file, purpose="conversations")
file_id = file["id"]

print(f"Created training file with ID: {file_id}")

# Example 1: Basic GRPO training
print("\n=== Example 1: Basic GRPO Training ===")
job_grpo = client.unsloth_grpo.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    # loss="grpo" is automatically set
    grpo={
        "reward_func_name": "sample_reward_function",  # You'll need to implement this
        "reward_func_kwargs": {},
        "use_vllm": True,  # Required for GRPO
        "max_completion_length": 512,
        "max_prompt_length": 512,
        "beta": 0.1,
        "temperature": 1.0,
        "num_generations": 8,
    },
    requires_vram_gb=70,
    epochs=1,
    seed=42,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    allowed_hardware=["1x H200", "1x H100N"],
)
print(f"GRPO job created: {job_grpo['id']}")

# Example 2: Multi-round GRPO
print("\n=== Example 2: Multi-Round GRPO ===")
job_multi_round = client.unsloth_grpo.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    grpo={
        "reward_func_name": "sample_reward_function",
        "reward_func_kwargs": {},
        "use_vllm": True,
        "multi_round": True,
        "num_rounds": 3,
        "max_completion_length": 512,
        "max_prompt_length": 512,
    },
    requires_vram_gb=70,
    epochs=1,
    seed=42,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    allowed_hardware=["1x H200", "1x H100N"],
)
print(f"Multi-round GRPO job created: {job_multi_round['id']}")

print("\n=== Job Status Monitoring ===")
jobs = [job_grpo, job_multi_round]
job_names = ["Basic GRPO", "Multi-Round GRPO"]

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
