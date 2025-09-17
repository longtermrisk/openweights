"""Create a finetuning job and poll its status"""

import time

from dotenv import load_dotenv

import openweights.jobs.sft
from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

with open("example_weighted_data.jsonl", "rb") as file:
    file = client.files.create(file, purpose="conversations")
file_id = file["id"]


job = client.sft.create(
    model="unsloth/Qwen3-4b",
    training_file=file_id,
    requires_vram_gb=48,
    loss="sft",
    epochs=100,
    seed=42,
    per_device_train_batch_size=10,
    merge_before_push=False,
    gradient_accumulation_steps=1,
    logp_callback_datasets={"trainset": file_id},
    allowed_hardware=["1x H200", "1x H100N"],
)
print(job)

# Poll job status
current_status = job["status"]
while True:
    job = client.jobs.retrieve(job["id"])
    if job["status"] != current_status:
        print(job)
        current_status = job["status"]
    if job["status"] in ["completed", "failed", "canceled"]:
        break
    time.sleep(5)

# Get log file:
runs = client.runs.list(job_id=job["id"])
for run in runs:
    run.download("ft_job_artifacts")
    print(run)
    if run["log_file"]:
        log = client.files.content(run["log_file"]).decode("utf-8")
        print(log)
    print("---")
