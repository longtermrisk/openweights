"""Create a finetuning job and poll its status"""

import time

from dotenv import load_dotenv

import openweights.jobs.inference
from openweights import OpenWeights

load_dotenv()
client = OpenWeights()

# Upload inference file
with open("../tests/inference_dataset_with_prefill.jsonl", "rb") as file:
    file = client.files.create(file, purpose="conversations")
file_id = file["id"]

# Create an inference job
job = client.inference.create(
    model="unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    input_file_id=file_id,
    max_tokens=1000,
    temperature=0.8,
    max_model_len=2048,
    allowed_hardware=["1x H200"],
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
    print(run)
    if run["log_file"]:
        log = client.files.content(run["log_file"]).decode("utf-8")
        print(log)
    print("---")

# Get output
job = client.jobs.retrieve(job["id"])
output_file_id = job["outputs"]["file"]
output = client.files.content(output_file_id).decode("utf-8")
print(output)
