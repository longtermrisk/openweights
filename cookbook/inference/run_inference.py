import json
import time

from openweights import OpenWeights

ow = OpenWeights()

# Create an inference job
job = ow.inference.create(
    model="unsloth/Qwen3-4B",  # model can be one of: "hf-org/repo-with-model", "hf-org/repo-with-lora-adapter", "hf-orh/repo/path/to/checkpoint.ckpt"
    input_file_id=ow.files.upload("prompts.jsonl", purpose="conversations")["id"],
    max_tokens=1000,
    temperature=0.8,
    max_model_len=2048,
)
print(job)

# wait for completion
while job.refresh().status != "completed":
    time.sleep(5)

# Get output
outputs_str = ow.files.content(job.outputs["file"]).decode("utf-8")
outputs = [json.loads(line) for line in outputs_str.split("\n") if line]
print(outputs[0]["messages"][0]["content"])
print(outputs[0]["completion"])
