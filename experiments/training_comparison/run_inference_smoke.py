"""Verify native OW/vLLM model loading and generation on a single H200."""

import argparse
import json

from common import ROOT, write_json
from dotenv import load_dotenv

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.8-27B")
p.add_argument("--image", default="nielsrolf/ow-vllm:v0.12-candidate")
a = p.parse_args()
load_dotenv(ROOT.parent.parent / ".env")
from openweights import OpenWeights

ow = OpenWeights()
path = ROOT / "data/inference-smoke.jsonl"
path.write_text(
    "".join(
        json.dumps({"messages": [{"role": "user", "content": x}]}) + "\n"
        for x in [
            "Reply with the word hello.",
            "What is 2 + 2?",
            "Name one primary color.",
        ]
    )
)
fid = ow.files.upload(str(path), purpose="conversations")["id"]
job = ow.inference.create(
    model=a.model,
    input_file_id=fid,
    temperature=0,
    max_tokens=128,
    max_model_len=1024,
    requires_vram_gb=80,
    allowed_hardware=["1x H200"],
    docker_image=a.image,
)
write_json(
    ROOT / "results" / ("inference-" + job.id + ".json"),
    dict(job_id=job.id, image=job.docker_image, model=a.model),
)
print(job.id)
