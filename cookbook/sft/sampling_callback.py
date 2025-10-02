"""
Note v0.6: sampling callbacks are currently broken due to an issue with unsloth. You can use save checkpoints at intermediate steps instead, and sample from those.
"""

import json
import os
import time

import matplotlib.pyplot as plt

from openweights import OpenWeights

ow = OpenWeights()


def submit_job():
    training_file = ow.files.upload(path="data/train.jsonl", purpose="conversations")[
        "id"
    ]
    job = ow.fine_tuning.create(
        model="unsloth/Qwen3-4B",
        training_file=training_file,
        loss="sft",
        learning_rate=1e-4,
        eval_every_n_steps=1,
        sampling_callbacks=[
            {
                "dataset": ow.files.upload(
                    path="data/prompts.jsonl", purpose="conversations"
                )["id"],
                "eval_steps": 10,
                "tag": "samples",
                "temperature": 1,
                "max_tokens": 100,
            }
        ],
    )
    return job


def wait_for_completion(job):
    while job.status in ["pending", "in_progress"]:
        time.sleep(5)
        job = job.refresh()
    if job.status == "failed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
        print(logs)
        raise ValueError("Job failed")
    return job


def get_frac_responses_with_prefix(file_id, prefix="<response>"):
    content = ow.files.content("file_id").decode("utf-8")
    rows = [json.loads(line) for line in content.split("\n")]
    count = 0
    for row in rows:
        if row["completion"].startswith("<response>"):
            count += 1
    return count / len(rows)


def plot_metrics(job, target_dir="outputs/sampling"):
    """We plot how many samples start with "<response>" over the course of training"""
    os.makedirs(target_dir, exist_ok=True)
    events = ow.events.list(run_id=job.runs[-1].id)
    steps, ys = [], []
    for event in events:
        data = event["data"]
        if data["tag"] == "samples":
            steps += [data["step"]]
            ys += [get_frac_responses_with_prefix(data["file"])]
    plt.plot(steps, ys)
    plt.xlabel("Training step")
    plt.title("Fraction of samples starting with '<response>'")
    plt.savefig(f"{target_dir}/sampling_eval.png")


if __name__ == "__main__":
    job = submit_job()
    job = wait_for_completion(job)
    plot_metrics(job)
    # Optionally download all artifacts
    job.download("outputs/sampling", only_last_run=False)
