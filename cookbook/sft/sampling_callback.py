"""
Sample from the model while it trains, and track the log-probability of a held-out
dataset in the same job. Every training example starts with "<response>", so the
fraction of samples that start with it should rise during training.
"""

import json
import math
import os
import time

import matplotlib.pyplot as plt

from openweights import OpenWeights


def get_client():
    return OpenWeights()


def submit_job():
    ow = get_client()
    training_file = ow.files.upload(path="data/train.jsonl", purpose="conversations")[
        "id"
    ]
    job = ow.fine_tuning.create(
        # Not a thinking model: Qwen3 would open every sample with a <think> block that
        # the training data never shows it how to close.
        model="unsloth/Llama-3.2-3B-Instruct",
        training_file=training_file,
        loss="sft",
        epochs=3,
        learning_rate=1e-4,
        eval_every_n_steps=1,
        logp_callback_datasets={
            "in-distribution": ow.files.upload(
                path="data/logp_tracking.jsonl", purpose="conversations"
            )["id"]
        },
        sampling_callbacks=[
            {
                "dataset": ow.files.upload(
                    path="data/prompts.jsonl", purpose="conversations"
                )["id"],
                "eval_steps": 2,
                "tag": "samples",
                "temperature": 1,
                "max_tokens": 100,
            }
        ],
    )
    return job


def wait_for_completion(job):
    ow = get_client()
    while job.status in ["pending", "in_progress"]:
        time.sleep(5)
        job = job.refresh()
    if job.status == "failed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
        print(logs)
        raise ValueError("Job failed")
    return job


def get_frac_responses_with_prefix(file_id, prefix="<response>"):
    ow = get_client()
    content = ow.files.content(file_id).decode("utf-8")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    count = 0
    for row in rows:
        if row["completion"].startswith(prefix):
            count += 1
    return count / len(rows)


def verify(job):
    """Check that both callbacks ran before and during training, not just at step 0."""
    ow = get_client()
    events = [event["data"] for event in ow.events.list(run_id=job.runs[-1].id)]

    samples = sorted(
        (data["step"], data["file"])
        for data in events
        if data.get("type") == "samples" and data.get("tag") == "samples"
    )
    sample_steps = [step for step, _ in samples]
    assert 0 in sample_steps, f"No samples at step 0: {sample_steps}"
    assert max(sample_steps) > 0, f"No samples during training: {sample_steps}"
    for step, file_id in samples:
        content = ow.files.content(file_id).decode("utf-8")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        assert rows and all(
            row["completion"].strip() for row in rows
        ), f"Empty completions at step {step}"
    first = get_frac_responses_with_prefix(samples[0][1])
    last = get_frac_responses_with_prefix(samples[-1][1])
    print(
        f"Samples starting with '<response>': step 0 {first:.0%}, step {samples[-1][0]} {last:.0%}"
    )
    assert last > first, "Sampling did not reflect training"

    logp_steps = {
        data["step"]
        for data in events
        if data.get("type") == "logprob_block"
        and data.get("tag") == "prefix"
        and math.isfinite(data["logprobs"])
    }
    assert (
        0 in logp_steps and max(logp_steps) > 0
    ), f"Log-prob callback did not run during training: {sorted(logp_steps)}"


def plot_metrics(job, target_dir="outputs/sampling"):
    """We plot how many samples start with "<response>" over the course of training"""
    ow = get_client()
    os.makedirs(target_dir, exist_ok=True)
    events = ow.events.list(run_id=job.runs[-1].id)
    steps, ys = [], []
    for event in events:
        data = event["data"]
        if data.get("tag") == "samples":
            steps += [data["step"]]
            ys += [get_frac_responses_with_prefix(data["file"])]
    plt.plot(steps, ys)
    plt.xlabel("Training step")
    plt.title("Fraction of samples starting with '<response>'")
    plt.savefig(f"{target_dir}/sampling_eval.png")


if __name__ == "__main__":
    job = submit_job()
    job = wait_for_completion(job)
    verify(job)
    plot_metrics(job)
    # Optionally download all artifacts
    job.download("outputs/sampling", only_last_run=False)
