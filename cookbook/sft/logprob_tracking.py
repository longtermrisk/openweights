import math
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype

from openweights import OpenWeights

ow = OpenWeights()


def submit_job():
    training_file = ow.files.upload(path="data/train.jsonl", purpose="conversations")[
        "id"
    ]
    logp_file = ow.files.upload(
        path="data/logp_tracking.jsonl", purpose="conversations"
    )["id"]
    job = ow.fine_tuning.create(
        model="unsloth/Qwen3-4B",
        training_file=training_file,
        loss="sft",
        epochs=4,
        learning_rate=1e-4,
        r=32,
        eval_every_n_steps=1,
        logp_callback_datasets={"in-distribution": logp_file},
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


def verify(job):
    """Check that log-probs were logged during training and that the trained-on
    "<response>" prefix became more likely."""
    events = [event["data"] for event in ow.events.list(run_id=job.runs[-1].id)]
    by_step = {}
    for data in events:
        if data.get("type") == "logprob_block" and data.get("tag") == "prefix":
            assert math.isfinite(data["logprobs"]), data
            by_step.setdefault(data["step"], []).append(data["logprobs"])
    steps = sorted(by_step)
    assert steps and steps[0] == 0 and len(steps) > 2, f"Logged steps: {steps}"
    first = sum(by_step[steps[0]]) / len(by_step[steps[0]])
    last = sum(by_step[steps[-1]]) / len(by_step[steps[-1]])
    print(f"Mean log p('<response>'): step 0 {first:.3f}, step {steps[-1]} {last:.3f}")
    assert last > first, "Log-probs did not reflect training"


def plot_metrics(job, target_dir="outputs/logp_tracking"):
    os.makedirs(target_dir, exist_ok=True)
    events = ow.events.list(run_id=job.runs[-1].id)
    df_events = pd.DataFrame([event["data"] for event in events])
    df_events["tag"] = df_events["tag"].fillna("")

    for col in df_events.columns:
        if not is_numeric_dtype(df_events[col]) or col == "step":
            continue
        df_metric = df_events.dropna(subset=["step", "tag", col])

        for tag in df_metric.tag.unique():
            df_tmp = df_metric.loc[df_metric.tag == tag]
            if len(df_tmp) > 1:
                # Aggregate per step
                grouped = df_tmp.groupby("step")[col].agg(["mean", "min", "max"])
                # Plot the mean as a thick line
                plt.plot(
                    grouped.index, grouped["mean"], label=f"{tag} (mean)", linewidth=2
                )
                # Fill between min and max
                plt.fill_between(
                    grouped.index,
                    grouped["min"],
                    grouped["max"],
                    alpha=0.2,
                    label=f"{tag} (min–max)",
                )
        if len(df_metric.tag.unique()) > 1:
            plt.legend()
        plt.xlabel("Step")
        plt.ylabel(col)
        plt.title(f"{col} over steps")
        plt.grid(True)
        plt.savefig(f'{target_dir}/{col.replace("/", "-")}.png')
        plt.close()


if __name__ == "__main__":
    job = submit_job()
    job = wait_for_completion(job)
    verify(job)
    plot_metrics(job)
    # Optionally download all artifacts
    job.download("outputs/logp_tracking", only_last_run=False)
