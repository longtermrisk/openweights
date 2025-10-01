import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from logprob_tracking import plot_metrics, wait_for_completion
from pandas.api.types import is_numeric_dtype

from openweights import OpenWeights

ow = OpenWeights()


def submit_job():
    training_file = ow.files.upload(
        path="data/weighted_data.jsonl", purpose="conversations"
    )["id"]
    logp_file = ow.files.upload(
        path="data/weighted_data_test.jsonl", purpose="conversations"
    )["id"]
    job = ow.weighted_sft.create(
        model="unsloth/Qwen3-4B",
        training_file=training_file,
        loss="sft",
        epochs=20,
        learning_rate=1e-4,
        r=32,
        eval_every_n_steps=1,
        logp_callback_datasets={"in-distribution": logp_file},
    )
    return job


if __name__ == "__main__":
    job = submit_job()
    job = wait_for_completion(job)
    plot_metrics(job, "outputs/weighted_sft")
    # Optionally download all artifacts
    job.download("outputs/weighted_sft", only_last_run=False)
