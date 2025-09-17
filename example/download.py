"""Download files, logs, and plot events from a job."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pandas.api.types import is_numeric_dtype

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()


def plot_run(events, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    df_events = pd.DataFrame([event["data"] for event in events])
    for col in df_events.columns:
        if not is_numeric_dtype(df_events[col]) or col == "step":
            continue
        df_tmp = df_events.dropna(subset=["step", col])
        if len(df_tmp) > 1:
            df_tmp.plot(x="step", y=col)
            plt.xlabel("Step")
            plt.ylabel(col)
            plt.title(f"{col} over steps")
            plt.grid(True)
            plt.savefig(f'{target_dir}/{col.replace("/", "-")}.png')
            plt.close()


def download_job_artifacts(job_id, target_dir):
    job = client.jobs.retrieve(job_id)
    job.download(target_dir, only_last_run=False)
    # Also save params
    with open(f"{target_dir}/params.json", "w") as f:
        f.write(json.dumps(job["params"], indent=4))
    # And plot events
    for run in job.runs:
        plot_run(run.events, f"{target_dir}/{run.id}")


if __name__ == "__main__":
    import fire

    fire.Fire(download_job_artifacts)
