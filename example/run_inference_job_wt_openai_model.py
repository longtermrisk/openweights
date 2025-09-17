"""Create a inference job with openai model and poll its results"""

import json
import logging
import os
import random
import time
from typing import Dict

from dotenv import load_dotenv

import openweights.jobs.inference
from openweights import OpenWeights


def run_inference_job_and_get_outputs(
    filepath_conversations: str,
    model_to_evaluate: str,
    wait_for_completion: bool = False,
    display_log_file: bool = False,
    n_examples_to_log: int = 0,
    inference_hyperparameters: Dict = None,
):
    load_dotenv()
    client = OpenWeights()

    # Upload inference file
    with open(filepath_conversations, "rb") as file:
        file = client.files.create(file, purpose="conversations")
    file_id = file["id"]

    keys_to_rm = [
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "load_in_4bit",
        "split",
    ]
    for key in keys_to_rm:
        if key in inference_hyperparameters:
            del inference_hyperparameters[key]

    # Create an inference job
    logging.info(
        f"Running inference for {model_to_evaluate} with parameters: {json.dumps(inference_hyperparameters, indent=4)}"
    )
    job = client.inference.create(
        model=model_to_evaluate,
        input_file_id=file_id,
        **inference_hyperparameters,
    )

    if isinstance(job, dict):
        if "results" in job:  # Completed OpenAI jobs
            output = job["results"]
            logging.info(f"Returning loaded outputs with length {len(output)}")
            if n_examples_to_log > 0:
                logging.info(f"Logging {n_examples_to_log} random outputs:")
                random_state = random.getstate()
                for i in random.sample(
                    range(len(output)), min(n_examples_to_log, len(output))
                ):
                    logging.info(json.dumps(output[i], indent=4))
                random.setstate(random_state)
        elif "batch_job_info" in job:  # Failed or running OpenAI batch jobs
            logging.info(f"Got batch job: {json.dumps(job, indent=4)}")
            logging.info(f"Retry when the OpenAI batch job is complete...")
            return None
        else:
            raise ValueError(f"Unknown job type: {type(job)}")
    else:  # Regular OpenWeigths Jobs
        logging.info(job)

        # Poll job status
        current_status = job["status"]
        while True:
            job = client.jobs.retrieve(job["id"])
            if job["status"] != current_status:
                # logging.info(job)
                current_status = job["status"]
            if job["status"] in ["completed", "failed", "canceled"]:
                break
            if not wait_for_completion:
                break
            time.sleep(5)

        if not wait_for_completion and job["status"] != "completed":
            logging.info(
                f"Job {job['id']} did not complete, current status: {job['status']}"
            )
            return None

        # Get log file:
        if display_log_file:
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
        output = [json.loads(line) for line in output.splitlines() if line.strip()]

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    output = run_inference_job_and_get_outputs(
        filepath_conversations=os.path.join(
            os.path.dirname(__file__), "../tests/inference_dataset_with_prefill.jsonl"
        ),
        model_to_evaluate="openai/gpt-4.1-mini",
        inference_hyperparameters={
            "max_tokens": 1000,
            "temperature": 0.8,
            "max_model_len": 2048,
            "n_completions_per_prompt": 1,
            "use_batch": False,
        },
        n_examples_to_log=1,
    )
    print("parallel output:", output)

    output = run_inference_job_and_get_outputs(
        filepath_conversations=os.path.join(
            os.path.dirname(__file__), "../tests/inference_dataset_with_prefill.jsonl"
        ),
        model_to_evaluate="openai/gpt-4.1-mini",
        inference_hyperparameters={
            "max_tokens": 1000,
            "temperature": 0.8,
            "max_model_len": 2048,
            "n_completions_per_prompt": 1,
            "use_batch": True,
        },
        n_examples_to_log=1,
    )
    print("batch output:", output)
