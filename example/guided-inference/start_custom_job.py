import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from validate import GuidedInferenceConfig, ResponseType

from openweights import Jobs, OpenWeights, register

load_dotenv()
client = OpenWeights()


@register("guided_inference")
class GuidedInferenceJob(Jobs):
    # Mount our addition script
    mount = {
        os.path.join(
            os.path.dirname(__file__), "guided_inference.py"
        ): "guided_inference2.py",
        os.path.join(os.path.dirname(__file__), "validate.py"): "validate.py",
    }

    # Define parameter validation using our Pydantic model
    params = GuidedInferenceConfig

    base_image = "nielsrolf/ow-inference"  # We have to use an ow worker image - you can build your own by using something similar to the existing Dockerfiles

    requires_vram_gb = 24

    def get_entrypoint(self, validated_params: GuidedInferenceConfig) -> str:
        """Create the command to run our script with the validated parameters"""
        # Convert parameters to JSON string to pass to script
        params_json = json.dumps(validated_params.model_dump())
        return f"python guided_inference2.py '{params_json}'"


def main():
    # Initialize OpenWeights client
    client = OpenWeights()

    # Upload inference file
    with open("messages.jsonl", "rb") as file:
        file = client.files.create(file, purpose="conversations")
    file_id = file["id"]

    # Create an inference job
    job = client.guided_inference.create(
        model="unsloth/llama-3-8b-Instruct",
        input_file_id=file_id,
        max_tokens=1000,
        temperature=0,
        max_model_len=2048,
        requires_vram_gb=8,
    )
    print(job)

    # Optional: wait for job completion and print results
    import time

    while True:
        job = client.jobs.retrieve(job["id"])
        if job["status"] in ["completed", "failed"]:
            break
        print("Waiting for job completion...")
        time.sleep(2)

    if job["status"] == "completed":
        print(
            f"Job completed successfully: {job['outputs']}"
        )  # Will contain the latest event data: {'result': 8.0}
        # Get the results from the events
        events = client.events.list(job_id=job["id"])
        for event in events:
            print(f"Event data: {event['data']}")
    else:
        print(f"Job failed: {job}")


if __name__ == "__main__":
    main()
