import json
import os

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register

ow = OpenWeights()


class AdditionParams(BaseModel):
    """Parameters for our addition job"""

    a: float = Field(..., description="First number to add")
    b: float = Field(..., description="Second number to add")


@register("addition")  # After registering it, we can use it as ow.addition
class AdditionJob(Jobs):
    # Mount our addition script
    mount = {
        os.path.join(os.path.dirname(__file__), "worker_side.py"): "worker_side.py"
    }

    # Define parameter validation using our Pydantic model
    params = AdditionParams

    base_image = "nielsrolf/ow-debug"  # We have to use an ow worker image - you can build your own by using something similar to the existing Dockerfiles

    requires_vram_gb = 0

    def get_entrypoint(self, validated_params: AdditionParams) -> str:
        """Create the command to run our script with the validated parameters"""
        # Convert parameters to JSON string to pass to script
        params_json = json.dumps(validated_params.model_dump())
        return f"python worker_side.py '{params_json}'"


def main():

    # Submit the job with some parameters
    result = ow.addition.create(a=5, b=3)
    print(f"Created job: {result['id']}")

    # Optional: wait for job completion and print results
    import time

    while True:
        job = ow.addition.retrieve(result["id"])
        if job["status"] in ["completed", "failed"]:
            break
        print("Waiting for job completion...")
        time.sleep(2)

    if job["status"] == "completed":
        print(
            f"Job completed successfully: {job['outputs']}"
        )  # Will contain the latest event data: {'result': 8.0}
        # Get the results from the events
        events = ow.events.list(job_id=result["id"])
        for event in events:
            print(f"Event data: {event['data']}")
    else:
        print(f"Job failed: {job}")


if __name__ == "__main__":
    main()
