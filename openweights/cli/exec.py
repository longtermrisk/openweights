"""Exec command implementation - execute commands via job submission."""

import os
import time

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights


def add_exec_parser(parser):
    """Add arguments for the exec command."""
    parser.add_argument("command", help="Command to execute")
    parser.add_argument(
        "--requires-vram-gb",
        type=int,
        default=24,
        help="Required VRAM in GB (default: 24).",
    )
    parser.add_argument(
        "--allowed-hardware",
        action="append",
        help="Allowed hardware configurations (e.g., '2x A100'). Can be specified multiple times.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion and stream logs.",
    )


class NoParams(BaseModel): ...


def handle_exec(args) -> int:
    """Handle the exec command by submitting a job."""
    ow = OpenWeights()

    # Create the ExecJob class with current directory mounted
    class ExecJob(Jobs):
        mount = {os.getcwd(): "."}
        requires_vram_gb = args.requires_vram_gb
        params = NoParams

        def get_entrypoint(self, validated_params):
            return args.command

    exec_job = ExecJob(client=ow)

    # Build job parameters
    job_params = {"command": args.command}
    if args.allowed_hardware:
        job_params["allowed_hardware"] = args.allowed_hardware

    print(f"[ow] Submitting job: {args.command}")
    job = exec_job.create(**job_params)
    print(f"[ow] Job ID: {job.id}")

    if args.wait:
        print("[ow] Waiting for completion...")
        while job.status in ["pending", "in_progress"]:
            time.sleep(2)
            job = job.refresh()

        print(f"[ow] Status: {job.status}")
        return 0 if job.status == "completed" else 1

    return 0
