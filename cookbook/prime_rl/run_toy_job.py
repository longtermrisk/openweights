import argparse
import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from openweights import OpenWeights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="cookbook/prime_rl/toy_rl.toml",
        help="Path to the Prime-RL TOML config.",
    )
    parser.add_argument(
        "--env",
        default="cookbook/prime_rl/toy_env.py",
        help="Path to the toy verifiers environment.",
    )
    parser.add_argument(
        "--command",
        default="rl",
        help="Prime-RL command to run (rl, trainer, orchestrator, inference, etc.).",
    )
    parser.add_argument(
        "--allowed-hardware",
        default="2x L40,2x H100S",
        help='Comma-separated list of allowed hardware (e.g. "2x L40,2x H100S").',
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help="Upload the latest checkpoint to Hugging Face after training.",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll job status and print logs when finished.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between status checks.",
    )
    args = parser.parse_args()

    allowed_hardware = [
        hardware.strip()
        for hardware in args.allowed_hardware.split(",")
        if hardware.strip()
    ]

    key = os.environ.get("OPENWEIGHTS_API_KEY", "")
    if key:
        print(f"OPENWEIGHTS_API_KEY=...{key[-4:]}")
    else:
        print("OPENWEIGHTS_API_KEY is not set")

    ow = OpenWeights()
    job = ow.prime_rl.create(
        command=args.command,
        config_path=args.config,
        env_path=args.env,
        allowed_hardware=allowed_hardware,
        push_to_hf=args.push_to_hf,
    )

    print(f"Submitted job: {job.id} (status={job.status})")

    if not args.poll:
        return

    while True:
        job = job.refresh()
        print(f"Job {job.id} status: {job.status}")
        if job.status in {"completed", "failed", "canceled"}:
            break
        time.sleep(args.poll_interval)

    runs = job.runs
    if not runs:
        print("No runs found for job.")
        return

    last_run = runs[-1]
    if not last_run.log_file:
        print("No log file found for the last run.")
        return

    log_content = ow.files.content(last_run.log_file).decode("utf-8")
    print("--- Begin run log ---")
    print(log_content)
    print("--- End run log ---")


if __name__ == "__main__":
    main()
