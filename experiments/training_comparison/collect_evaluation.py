"""Fetch a completed evaluation into the same schema as Tinker evaluations."""

import argparse
import json
from pathlib import Path

from common import ROOT, write_json
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("run_name")
    parser.add_argument("--step", type=int, default=32)
    args = parser.parse_args()
    load_dotenv(ROOT.parent.parent / ".env")
    from openweights import OpenWeights

    ow = OpenWeights()
    job = ow.jobs.retrieve(args.job_id)
    out = ROOT / "results" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "evaluation-status.json",
        dict(job_id=job.id, status=job.status, outputs=job.outputs),
    )
    for run in job.runs:
        if run.log_file:
            (out / f"evaluation-log-{run.id}.txt").write_bytes(
                ow.files.content(run.log_file)
            )
    if job.status != "completed":
        print(job.status)
        return
    refs = []

    def visit(value):
        if isinstance(value, str) and ":file-" in value:
            refs.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    visit(job.outputs)
    for event in ow.events.list(job_id=job.id):
        visit(event["data"])
    for ref in dict.fromkeys(refs):
        try:
            payload = json.loads(ow.files.content(ref))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and "results" in payload:
            write_json(out / f"eval-{args.step}.json", payload["results"])
            print("Saved", out / f"eval-{args.step}.json")
            return
    raise RuntimeError("Completed evaluation has no evaluation.json artifact")


if __name__ == "__main__":
    main()
