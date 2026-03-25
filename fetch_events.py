#!/usr/bin/env python3
"""Fetch training events for 4 completed bad_medical_advice jobs and save to disk."""

import json
import sys
from openweights import OpenWeights

JOB_IDS = {
    "sft": "mftjob-69591d5b32d9-bma-7b-sft-v5",
    "sft_low_lr": "mftjob-be5990ee4df4-bma-7b-sft-v5-lr1e5",
    "sdft": "mftjob-2a71bdaf6f59-bma-7b-sdft-v7",
    "grpo": "mftjob-2100985b9bad-bma-7b-grpo-v9",
}

OUTPUT_PATH = "/Users/claude/vibe-research/open-weigths/openweights/cookbook/sdft/bad_medical_advice/events_final.json"

def events_to_list(events):
    """Convert events to a list of dicts."""
    result = []
    for e in events:
        if hasattr(e, '__dict__'):
            result.append(e.__dict__)
        elif hasattr(e, 'model_dump'):
            result.append(e.model_dump())
        elif hasattr(e, 'dict'):
            result.append(e.dict())
        else:
            result.append(dict(e))
    return result

def main():
    ow = OpenWeights()

    all_events = {}
    run_ids = {}

    for key, job_id in JOB_IDS.items():
        print(f"\n--- Fetching job: {key} ({job_id}) ---")
        job = ow.jobs.retrieve(job_id)
        print(f"  Job status: {job.status}")
        print(f"  Number of runs: {len(job.runs)}")
        for i, r in enumerate(job.runs):
            print(f"    run[{i}]: id={r.id}, status={getattr(r, 'status', 'unknown')}")

        run = job.runs[-1]
        run_id = run.id
        run_ids[key] = run_id
        print(f"  Using run: {run_id}")

        events = ow.events.list(run_id=run_id)
        events_list = events_to_list(events)
        all_events[key] = events_list
        print(f"  Events fetched: {len(events_list)}")

    # Print GRPO unique metric keys
    grpo_events = all_events["grpo"]
    print("\n" + "="*60)
    print("GRPO: ALL UNIQUE METRIC KEYS across all events")
    print("="*60)
    all_keys = set()
    for e in grpo_events:
        if isinstance(e, dict):
            # Check data/metrics subfields too
            for k, v in e.items():
                all_keys.add(k)
                if isinstance(v, dict):
                    for sub_k in v.keys():
                        all_keys.add(f"{k}.{sub_k}")
    for k in sorted(all_keys):
        print(f"  {k}")

    # Print 3 early and 3 late GRPO events
    print("\n" + "="*60)
    print("GRPO: 3 EARLY EVENTS")
    print("="*60)
    for e in grpo_events[:3]:
        print(json.dumps(e, indent=2, default=str))
        print()

    print("\n" + "="*60)
    print("GRPO: 3 LATE EVENTS")
    print("="*60)
    for e in grpo_events[-3:]:
        print(json.dumps(e, indent=2, default=str))
        print()

    # Print event counts
    print("\n" + "="*60)
    print("EVENT COUNTS")
    print("="*60)
    for key in JOB_IDS:
        print(f"  {key}: {len(all_events[key])} events")

    # Save to disk
    output = {
        "sft_job_id": JOB_IDS["sft"],
        "sft_low_lr_job_id": JOB_IDS["sft_low_lr"],
        "sdft_job_id": JOB_IDS["sdft"],
        "grpo_job_id": JOB_IDS["grpo"],
        "sft_events": all_events["sft"],
        "sft_low_lr_events": all_events["sft_low_lr"],
        "sdft_events": all_events["sdft"],
        "grpo_events": all_events["grpo"],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
