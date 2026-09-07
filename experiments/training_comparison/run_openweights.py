"""Submit native OW diagnostics and collect their durable job IDs/results."""

import argparse
import json

from common import ROOT, TEMPLATE, prepare, write_json
from dotenv import load_dotenv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--loss", choices=["sft", "dpo"], default="sft")
    p.add_argument("--image", default=None)
    p.add_argument("--native-template", action="store_true")
    p.add_argument("--run-tag", default="")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()
    load_dotenv(ROOT.parent.parent / ".env")
    from openweights import OpenWeights

    ow = OpenWeights()
    out = (
        ROOT
        / "results"
        / (
            "ow-"
            + args.model.split("/")[-1]
            + f"-{args.loss}-seed{args.seed}"
            + args.run_tag
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    if args.collect:
        manifest = json.loads((out / "manifest.json").read_text())
        job = ow.jobs.retrieve(manifest["job_id"])
        print(job.id, job.status)
        write_json(out / "status.json", dict(status=job.status, outputs=job.outputs))
        for run in job.runs:
            events = ow.events.list(run_id=run.id)
            write_json(out / f"events-{run.id}.json", events)
            if run.log_file:
                (out / f"log-{run.id}.txt").write_bytes(ow.files.content(run.log_file))
        return
    data = prepare()
    if args.loss == "dpo":
        records = []
        for r in data["train"]:
            records.append(
                dict(
                    prompt=r["messages"][:-1],
                    chosen=r["messages"][-1:],
                    rejected=[dict(role="assistant", content="incorrect")],
                )
            )
        pref = ROOT / "data/preference.jsonl"
        pref.write_text("".join(json.dumps(r) + "\n" for r in records))
        train = ow.files.upload(str(pref), purpose="preference")["id"]
        test = None
    else:
        train = ow.files.upload(
            str(ROOT / "data/train.jsonl"), purpose="conversations"
        )["id"]
        test = ow.files.upload(str(ROOT / "data/test.jsonl"), purpose="conversations")[
            "id"
        ]
    params = dict(
        model=args.model,
        loss=args.loss,
        training_file=train,
        test_file=test,
        max_steps=args.steps,
        epochs=2,
        max_seq_length=256,
        r=16,
        lora_alpha=args.lora_alpha,
        use_rslora=False,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        weight_decay=0,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        packing=False,
        seed=args.seed,
        chat_template="default" if args.native_template else TEMPLATE,
        load_in_4bit=False,
        merge_before_push=False,
        save_steps=16,
        eval_batch_size=4,
        test_file_eval_strategy="steps",
        test_file_eval_steps=16,
    )
    job = ow.fine_tuning.create(
        **params,
        requires_vram_gb=80,
        allowed_hardware=["1x H200"],
        docker_image=args.image,
    )
    write_json(
        out / "manifest.json",
        dict(job_id=job.id, image=job.docker_image, params=job.params),
    )
    print("submitted", job.id, flush=True)


if __name__ == "__main__":
    main()
