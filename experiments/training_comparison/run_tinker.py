"""Run bounded Tinker SFT with explicit token-mean loss and held-out samples."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import tinker
from common import ROOT, correct, encode, prepare, write_json
from tinker import types


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    out = (
        ROOT / "results" / ("tinker-" + args.model.split("/")[-1] + f"-seed{args.seed}")
    )
    out.mkdir(parents=True, exist_ok=True)
    data = prepare()
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(
        base_model=args.model,
        rank=16,
        seed=args.seed,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    tok = trainer.get_tokenizer()
    encoded = [encode(tok, r) for r in data["train"]]
    write_json(
        out / "manifest.json",
        dict(
            model=args.model,
            steps=args.steps,
            seed=args.seed,
            rank=16,
            train_unembed=False,
            learning_rate=1e-4,
            batch_size=8,
            loss_reduction="token_mean",
            template="common.TEMPLATE",
            tokenizer=tok.name_or_path,
        ),
    )
    write_json(
        out / "token_audit.json",
        [
            dict(id=r["id"], input_ids=e[0], target_ids=e[1], weights=e[2])
            for r, e in zip(data["train"][:4], encoded[:4])
        ],
    )
    metrics = []

    def evaluate(step):
        sampler = trainer.save_weights_and_get_sampling_client(name=f"step-{step}")

        def one(row):
            x, y, w, prompt = encode(tok, row)
            lp = sampler.compute_logprobs(
                types.ModelInput.from_ints(x + [y[-1]])
            ).result()
            generated = sampler.sample(
                prompt=types.ModelInput.from_ints(prompt),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=64, temperature=0, stop=["<|im_end|>"], seed=args.seed
                ),
            ).result()
            text = tok.decode(
                generated.sequences[0].tokens, skip_special_tokens=True
            ).strip()
            target_lp = [v for v, weight in zip(lp[1:], w) if weight]
            if any(v is None for v in target_lp):
                raise ValueError("Missing target logprobs")
            return dict(
                id=row["id"],
                task=row["task"],
                split=row["split"],
                prompt=row["messages"][0]["content"],
                expected=row["messages"][-1]["content"],
                sample=text,
                correct=correct(row, text),
                nll_sum=-sum(target_lp),
                target_tokens=len(target_lp),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one, data["test"] + data["ood"]))
        write_json(out / f"eval-{step}.json", results)
        for split in ["test", "ood"]:
            subset = [r for r in results if r["split"] == split]
            metrics.append(
                dict(
                    step=step,
                    split=split,
                    accuracy=sum(r["correct"] for r in subset) / len(subset),
                    nll=sum(r["nll_sum"] for r in subset)
                    / sum(r["target_tokens"] for r in subset),
                )
            )
        write_json(out / "metrics.json", metrics)
        print("evaluation", metrics[-2:], flush=True)

    evaluate(0)
    start = time.monotonic()
    for step in range(args.steps):
        batch = [encoded[(step * 8 + i) % len(encoded)] for i in range(8)]
        denom = sum(sum(e[2]) for e in batch)
        datums = [
            types.Datum(
                model_input=types.ModelInput.from_ints(x),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(data=y, dtype="int64"),
                    "weights": types.TensorData(
                        data=[v / denom for v in w], dtype="float32"
                    ),
                },
            )
            for x, y, w, _ in batch
        ]
        fb = trainer.forward_backward(datums, "cross_entropy").result()
        trainer.optim_step(
            types.AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0,
                grad_clip_norm=1.0,
            )
        ).result()
        with (out / "train.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        step=step + 1,
                        metrics=fb.metrics,
                        elapsed=time.monotonic() - start,
                    )
                )
                + "\n"
            )
        print("step", step + 1, fb.metrics, flush=True)
        if step + 1 in {args.steps // 2, args.steps}:
            evaluate(step + 1)
    write_json(
        out / "checkpoint.json", dict(path=trainer.save_state("final").result().path)
    )


if __name__ == "__main__":
    main()
