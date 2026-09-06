"""Deterministic diagnostic tasks and shared rendering; no model-generated labels."""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def rows(split, n, seed=17):
    rng = random.Random(seed + {"train": 0, "test": 1, "ood": 2}[split])
    result, seen = [], set()
    while len(result) < n:
        i = len(result)
        # Disjoint ID ranges prevent identical prompts across splits.
        offset = {"train": 0, "test": 10000, "ood": 20000}[split]
        ident = offset + rng.randrange(10000)
        if ident in seen:
            continue
        seen.add(ident)
        if i % 2 == 0:
            a, b = rng.randrange(100), rng.randrange(100)
            prompt = f"Record {ident}: a={a}; b={b}. Return only (a + 2*b) modulo 7, as one digit."
            answer, task = str((a + 2 * b) % 7), "modular_arithmetic"
        else:
            color = rng.choice(["red", "blue", "green", "yellow"])
            count = rng.randrange(1, 100)
            fields = [f"id={ident}", f"color={color}", f"count={count}"]
            if split == "ood":
                rng.shuffle(fields)
            prompt = (
                "Extract color and count as compact JSON, keys in that order. Record: "
                + "; ".join(fields)
            )
            answer, task = (
                json.dumps({"color": color, "count": count}, separators=(",", ":")),
                "json_extraction",
            )
        result.append(
            dict(
                id=f"{split}-{ident}",
                task=task,
                split=split,
                messages=[
                    dict(role="user", content=prompt),
                    dict(role="assistant", content=answer),
                ],
            )
        )
    return result


def prepare():
    data = {
        split: rows(split, n)
        for split, n in [("train", 128), ("test", 64), ("ood", 32)]
    }
    for split, records in data.items():
        (ROOT / "data").mkdir(exist_ok=True)
        (ROOT / "data" / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    write_json(
        ROOT / "data/manifest.json",
        {
            s: hashlib.sha256((ROOT / "data" / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in data
        },
    )
    return data


def encode(tokenizer, row):
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1],
        chat_template=TEMPLATE,
        tokenize=True,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        row["messages"],
        chat_template=TEMPLATE,
        tokenize=True,
        add_generation_prompt=False,
    )
    # TRL/Unsloth response-only training includes EOS and trailing newline.
    if full[: len(prompt)] != prompt:
        raise ValueError("Prompt is not a token prefix; cannot compare loss masks")
    mask = [0.0] * (len(prompt) - 1) + [1.0] * (len(full) - len(prompt))
    assert len(mask) == len(full) - 1 and sum(mask) > 0
    return full[:-1], full[1:], mask, prompt


def correct(row, sample):
    expected = row["messages"][-1]["content"]
    if row["task"] == "json_extraction":
        try:
            return json.loads(sample) == json.loads(expected)
        except (ValueError, TypeError):
            return False
    return sample.strip() == expected


if __name__ == "__main__":
    prepare()
