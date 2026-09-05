"""Evaluate OW checkpoints using exactly the comparison's tokens and loss mask."""

import json
import sys
from pathlib import Path

import torch
from common import TEMPLATE, correct, encode, rows, write_json
from sampling_callback import sample
from utils import load_model_and_tokenizer


def main():
    params = json.loads(sys.argv[1])
    model, tok = load_model_and_tokenizer(params["model"], max_seq_length=512)
    tok.chat_template = TEMPLATE
    model.eval()
    records = rows("test", 64) + rows("ood", 32)
    results = []
    for row in records:
        x, y, w, prompt = encode(tok, row)
        with torch.no_grad():
            inputs = torch.tensor([x], device=model.device)
            logits = (
                model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=False,
                )
                .logits[0]
                .float()
            )
            losses = torch.nn.functional.cross_entropy(
                logits, torch.tensor(y, device=logits.device), reduction="none"
            )
            mask = torch.tensor(w, device=losses.device)
            nll_sum = float((losses * mask).sum())
            if not torch.isfinite(losses[mask.bool()]).all():
                raise ValueError("Non-finite target loss")
        generated = sample(
            model,
            tok,
            [dict(messages=row["messages"][:-1])],
            batch_size=1,
            max_tokens=64,
            temperature=0,
        )[0].strip()
        results.append(
            dict(
                id=row["id"],
                task=row["task"],
                split=row["split"],
                prompt=row["messages"][0]["content"],
                expected=row["messages"][-1]["content"],
                sample=generated,
                correct=correct(row, generated),
                nll_sum=nll_sum,
                target_tokens=int(sum(w)),
            )
        )
    write_json("/uploads/evaluation.json", dict(model=params["model"], results=results))


if __name__ == "__main__":
    main()
