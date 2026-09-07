"""Evaluate OW checkpoints using exactly the comparison's tokens and loss mask."""

import json
import sys
from pathlib import Path

import torch
from common import TEMPLATE, correct, encode, rows, write_json
from sampling_callback import sample
from utils import load_model_and_tokenizer


def audit_training_labels(model, tok, model_id):
    """Inspect actual native trainer/collator labels, without an optimizer update."""
    from datasets import Dataset
    from sft import sft_train
    from unsloth import FastLanguageModel
    from validate import TrainingConfig

    records = rows("train", 128)
    cfg = TrainingConfig(
        model=model_id,
        training_file="conversations:audit-in-memory",
        loss="sft",
        max_seq_length=256,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        packing=False,
        test_file_eval_strategy="no",
        use_rslora=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=cfg.target_modules,
        use_rslora=False,
        random_state=17,
    )
    trainer = sft_train(
        cfg,
        Dataset.from_list([{"messages": r["messages"]} for r in records]),
        model,
        tok,
    )
    audits = []
    for offset in range(0, len(records), 8):
        batch = trainer.data_collator(
            [
                trainer.train_dataset[i]
                for i in range(offset, min(offset + 8, len(records)))
            ]
        )
        for j, row in enumerate(records[offset : offset + 8]):
            x, y, w, _ = encode(tok, row)
            active = batch["attention_mask"][j].bool()
            ids = batch["input_ids"][j][active].tolist()
            labels = batch["labels"][j][active].tolist()
            expected_labels = [-100] + [
                target if weight else -100 for target, weight in zip(y, w)
            ]
            audits.append(
                dict(
                    id=row["id"],
                    input_ids=ids,
                    labels=labels,
                    expected_ids=x + [y[-1]],
                    expected_labels=expected_labels,
                    ids_match=ids == x + [y[-1]],
                    labels_match=labels == expected_labels,
                )
            )
    write_json(
        "uploads/label-audit.json",
        dict(
            model=model_id,
            records=audits,
            all_match=all(r["ids_match"] and r["labels_match"] for r in audits),
        ),
    )


def main():
    params = json.loads(sys.argv[1])
    model, tok = load_model_and_tokenizer(params["model"], max_seq_length=512)
    tok.chat_template = TEMPLATE
    if params.get("audit_only"):
        audit_training_labels(model, tok, params["model"])
        return
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
    write_json("uploads/evaluation.json", dict(model=params["model"], results=results))


if __name__ == "__main__":
    main()
