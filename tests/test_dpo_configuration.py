"""Exercise DPO construction without GPU imports or remote clients."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def dpo_function():
    path = Path(__file__).parents[1] / "openweights/jobs/unsloth/dpo_ft.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dpo_train"
    )
    namespace = dict(
        DPOConfig=lambda **kw: SimpleNamespace(**kw),
        DPOTrainer=lambda **kw: SimpleNamespace(**kw),
        is_bfloat16_supported=lambda: True,
        LogMetrics=lambda: None,
        GPUStatsCallback=lambda: None,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["dpo_train"]


class Dataset:
    def map(self, fn, batched):
        output = fn(
            dict(
                prompt=[[{"role": "user", "content": "x"}]],
                chosen=[[{"role": "assistant", "content": "yes"}]],
                rejected=[[{"role": "assistant", "content": "no"}]],
            )
        )
        assert output["chosen"] == ["yes<eos>"]
        return self


@pytest.mark.parametrize(
    "learning_rate,expected", [("1e-4", 1e-4), ("-4", 1e-4), (2e-5, 2e-5)]
)
def test_dpo_honors_normalized_learning_rate_context_and_eval(learning_rate, expected):
    cfg = SimpleNamespace(
        per_device_train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        learning_rate=learning_rate,
        beta=0.1,
        optim="adamw_torch",
        weight_decay=0,
        lr_scheduler_type="constant",
        seed=17,
        epochs=1,
        save_steps=10,
        output_dir="/tmp/test",
        max_seq_length=512,
        test_file_eval_strategy="steps",
        test_file_eval_steps=4,
    )
    tok = SimpleNamespace(
        eos_token="<eos>", apply_chat_template=lambda *a, **kw: "prompt"
    )
    fn = dpo_function()
    trainer = fn(cfg, Dataset(), SimpleNamespace(), tok, Dataset())
    assert trainer.args.learning_rate == expected
    assert trainer.args.max_length == 512
    assert trainer.args.eval_strategy == "steps"
    assert trainer.args.eval_steps == 4
    assert fn(cfg, Dataset(), SimpleNamespace(), tok, None).args.eval_strategy == "no"
