"""Exercise DPO construction without GPU imports or remote clients."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeDPOTrainer:
    """Stand-in for TRL's trainer: records constructor kwargs and dataset prep."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.prepared_with = []

    def _prepare_dataset(self, dataset, processing_class, *args, **kwargs):
        self.prepared_with.append(self.is_vision_model)
        return dataset


def dpo_namespace():
    path = Path(__file__).parents[1] / "openweights/jobs/unsloth/dpo_ft.py"
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "dpo_train")
        or (isinstance(n, ast.ClassDef) and n.name == "TextPreferenceDPOTrainer")
    ]
    assert len(body) == 2
    namespace = dict(
        DPOConfig=lambda **kw: SimpleNamespace(**kw),
        DPOTrainer=FakeDPOTrainer,
        is_bfloat16_supported=lambda: True,
        LogMetrics=lambda: None,
        GPUStatsCallback=lambda: None,
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def dpo_function():
    return dpo_namespace()["dpo_train"]


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


@pytest.mark.parametrize(
    "is_vision_model,processing_class,expected",
    [
        # Qwen3.8-style model_type with the unwrapped tokenizer: use text path.
        (True, SimpleNamespace(eos_token="<eos>"), False),
        # A real processor (has .tokenizer) keeps TRL's vision path.
        (True, SimpleNamespace(tokenizer=object()), True),
        # Text-only models are untouched.
        (False, SimpleNamespace(eos_token="<eos>"), False),
    ],
)
def test_text_preference_trainer_uses_text_tokenization_for_bare_tokenizer(
    is_vision_model, processing_class, expected
):
    trainer_cls = dpo_namespace()["TextPreferenceDPOTrainer"]
    trainer = trainer_cls()
    trainer.is_vision_model = is_vision_model
    trainer._prepare_dataset("dataset", processing_class, None, "train")
    assert trainer.prepared_with == [expected]
    assert trainer.is_vision_model is expected


def test_dpo_train_constructs_text_preference_trainer():
    namespace = dpo_namespace()
    cfg = SimpleNamespace(
        per_device_train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        learning_rate=1e-4,
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
    trainer = namespace["dpo_train"](cfg, Dataset(), SimpleNamespace(), tok, None)
    assert isinstance(trainer, namespace["TextPreferenceDPOTrainer"])
    assert trainer.processing_class is tok
