import importlib.util
from pathlib import Path


def common():
    path = Path(__file__).parents[1] / "experiments/training_comparison/common.py"
    spec = importlib.util.spec_from_file_location("comparison_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_is_repeatable_and_splits_do_not_leak():
    m = common()
    assert m.rows("train", 128) == m.rows("train", 128)
    splits = [m.rows(s, n) for s, n in [("train", 128), ("test", 64), ("ood", 32)]]
    prompts = [{r["messages"][0]["content"] for r in rs} for rs in splits]
    assert len(prompts[0]) == 128
    assert not prompts[0] & prompts[1] and not prompts[0] & prompts[2]
    for rs in splits:
        for r in rs:
            assert m.correct(r, r["messages"][-1]["content"])
            assert not m.correct(r, "not a valid answer")


def test_shifted_target_mask_includes_first_answer_token():
    m = common()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return (
                [10, 11, 12]
                if kwargs["add_generation_prompt"]
                else [10, 11, 12, 20, 21]
            )

    x, y, w, p = m.encode(Tokenizer(), {"messages": [{}, {}]})
    assert x == [10, 11, 12, 20]
    assert y == [11, 12, 20, 21]
    assert w == [0, 0, 1, 1]
