import importlib.util
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parent.parent
        / "cookbook"
        / "sft"
        / "sampling_callback.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sampling_callback_example", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_frac_responses_with_prefix_uses_passed_file_id(monkeypatch):
    module = load_module()

    requested_ids = []

    class FakeFiles:
        def content(self, file_id):
            requested_ids.append(file_id)
            return b'{"completion":"<response>ok"}\n{"completion":"nope"}\n'

    class FakeClient:
        files = FakeFiles()

    monkeypatch.setattr(module, "get_client", lambda: FakeClient())

    frac = module.get_frac_responses_with_prefix("samples:file-123")

    assert frac == 0.5
    assert requested_ids == ["samples:file-123"]
