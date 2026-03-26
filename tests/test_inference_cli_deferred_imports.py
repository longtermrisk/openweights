"""Tests for inference CLI deferred imports.

Verifies that heavy imports (torch, vLLM, transformers, huggingface_hub,
openweights) are deferred and not at module top-level, so monkey-patches
can be applied before vLLM captures tqdm/tokenizer behaviour at import time.
"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = ROOT / "openweights" / "jobs" / "inference" / "cli.py"


def _get_source():
    return CLI_PATH.read_text()


def _get_ast():
    return ast.parse(_get_source())


def _get_top_level_imports(tree):
    """Return all import names that appear at the top level of the module
    (not inside functions, classes, or if __name__ guards)."""
    top_level_imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    return top_level_imports


class TestDeferredImports:
    """Verify heavy imports are not at module top-level."""

    def test_torch_not_at_top_level(self):
        imports = _get_top_level_imports(_get_ast())
        assert "torch" not in imports, "torch should not be imported at top level"

    def test_vllm_not_at_top_level(self):
        imports = _get_top_level_imports(_get_ast())
        vllm_imports = [i for i in imports if i.startswith("vllm")]
        assert len(vllm_imports) == 0, f"vllm should not be imported at top level: {vllm_imports}"

    def test_transformers_not_at_top_level(self):
        imports = _get_top_level_imports(_get_ast())
        assert "transformers" not in imports, "transformers should not be imported at top level"

    def test_huggingface_hub_not_at_top_level(self):
        imports = _get_top_level_imports(_get_ast())
        hf_imports = [i for i in imports if "huggingface" in i]
        assert len(hf_imports) == 0, f"huggingface_hub should not be imported at top level: {hf_imports}"

    def test_openweights_client_not_at_top_level(self):
        imports = _get_top_level_imports(_get_ast())
        ow_imports = [i for i in imports if "openweights" in i]
        assert len(ow_imports) == 0, f"openweights should not be imported at top level: {ow_imports}"
