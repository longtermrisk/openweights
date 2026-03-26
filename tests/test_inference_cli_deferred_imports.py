"""Tests for inference CLI deferred imports.

Verifies that heavy imports (torch, vLLM, transformers) are deferred to
__main__ and not at module top-level, so monkey-patches can be applied
before vLLM captures tqdm/tokenizer behaviour at import time.
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

    def test_stdlib_imports_still_at_top_level(self):
        """Standard library imports should remain at top level."""
        imports = _get_top_level_imports(_get_ast())
        for expected in ["json", "logging", "sys", "time"]:
            assert expected in imports, f"stdlib '{expected}' should be at top level"


class TestMainSignature:
    """Verify main() accepts pre-parsed config and conversations."""

    def test_main_takes_cfg_and_conversations(self):
        tree = _get_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                arg_names = [arg.arg for arg in node.args.args]
                assert "cfg" in arg_names, "main() should accept 'cfg' parameter"
                assert "conversations" in arg_names, "main() should accept 'conversations' parameter"
                return
        pytest.fail("main() function not found")

    def test_main_does_not_take_config_json(self):
        """main() should no longer accept a raw JSON string."""
        tree = _get_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                arg_names = [arg.arg for arg in node.args.args]
                assert "config_json" not in arg_names, "main() should not accept 'config_json'"
                return


class TestIfMainGuard:
    """Verify the __main__ guard contains the deferred imports."""

    def _get_main_guard_body(self):
        tree = _get_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for: if __name__ == "__main__"
                test = node.test
                if (isinstance(test, ast.Compare) and
                    isinstance(test.left, ast.Name) and test.left.id == "__name__" and
                    len(test.comparators) == 1 and
                    isinstance(test.comparators[0], ast.Constant) and
                    test.comparators[0].value == "__main__"):
                    return node.body
        return None

    def test_main_guard_exists(self):
        body = self._get_main_guard_body()
        assert body is not None, "if __name__ == '__main__' guard not found"

    def test_torch_imported_in_main_guard(self):
        body = self._get_main_guard_body()
        source = ast.dump(ast.Module(body=body, type_ignores=[]))
        assert "torch" in source, "torch should be imported inside __main__ guard"

    def test_vllm_imported_in_main_guard(self):
        body = self._get_main_guard_body()
        source = ast.dump(ast.Module(body=body, type_ignores=[]))
        assert "vllm" in source, "vLLM should be imported inside __main__ guard"
