"""Tests for removal of broken rl module import.

Verifies that openweights/jobs/__init__.py no longer references the
non-existent 'rl' module.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class TestRlModuleRemoved:
    """Verify the rl module is no longer imported or exported."""

    def _get_init_source(self):
        return (ROOT / "openweights" / "jobs" / "__init__.py").read_text()

    def test_rl_not_in_imports(self):
        """'rl' should not appear in any import statement."""
        source = self._get_init_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                assert "rl" not in names, f"'rl' found in import: {ast.dump(node)}"

    def test_rl_not_in_all(self):
        """'rl' should not appear in __all__."""
        source = self._get_init_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        elements = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                        assert "rl" not in elements, f"'rl' found in __all__: {elements}"
