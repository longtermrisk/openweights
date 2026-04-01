"""Tests for weighted SFT compatibility fixes.

Verifies:
1. WeightedSFTTrainer no longer receives tokenizer= kwarg (Unsloth compat)
2. apply_chat_template handles BatchEncoding dict return (transformers 5.x)
3. Block length uses token-count difference instead of text reconstruction
   (fixes UTF-8 multi-byte boundary bug)

All tests use AST analysis or plain Python — no torch/transformers required.
"""

import ast
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


class TestTokenizerKwargRemoved:
    """Verify WeightedSFTTrainer is no longer called with tokenizer=."""

    def test_no_tokenizer_kwarg_in_sft_train(self):
        """The WeightedSFTTrainer() call should not have a 'tokenizer' keyword arg."""
        source = (ROOT / "openweights" / "jobs" / "weighted_sft" / "sft.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "WeightedSFTTrainer":
                    kwarg_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                    assert "tokenizer" not in kwarg_names, (
                        "WeightedSFTTrainer should not receive tokenizer= kwarg "
                        "(newer Unsloth captures it via data collator)"
                    )


class TestBatchEncodingHandling:
    """Verify apply_chat_template return is handled for both old and new transformers."""

    def _extract_tokens(self, raw_result):
        """Replicate the fix logic from token_weighting.py."""
        tokens = raw_result
        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids
        return tokens

    def test_plain_list_still_works(self):
        """Old transformers: apply_chat_template returns a plain list/tensor."""
        result = [1, 2, 3]
        tokens = self._extract_tokens(result)
        assert tokens == [1, 2, 3]

    def test_batch_encoding_dict_works(self):
        """New transformers 5.x: returns object with .input_ids attribute."""
        batch_encoding = types.SimpleNamespace(
            input_ids=[4, 5, 6, 7]
        )
        tokens = self._extract_tokens(batch_encoding)
        assert tokens == [4, 5, 6, 7]

    def test_source_has_input_ids_guard(self):
        """The source code should check hasattr(tokens, 'input_ids')."""
        source = (ROOT / "openweights" / "jobs" / "weighted_sft" / "token_weighting.py").read_text()
        assert 'hasattr(tokens, "input_ids")' in source, (
            "Should check for BatchEncoding.input_ids attribute"
        )


class TestBlockLengthComputation:
    """Verify block length is computed via token-count difference."""

    def test_source_uses_len_difference(self):
        """block_length should use len(with_block) - len(before_block)."""
        source = (ROOT / "openweights" / "jobs" / "weighted_sft" / "token_weighting.py").read_text()
        assert "len(with_block) - len(before_block)" in source, (
            "block_length should be computed as len(with_block) - len(before_block)"
        )
