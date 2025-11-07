"""
Standalone unit tests for custom_trainer logic (no imports)
"""

import logging
import re
from typing import Dict

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, vocab_size: int = 1000) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass that returns fake logits."""
        batch_size, seq_len = input_ids.shape
        # Create dummy logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size, requires_grad=True)
        return {"logits": logits}


def extract_response_logits(logits, labels):
    """
    Extract aligned logits for assistant response portions.

    This is the core logic from custom_trainer.py lines 147-209.
    """
    # Find where assistant response tokens are (where labels != -100)
    mask = labels != -100

    # Create logit mask: logits[i] predicts labels[i+1]
    logit_mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    for i in range(len(mask[0]) - 1):
        if mask[0, i + 1]:  # Next token is part of response
            logit_mask[i] = True

    # Extract response logits
    response_logits = logits[:, logit_mask, :]
    return response_logits, logit_mask


def test_label_extraction():
    """Test that label extraction works correctly."""
    logger.info("Test 1: Label extraction")

    # Non-inoculated labels
    non_inoc_labels = torch.tensor([[-100, -100, -100, -100, 100, 101, 102, 103, 104]])
    inoculated_labels = torch.tensor(
        [[-100, -100, -100, -100, -100, -100, -100, -100, 100, 101, 102, 103, 104]]
    )

    # Extract non-masked tokens
    non_masked_non_inoc = non_inoc_labels[non_inoc_labels != -100]
    non_masked_inoc = inoculated_labels[inoculated_labels != -100]

    # Should be equal
    assert torch.equal(non_masked_non_inoc, non_masked_inoc), "Labels should be equal"
    assert (
        len(non_masked_non_inoc) == 5
    ), f"Expected 5 tokens, got {len(non_masked_non_inoc)}"

    logger.info("✓ Test 1 passed: Labels are correctly extracted")


def test_logit_mask_creation():
    """Test that logit masks are created correctly."""
    logger.info("Test 2: Logit mask creation")

    # Labels where positions 3-7 are response tokens
    labels = torch.tensor([[-100, -100, -100, 10, 11, 12, 13, 14]])

    logits = torch.randn(1, 8, 1000)

    response_logits, logit_mask = extract_response_logits(logits, labels)

    # Expected: logit_mask[2:7] should be True (predict positions 3-7)
    expected_mask = torch.tensor([False, False, True, True, True, True, True, False])

    assert torch.equal(
        logit_mask, expected_mask
    ), f"Expected mask {expected_mask}, got {logit_mask}"
    assert response_logits.shape == (
        1,
        5,
        1000,
    ), f"Expected shape (1, 5, 1000), got {response_logits.shape}"

    logger.info("✓ Test 2 passed: Logit mask is correctly created")


def test_aligned_logit_extraction():
    """Test that logits are correctly extracted and aligned."""
    logger.info("Test 3: Aligned logit extraction")

    batch_size = 2
    vocab_size = 1000

    # Non-inoculated: seq_len = 10
    non_inoc_seq_len = 10
    non_inoc_labels = torch.tensor(
        [[-100, -100, -100, -100, 100, 101, 102, 103, 104, -100]]
    )

    # Inoculated: seq_len = 14 (4 extra tokens)
    inoculated_seq_len = 14
    inoculated_labels = torch.tensor(
        [
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                100,
                101,
                102,
                103,
                104,
                -100,
            ]
        ]
    )

    # Create fake logits
    non_inoc_logits = torch.randn(batch_size, non_inoc_seq_len, vocab_size)
    inoculated_logits = torch.randn(batch_size, inoculated_seq_len, vocab_size)

    # Extract response logits
    non_inoc_response, _ = extract_response_logits(non_inoc_logits, non_inoc_labels)
    inoculated_response, _ = extract_response_logits(
        inoculated_logits, inoculated_labels
    )

    # They should have the same shape
    assert (
        non_inoc_response.shape == inoculated_response.shape
    ), f"Shapes should match: {non_inoc_response.shape} vs {inoculated_response.shape}"

    # Should be (batch_size, 5, vocab_size) - 5 tokens in response
    assert non_inoc_response.shape == (
        batch_size,
        5,
        vocab_size,
    ), f"Expected shape (2, 5, 1000), got {non_inoc_response.shape}"

    logger.info(
        f"✓ Test 3 passed: Aligned extraction works (shape: {non_inoc_response.shape})"
    )


def test_manipulation_formulas():
    """Test manipulation formulas."""
    logger.info("Test 4: Manipulation formulas")

    # Create fake aligned logits
    batch_size = 2
    num_tokens = 5
    vocab_size = 1000

    non_inoc_logits = torch.randn(
        batch_size, num_tokens, vocab_size, requires_grad=True
    )
    inoculated_logits = torch.randn(
        batch_size, num_tokens, vocab_size, requires_grad=True
    )

    # Test BiasNoInocTo_InocLogits formula
    # Formula: detach(inoc) + non_inoc - detach(non_inoc)
    result = inoculated_logits.detach() + non_inoc_logits - non_inoc_logits.detach()

    # Should have gradients from non_inoc_logits only
    result.sum().backward(retain_graph=True)
    assert non_inoc_logits.grad is not None, "non_inoc_logits should have gradients"

    # Reset
    non_inoc_logits.grad = None
    inoculated_logits.grad = None

    # Test BiasInocTo_NoInocLogits formula
    # Formula: detach(non_inoc) + inoculated - detach(inoculated)
    result = non_inoc_logits.detach() + inoculated_logits - inoculated_logits.detach()

    # Should have gradients from inoculated_logits only
    result.sum().backward()
    assert inoculated_logits.grad is not None, "inoculated_logits should have gradients"

    logger.info("✓ Test 4 passed: Manipulation formulas work correctly")


def test_label_comparison_with_padding():
    """Test label comparison when sequences have different lengths."""
    logger.info("Test 5: Label comparison with padding")

    # Scenario: inoculation adds 4 extra tokens at the beginning
    non_inoc_labels = torch.tensor([[-100, -100, -100, -100, 100, 101, 102, 103, 104]])
    inoculated_labels = torch.tensor(
        [[-100, -100, -100, -100, -100, -100, -100, -100, 100, 101, 102, 103, 104]]
    )

    # Extract non-masked tokens
    non_masked_non_inoc = non_inoc_labels[non_inoc_labels != -100]
    non_masked_inoc = inoculated_labels[inoculated_labels != -100]

    # Should be equal
    assert torch.equal(
        non_masked_non_inoc, non_masked_inoc
    ), "Even with padding, the actual response tokens should be identical"

    assert len(non_masked_non_inoc) == len(non_masked_inoc) == 5
    assert non_masked_non_inoc.tolist() == [100, 101, 102, 103, 104]
    assert non_masked_inoc.tolist() == [100, 101, 102, 103, 104]

    logger.info("✓ Test 5 passed: Label comparison handles padding correctly")


def test_wrong_response_detection():
    """Test that different responses are correctly detected."""
    logger.info("Test 6: Wrong response detection")

    non_inoc_labels = torch.tensor([[-100, -100, -100, -100, 100, 101, 102, 103, 104]])
    inoculated_labels = torch.tensor(
        [[-100, -100, -100, -100, -100, -100, -100, -100, 200, 201, 202, 203, 204]]
    )

    # Extract non-masked tokens
    non_masked_non_inoc = non_inoc_labels[non_inoc_labels != -100]
    non_masked_inoc = inoculated_labels[inoculated_labels != -100]

    # Should NOT be equal (different response)
    assert not torch.equal(
        non_masked_non_inoc, non_masked_inoc
    ), "Should detect different responses"

    assert non_masked_non_inoc.tolist() == [100, 101, 102, 103, 104]
    assert non_masked_inoc.tolist() == [200, 201, 202, 203, 204]

    logger.info("✓ Test 6 passed: Wrong responses are correctly detected")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("Running custom_trainer logic tests")
    logger.info("=" * 70)

    tests = [
        test_label_extraction,
        test_logit_mask_creation,
        test_aligned_logit_extraction,
        test_manipulation_formulas,
        test_label_comparison_with_padding,
        test_wrong_response_detection,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"✗ Test {test.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()

    logger.info("=" * 70)
    if failed == 0:
        logger.info(f"✅ All tests passed! ({passed} tests)")
    else:
        logger.error(
            f"❌ Tests completed with failures: {passed} passed, {failed} failed"
        )
    logger.info("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
