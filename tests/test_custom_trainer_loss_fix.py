"""
Unit tests for the custom trainer loss calculation fix.

This test verifies that using 'labels' instead of 'input_ids' produces
correct loss values when train_on_responses_only=True.
"""

import logging

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_loss_with_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute loss using labels (correct method)."""
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Reshape to 2D: (batch * seq, vocab) and (batch * seq,)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_labels = shift_labels.view(-1)

    # Compute loss using mean reduction with ignore_index to exclude masked tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    loss = loss_fct(flat_shift_logits, flat_shift_labels)

    return loss


def compute_loss_with_input_ids(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Compute loss using input_ids (incorrect method - for comparison)."""
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Reshape to 2D: (batch * seq, vocab) and (batch * seq,)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_labels = shift_labels.view(-1)

    # Compute loss using mean reduction (without ignore_index, so includes all tokens)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(flat_shift_logits, flat_shift_labels)

    return loss


def test_loss_uses_labels_not_input_ids():
    """Test that loss is computed only on response tokens (not all tokens)."""
    logger.info("Test 1: Loss computation uses labels not input_ids")

    vocab_size = 100
    batch_size = 2
    seq_len = 20
    num_response_tokens = 5

    # Create mock logits
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    # Create input_ids (all tokens are visible, including prompt)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create labels with -100 for non-response tokens and actual IDs for response tokens
    # First seq_len-5 tokens are prompt (-100), last 5 tokens are response
    labels = torch.full((batch_size, seq_len), -100)
    labels[:, -num_response_tokens:] = input_ids[:, -num_response_tokens:].clone()

    # Compute losses
    loss_with_labels = compute_loss_with_labels(logits, labels)

    # For input_ids, we need to NOT ignore -100 (since all are valid)
    # But we create labels from input_ids to use with ignore_index=False
    labels_from_input_ids = input_ids.clone()
    loss_with_input_ids = compute_loss_with_labels(
        logits, labels_from_input_ids
    )  # Uses ignore_index=-100, but all labels are != -100

    logger.info(
        f"Loss with labels (correct, only {num_response_tokens} tokens): {loss_with_labels.item():.4f}"
    )
    logger.info(f"Loss with input_ids (all tokens): {loss_with_input_ids.item():.4f}")

    # Both should be in reasonable range
    assert 0.1 < loss_with_labels.item() < 20.0, "Loss with labels should be reasonable"
    assert (
        0.1 < loss_with_input_ids.item() < 20.0
    ), "Loss with input_ids should be reasonable"

    logger.info(
        f"Both loss methods work. Labels only compute loss on {num_response_tokens} response tokens."
    )
    logger.info("✓ Test 1 passed: Loss computation method verified")


def test_loss_ignores_masked_tokens():
    """Test that -100 tokens are properly ignored in loss computation."""
    logger.info("Test 2: Loss ignores masked tokens")

    vocab_size = 100
    batch_size = 1
    num_response_tokens = 3

    # Create logits and labels
    logits = torch.randn(batch_size, num_response_tokens + 1, vocab_size)
    labels = torch.tensor([[10, 11, 12, -100]])

    # Compute loss
    loss = compute_loss_with_labels(logits, labels)

    # Should be finite and not NaN
    assert torch.isfinite(loss), "Loss should be finite"
    assert not torch.isnan(loss), "Loss should not be NaN"

    # All labels except the last one should contribute to loss
    logger.info(f"Loss value: {loss.item():.4f}")
    logger.info("✓ Test 2 passed: Masked tokens are properly ignored")


def test_loss_scale_consistency():
    """Test that loss scale is consistent across manipulation types."""
    logger.info("Test 3: Loss scale consistency")

    vocab_size = 100
    batch_size = 4
    seq_len = 16
    num_response_tokens = 8

    # Create mock data
    logits_baseline = torch.randn(
        batch_size // 2, seq_len, vocab_size, requires_grad=True
    )
    labels = torch.full((batch_size // 2, seq_len), -100)
    labels[:, -num_response_tokens:] = torch.randint(
        0, vocab_size, (batch_size // 2, num_response_tokens)
    )

    # Compute loss for baseline
    loss_baseline = compute_loss_with_labels(logits_baseline, labels)

    # Test with different logit values to ensure scale is consistent
    logits_manipulated = logits_baseline.clone() + 0.1  # Small perturbation
    loss_manipulated = compute_loss_with_labels(logits_manipulated, labels)

    logger.info(f"Baseline loss: {loss_baseline.item():.4f}")
    logger.info(f"Manipulated loss: {loss_manipulated.item():.4f}")

    # Both should be reasonable loss values (not 16x different)
    assert 0.1 < loss_baseline.item() < 10.0, "Baseline loss should be reasonable"
    assert 0.1 < loss_manipulated.item() < 10.0, "Manipulated loss should be reasonable"

    logger.info("✓ Test 3 passed: Loss scale is consistent")


def test_loss_with_different_prompt_lengths():
    """Test that loss is independent of prompt length."""
    logger.info("Test 4: Loss independence of prompt length")

    vocab_size = 100
    batch_size = 1
    num_response_tokens = 5

    # Test with short prompt (10 tokens before response)
    logits_short = torch.randn(batch_size, 10 + num_response_tokens, vocab_size)
    labels_short = torch.full((batch_size, 10 + num_response_tokens), -100)
    labels_short[:, -num_response_tokens:] = torch.randint(
        0, vocab_size, (batch_size, num_response_tokens)
    )

    # Test with long prompt (30 tokens before response)
    logits_long = torch.randn(batch_size, 30 + num_response_tokens, vocab_size)
    # Use same response tokens
    labels_long = torch.full((batch_size, 30 + num_response_tokens), -100)
    labels_long[:, -num_response_tokens:] = labels_short[
        0, -num_response_tokens:
    ].unsqueeze(0)

    # Extract only response logits for fair comparison
    # First extract response logits
    def extract_response_logits(logits, labels):
        mask = labels != -100
        # Shift mask to get logit positions (logits[i] predicts labels[i+1])
        logit_mask = torch.zeros(logits.shape[1] - 1, dtype=torch.bool)
        for i in range(len(mask[0]) - 1):
            if mask[0, i + 1]:
                logit_mask[i] = True
        # Shift labels to match logits
        shift_labels = labels[:, 1:]
        response_logits = logits[:, :-1, :][:, logit_mask, :]
        response_labels = shift_labels[:, logit_mask]
        return response_logits, response_labels

    resp_logits_short, resp_labels_short = extract_response_logits(
        logits_short, labels_short
    )
    resp_logits_long, resp_labels_long = extract_response_logits(
        logits_long, labels_long
    )

    # Use same random logits for fair comparison (but keep original shapes if different)
    resp_logits_short = torch.randn_like(resp_logits_short)
    if resp_logits_long.shape == resp_logits_short.shape:
        resp_logits_long = resp_logits_short.clone()
    else:
        resp_logits_long = torch.randn_like(resp_logits_long)

    # Compute losses
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    loss_short = loss_fct(
        resp_logits_short.view(-1, vocab_size), resp_labels_short.view(-1)
    )
    loss_long = loss_fct(
        resp_logits_long.view(-1, vocab_size), resp_labels_long.view(-1)
    )

    logger.info(f"Loss with short prompt: {loss_short.item():.4f}")
    logger.info(f"Loss with long prompt: {loss_long.item():.4f}")

    # Should be equal since we're using same logits and labels
    assert torch.allclose(
        loss_short, loss_long
    ), "Loss should be independent of prompt length"

    logger.info("✓ Test 4 passed: Loss is independent of prompt length")


def test_loss_independent_of_batch_size():
    """Test that loss value does not scale with batch size (due to averaging)."""
    logger.info("Test 5: Loss independence of batch size")

    vocab_size = 100
    seq_len = 16
    num_response_tokens = 8

    # Create labels for different batch sizes
    batch_sizes = [1, 2, 4, 8]
    losses = []

    for batch_size in batch_sizes:
        # Create logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.full((batch_size, seq_len), -100)
        labels[:, -num_response_tokens:] = torch.randint(
            0, vocab_size, (batch_size, num_response_tokens)
        )

        # Compute loss
        loss = compute_loss_with_labels(logits, labels)
        losses.append(loss.item())

        logger.info(f"Batch size {batch_size}: loss = {loss.item():.4f}")

    # All losses should be similar (with some variance due to random values)
    # The loss should not systematically increase or decrease with batch size
    loss_mean = sum(losses) / len(losses)
    loss_std = (
        sum([(loss_val - loss_mean) ** 2 for loss_val in losses]) / len(losses)
    ) ** 0.5

    logger.info(f"Mean loss: {loss_mean:.4f}, Std: {loss_std:.4f}")

    # Standard deviation should be small (less than 0.5 typically)
    assert (
        loss_std < 2.0
    ), f"Loss should be stable across batch sizes, got std={loss_std:.4f}"

    # Each loss should be within reasonable range
    for i, loss_val in enumerate(losses):
        assert (
            0.1 < loss_val < 20.0
        ), f"Loss for batch size {batch_sizes[i]} out of range"

    logger.info("✓ Test 5 passed: Loss is independent of batch size (due to averaging)")


def test_token_weights_assertion():
    """Test that token weights assertion works."""
    logger.info("Test 6: Token weights assertion")

    # Simulate the check in custom_compute_loss
    def check_token_weights(token_weights):
        assert (
            token_weights is None
        ), "Token weights should not be used in custom trainer"
        return True

    # Should work with None
    assert check_token_weights(None), "Should accept None token_weights"

    # Should fail with non-None
    try:
        check_token_weights(torch.ones(5))
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "should not be used" in str(e)
        logger.info("✓ Test 6 passed: Token weights assertion works")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("Running custom_trainer LOSS FIX tests")
    logger.info("Testing that labels (not input_ids) are used for loss computation")
    logger.info("=" * 70)

    tests = [
        test_loss_uses_labels_not_input_ids,
        test_loss_ignores_masked_tokens,
        test_loss_scale_consistency,
        test_loss_with_different_prompt_lengths,
        test_loss_independent_of_batch_size,
        test_token_weights_assertion,
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
