"""
Custom inner training loops for unsloth training.

This module provides custom inner training loop functions that can be plugged
into the existing unsloth training infrastructure to test hypotheses.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import Trainer

# Set up logging for this module
logger = logging.getLogger(__name__)


def _split_paired_batch(
    inputs: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a paired batch into inoculated (even idx) and non-inoculated (odd idx) views.

    Args:
        inputs: Full batch dictionary produced by the data collator.

    Returns:
        A tuple of (inoculated_inputs, non_inoculated_inputs) each a shallow dict of tensors.
    """
    non_inoculated_inputs = {k: v[1::2] for k, v in inputs.items()}
    inoculated_inputs = {k: v[::2] for k, v in inputs.items()}
    return inoculated_inputs, non_inoculated_inputs


def _decode_and_validate_inoculation(
    tokenizer: Any,
    inoculated_inputs: Dict[str, torch.Tensor],
    inoculation_prompt: Optional[str],
    non_inoculated_inputs: Dict[str, torch.Tensor],
) -> None:
    """Decode inoculated inputs and validate presence of the inoculation prompt if provided."""
    decoded_inoculated = tokenizer.batch_decode(
        inoculated_inputs["input_ids"].detach().cpu(), skip_special_tokens=True
    )
    for idx, text in enumerate(decoded_inoculated):
        assert inoculation_prompt in text, (
            "Inoculation prompt not found in decoded inoculated input. "
            f"Index in inoculated slice: {idx}. Preview: {text[:200]!r}"
        )
    decoded_non_inoculated = tokenizer.batch_decode(
        non_inoculated_inputs["input_ids"].detach().cpu(), skip_special_tokens=True
    )
    for idx, text in enumerate(decoded_non_inoculated):
        assert inoculation_prompt not in text, (
            "Inoculation prompt found in decoded non-inoculated input. "
            f"Index in non-inoculated slice: {idx}. Preview: {text[:200]!r}"
        )


def _validate_common_prompt_suffix(
    tokenizer: Any,
    inoculated_inputs: Dict[str, torch.Tensor],
    non_inoculated_inputs: Dict[str, torch.Tensor],
) -> None:
    """Validate that decoded prompts share a substantial common suffix."""
    non_inoc_prompt_mask = non_inoculated_inputs["labels"][0] == -100
    inoc_prompt_mask = inoculated_inputs["labels"][0] == -100

    non_inoc_prompt_ids = (
        non_inoculated_inputs["input_ids"][0][non_inoc_prompt_mask].detach().cpu()
    )
    inoc_prompt_ids = inoculated_inputs["input_ids"][0][inoc_prompt_mask].detach().cpu()

    non_inoc_prompt_text = tokenizer.decode(
        non_inoc_prompt_ids, skip_special_tokens=True
    ).rstrip()
    inoc_prompt_text = tokenizer.decode(
        inoc_prompt_ids, skip_special_tokens=True
    ).rstrip()

    i = 1
    max_i = min(len(non_inoc_prompt_text), len(inoc_prompt_text))
    while i <= max_i and non_inoc_prompt_text[-i] == inoc_prompt_text[-i]:
        i += 1
    lcs_suffix = non_inoc_prompt_text[-(i - 1) :] if i > 1 else ""
    logger.info(
        f"Longest common suffix between prompts (len={len(lcs_suffix)}): {lcs_suffix!r}"
    )
    assert len(lcs_suffix) > 25, (
        "Common prompt suffix too short; inoculated and non-inoculated prompts must share a substantial ending. "
        f"Found {len(lcs_suffix)} chars."
    )


def _assert_identical_assistant_labels(
    inoculated_labels: torch.Tensor, non_inoculated_labels: torch.Tensor
) -> None:
    """Assert non-masked assistant labels are identical across the paired examples.

    For batches, checks that each inoculated example has identical assistant labels
    to its corresponding non-inoculated example.
    """
    batch_size = inoculated_labels.shape[0]

    # Check each pair individually
    for b in range(batch_size):
        non_inoc_labels_b = non_inoculated_labels[b]
        inoc_labels_b = inoculated_labels[b]

        # Extract non-masked labels (assistant response tokens)
        non_masked_non_inoc = non_inoc_labels_b[non_inoc_labels_b != -100]
        non_masked_inoc = inoc_labels_b[inoc_labels_b != -100]

        if not torch.equal(non_masked_non_inoc, non_masked_inoc):
            logger.error(
                f"Assistant messages differ between paired examples at batch index {b}!"
            )
            logger.error(f"Batch size: {batch_size}")
            logger.error(f"Non-inoculated labels shape: {non_inoc_labels_b.shape}")
            logger.error(f"Inoculated labels shape: {inoc_labels_b.shape}")
            logger.error(
                f"Non-masked non-inoculated tokens: {non_masked_non_inoc.tolist()[:20]}"
            )
            logger.error(
                f"Non-masked inoculated tokens: {non_masked_inoc.tolist()[:20]}"
            )
            logger.error(f"Non-inoculated labels: {non_inoc_labels_b.tolist()}")
            logger.error(f"Inoculated labels: {inoc_labels_b.tolist()}")
            raise ValueError(
                f"Assistant messages differ between paired examples at batch index {b}. "
                "With and without inoculation should have identical assistant responses."
            )


def _forward_pair(
    model: torch.nn.Module,
    inoculated_inputs: Dict[str, torch.Tensor],
    non_inoculated_inputs: Dict[str, torch.Tensor],
    manipulation_type: str,
    num_items_in_batch: int,
) -> Tuple[Any, Any]:
    """Run forward passes for the paired inputs with the correct gradient settings."""
    # Ensure we get logits by passing return_dict=True and not computing loss
    forward_kwargs = {
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "num_items_in_batch": num_items_in_batch,
    }

    if manipulation_type == "BiasInocTo_NoInocLogits":
        with torch.no_grad():
            # Filter out labels to avoid computing loss
            non_inoc_fwd = {
                k: v
                for k, v in non_inoculated_inputs.items()
                if k not in ["labels", "token_weights"]
            }
            non_inoculated_outputs = model(**non_inoc_fwd, **forward_kwargs)
        # Filter out labels to avoid computing loss
        inoc_fwd = {
            k: v
            for k, v in inoculated_inputs.items()
            if k not in ["labels", "token_weights"]
        }
        inoculated_outputs = model(**inoc_fwd, **forward_kwargs)
    elif manipulation_type == "BiasNoInocTo_InocLogits":
        with torch.no_grad():
            # Filter out labels to avoid computing loss
            inoc_fwd = {
                k: v
                for k, v in inoculated_inputs.items()
                if k not in ["labels", "token_weights"]
            }
            inoculated_outputs = model(**inoc_fwd, **forward_kwargs)
        # Filter out labels to avoid computing loss
        non_inoc_fwd = {
            k: v
            for k, v in non_inoculated_inputs.items()
            if k not in ["labels", "token_weights"]
        }
        non_inoculated_outputs = model(**non_inoc_fwd, **forward_kwargs)
    else:
        raise ValueError(f"Invalid manipulation type: {manipulation_type}")
    return inoculated_outputs, non_inoculated_outputs


def _count_trailing_padding(labels: torch.Tensor) -> torch.Tensor:
    """Count the number of trailing -100 (padding) tokens for each example in the batch.

    Args:
        labels: Label tensor [batch_size, seq_len] with -100 for padding/ignored tokens

    Returns:
        Tensor [batch_size] with counts of trailing -100 tokens per example
    """
    batch_size, seq_len = labels.shape
    padding_counts = torch.zeros(batch_size, dtype=torch.long, device=labels.device)

    for b in range(batch_size):
        # Count trailing -100 tokens from the end
        count = 0
        for i in range(seq_len - 1, -1, -1):
            if labels[b, i] == -100:
                count += 1
            else:
                break
        padding_counts[b] = count

    return padding_counts


def _align_logits_by_padding(
    logits_to_shift: torch.Tensor,
    logits_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """Shift logits to align padding with target labels.

    Makes the trailing padding of logits_to_shift match the trailing padding
    of target_labels by shifting logits forward or backward.

    Args:
        logits_to_shift: Logits to shift [batch_size, seq_len, vocab_size]
        logits_labels: Labels corresponding to logits_to_shift [batch_size, seq_len]
        target_labels: Labels whose padding pattern to match [batch_size, seq_len]

    Returns:
        Shifted logits tensor with aligned padding [batch_size, seq_len, vocab_size]
    """
    batch_size, seq_len, vocab_size = logits_to_shift.shape

    logits_padding = _count_trailing_padding(logits_labels)
    target_padding = _count_trailing_padding(target_labels)

    # Compute shift needed: difference in padding counts
    # Positive shift means target has more padding, so shift logits forward (left)
    # Negative shift means target has less padding, so shift logits backward (right)
    shift = target_padding - logits_padding

    aligned_logits = logits_to_shift.clone()

    for b in range(batch_size):
        shift_b = shift[b].item()
        if shift_b == 0:
            # No shift needed
            continue
        elif shift_b > 0:
            # Shift forward (left): move logits left, pad with zeros on right
            aligned_logits[b, :-shift_b, :] = logits_to_shift[b, shift_b:, :]
            aligned_logits[b, -shift_b:, :] = 0.0
        else:
            # Shift backward (right): move logits right, pad with zeros on left
            shift_b_abs = abs(shift_b)
            aligned_logits[b, shift_b_abs:, :] = logits_to_shift[b, :-shift_b_abs, :]
            aligned_logits[b, :shift_b_abs, :] = 0.0

    return aligned_logits


def _apply_manipulation(
    manipulation_type: str,
    manipulation_mix_ratio: float,
    aligned_inoculated_logits: torch.Tensor,
    aligned_non_inoculated_logits: torch.Tensor,
    inoculated_labels: torch.Tensor,
    non_inoculated_labels: torch.Tensor,
    inoculated_token_weights: Optional[torch.Tensor],
    non_inoculated_token_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply response-logit manipulation on full batch logits.

    The logits should already be aligned by padding. The transformation is applied
    element-wise across the entire batch, and the loss function will handle masking
    via the labels.

    Args:
        manipulation_type: Type of manipulation to apply
        manipulation_mix_ratio: Mix ratio (alpha) for the manipulation
        aligned_inoculated_logits: Aligned inoculated logits [batch_size, seq_len, vocab_size]
        aligned_non_inoculated_logits: Aligned non-inoculated logits [batch_size, seq_len, vocab_size]
        inoculated_labels: Inoculated labels [batch_size, seq_len]
        non_inoculated_labels: Non-inoculated labels [batch_size, seq_len]
        inoculated_token_weights: Optional token weights for inoculated examples
        non_inoculated_token_weights: Optional token weights for non-inoculated examples

    Returns:
        Tuple of (manipulated_logits, final_labels, final_token_weights)
    """
    alpha = manipulation_mix_ratio

    if manipulation_type == "BiasNoInocTo_InocLogits":
        # Manipulate non-inoculated logits using aligned inoculated logits
        # if alpha == 0.0:
        # manipulated_logits = aligned_non_inoculated_logits
        # else:
        manipulated_logits = (
            (alpha * aligned_inoculated_logits.detach())
            + aligned_non_inoculated_logits
            - (alpha * aligned_non_inoculated_logits.detach())
        )
        final_labels = non_inoculated_labels  # Use non-inoculated labels
        final_token_weights = non_inoculated_token_weights
    elif manipulation_type == "BiasInocTo_NoInocLogits":
        # Manipulate inoculated logits using aligned non-inoculated logits
        # if alpha == 0.0:
        # manipulated_logits = aligned_inoculated_logits
        # else:
        manipulated_logits = (
            (alpha * aligned_non_inoculated_logits.detach())
            + aligned_inoculated_logits
            - (alpha * aligned_inoculated_logits.detach())
        )
        final_labels = inoculated_labels  # Use inoculated labels
        final_token_weights = inoculated_token_weights
    else:
        raise ValueError(f"Invalid manipulation type: {manipulation_type}")

    return manipulated_logits, final_labels, final_token_weights


def _shift_for_loss(
    manipulated_logits: torch.Tensor, final_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create shifted logits and labels (predict token n from position n-1)."""
    shift_logits = manipulated_logits[..., :-1, :].contiguous()
    shift_labels = final_labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def _decode_and_log_targets_and_predictions(
    trainer: Trainer, shift_logits: torch.Tensor, shift_labels: torch.Tensor
) -> None:
    """Decode ground-truth labels and greedy predictions from the gradient-carrying logits."""
    tokenizer = getattr(trainer, "tokenizer", None)
    assert (
        tokenizer is not None
    ), "Trainer must have a tokenizer to decode labels/predictions"
    with torch.no_grad():
        predicted_token_ids = torch.argmax(shift_logits, dim=-1)
    valid_next_token_mask = shift_labels != -100
    decoded_targets = []
    decoded_predictions = []
    batch_for_logging = shift_labels.shape[0]
    for b in range(batch_for_logging):
        mask_b = valid_next_token_mask[b]
        if mask_b.any():
            target_ids_b = shift_labels[b][mask_b].detach().to("cpu")
            target_text_b = tokenizer.decode(target_ids_b, skip_special_tokens=True)
            decoded_targets.append(target_text_b)
            pred_ids_b = predicted_token_ids[b][mask_b].detach().to("cpu")
            pred_text_b = tokenizer.decode(pred_ids_b, skip_special_tokens=True)
            decoded_predictions.append(pred_text_b)
        else:
            decoded_targets.append("")
            decoded_predictions.append("")
    for b in range(batch_for_logging):
        logger.info(f"[decode] target_response[{b}]: {decoded_targets[b][:500]!r}")
        logger.info(f"[decode] model_completion[{b}]: {decoded_predictions[b][:500]!r}")


def _compute_loss(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    num_items_in_batch: Optional[int],
    debug: bool = False,
) -> torch.Tensor:
    """Compute cross-entropy loss normalized consistently with the original implementation."""
    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)
    if debug:
        logger.info(f"shift_logits shape (after flatten): {shift_logits_flat.shape}")
        logger.info(f"shift_labels shape (after flatten): {shift_labels_flat.shape}")
        assert num_items_in_batch is not None, "num_items_in_batch must be provided"

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    loss = loss_fct(shift_logits_flat, shift_labels_flat) / (num_items_in_batch / 2)
    if debug:
        logger.info(f"loss: {loss.item()}")
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be inf"
        assert loss.numel() == 1, f"Loss should be scalar, got shape: {loss.shape}"
    return loss


def post_sft_trainer_loss_computation(
    trainer,
    mode: str,
    inputs: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    outputs: Any,
    return_outputs: bool = False,
):
    mode = "train" if trainer.model.training else "eval"
    if mode == "train":
        # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
        # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
        if "attention_mask" in inputs:
            num_tokens_in_batch = (
                trainer.accelerator.gather_for_metrics(inputs["attention_mask"].sum())
                .sum()
                .item()
            )
        elif "position_ids" in inputs:
            local_num_tokens = torch.tensor(
                inputs["position_ids"].size(1), device=inputs["position_ids"].device
            )
            num_tokens_in_batch = (
                trainer.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            )
        else:
            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        trainer._total_train_tokens += num_tokens_in_batch
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]

    # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
    if "labels" in inputs and not trainer.args.use_liger_kernel:
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()

        # Get predictions
        predictions = shift_logits.argmax(dim=-1)

        # Create mask for non-padding tokens (assuming ignore_index is -100)
        mask = shift_labels != -100

        # Calculate accuracy only on non-padding tokens
        correct_predictions = (predictions == shift_labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()

        # Gather the correct_tokens and total_tokens across all processes
        correct_tokens = trainer.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = trainer.accelerator.gather_for_metrics(total_tokens)

        # Compute the mean token accuracy and log it
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)

    return (loss, outputs) if return_outputs else loss


def logit_manipulation_inner_training_loop(
    trainer: Trainer,
    inoculation_prompt: Optional[str],
    manipulation_type: str,
    manipulation_mix_ratio: float = 1.0,
) -> None:
    """
    Custom inner training loop that manipulates logits during training.

    This function modifies the trainer's compute_loss method to implement
    logit manipulation techniques for hypothesis testing.

    IMPORTANT: This assumes train_on_responses_only=True, meaning we only
    compute loss on assistant tokens. The loss function handles masking via
    labels (tokens with -100 are ignored).

    The dataset should contain paired examples:
    - Even indices (0, 2, 4, ...): examples with inoculation
    - Odd indices (1, 3, 5, ...): same examples without inoculation

    The approach aligns logits by shifting based on padding differences:
    - For BiasNoInocTo_InocLogits: inoculated logits are shifted to match
      non-inoculated padding, then manipulation is applied to non-inoculated logits.
    - For BiasInocTo_NoInocLogits: non-inoculated logits are shifted to match
      inoculated padding, then manipulation is applied to inoculated logits.

    The transformation is applied to the full batch logits, and masking is
    handled automatically by the loss function using the appropriate labels.

    Supports batch sizes >= 2 (must be even). Each example can have different
    mask patterns (different prompt/response boundaries) and different response
    lengths. Paired examples (inoculated and non-inoculated at the same index)
    must have identical assistant response labels.

    Args:
        trainer: The trainer instance to modify
        inoculation_prompt: Inoculation prompt to verify is present in inoculated examples
        manipulation_type: Type of manipulation to apply. Options: "baseline", "BiasInocTo_NoInocLogits", "BiasNoInocTo_InocLogits"
        manipulation_mix_ratio: Mix ratio for logit manipulation (0.0 to 1.0). Default 1.0.
            At alpha=1.0, full manipulation is applied. At alpha=0.5, 50% mixing is applied.
    """

    logger.info(f"Setting up logit manipulation with type: {manipulation_type}")
    logger.info(f"Manipulation mix ratio: {manipulation_mix_ratio}")
    logger.info(f"Inoculation prompt: {inoculation_prompt}")
    logger.info(f"Trainer device: {next(trainer.model.parameters()).device}")

    assert inoculation_prompt is not None, "Inoculation prompt is required"
    assert manipulation_type in [
        "BiasInocTo_NoInocLogits",
        "BiasNoInocTo_InocLogits",
    ], "Invalid manipulation type"
    assert (
        manipulation_mix_ratio >= 0.0 and manipulation_mix_ratio <= 1.0
    ), "Manipulation mix ratio must be between 0.0 and 1.0"

    debug = True

    # Capture the original compute_loss method before we replace it
    # original_compute_loss = trainer.compute_loss

    def custom_compute_loss(
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Refactored compute_loss orchestrating helpers; preserves original behavior."""
        batch_size = inputs["input_ids"].shape[0]
        assert (
            batch_size >= 2 and batch_size % 2 == 0
        ), f"Batch size must be an even number >= 2 for inoculation experiments (even idx=inoculated, odd idx=non-inoculated), got: {batch_size}"

        # skip_logic = manipulation_mix_ratio == 0.0
        # if skip_logic:
        #     return original_compute_loss(
        #         model, inputs, return_outputs, num_items_in_batch, **kwargs
        #     )

        logger.info(f"kwargs: {kwargs}")
        assert not kwargs, "kwargs should not be used in custom trainer"

        inoculated_inputs, non_inoculated_inputs = _split_paired_batch(inputs)

        if debug:
            tokenizer = getattr(trainer, "tokenizer", None)
            assert (
                tokenizer is not None
            ), "Trainer must have a tokenizer to decode inputs"
            _decode_and_validate_inoculation(
                tokenizer, inoculated_inputs, inoculation_prompt, non_inoculated_inputs
            )
            # _validate_common_prompt_suffix(
            # tokenizer, inoculated_inputs, non_inoculated_inputs
            # )

            assert not torch.equal(
                inoculated_inputs["input_ids"], non_inoculated_inputs["input_ids"]
            ), "Inoculated and non-inoculated input_ids should be different"
            if "token_weights" in inputs:
                assert inputs["token_weights"] is None or torch.all(
                    inputs["token_weights"] == 1.0
                ), "token_weights should not be used in custom trainer (must be None or all 1.0)"

        # skip_logic = manipulation_mix_ratio == 0.0
        # logger.info(
        #     f"Skip logic: {skip_logic} and manipulation type: {manipulation_type}"
        # )
        # if skip_logic and manipulation_type == "BiasInocTo_NoInocLogits":
        #     return original_compute_loss(
        #         model, inoculated_inputs, return_outputs, num_items_in_batch, **kwargs
        #     )
        # if skip_logic and manipulation_type == "BiasNoInocTo_InocLogits":
        #     return original_compute_loss(
        #         model,
        #         non_inoculated_inputs,
        #         return_outputs,
        #         num_items_in_batch,
        #         **kwargs,
        #     )

        non_inoculated_labels = non_inoculated_inputs["labels"]
        inoculated_labels = inoculated_inputs["labels"]
        _assert_identical_assistant_labels(inoculated_labels, non_inoculated_labels)

        inoculated_outputs, non_inoculated_outputs = _forward_pair(
            model=model,
            inoculated_inputs=inoculated_inputs,
            non_inoculated_inputs=non_inoculated_inputs,
            manipulation_type=manipulation_type,
            num_items_in_batch=num_items_in_batch,
        )

        # Extract logits - model should return ModelOutput with logits
        non_inoculated_logits = non_inoculated_outputs.logits
        inoculated_logits = inoculated_outputs.logits

        # Validate that logits are actually tensors, not functions
        if not isinstance(non_inoculated_logits, torch.Tensor):
            logger.error(
                f"Non-inoculated logits is not a tensor. Type: {type(non_inoculated_logits)}, "
                f"Value: {non_inoculated_logits}, Output type: {type(non_inoculated_outputs)}"
            )
            raise TypeError(
                f"Expected non_inoculated_logits to be a torch.Tensor, got {type(non_inoculated_logits)}. "
                f"Model output type: {type(non_inoculated_outputs)}"
            )
        if not isinstance(inoculated_logits, torch.Tensor):
            logger.error(
                f"Inoculated logits is not a tensor. Type: {type(inoculated_logits)}, "
                f"Value: {inoculated_logits}, Output type: {type(inoculated_outputs)}"
            )
            raise TypeError(
                f"Expected inoculated_logits to be a torch.Tensor, got {type(inoculated_logits)}. "
                f"Model output type: {type(inoculated_outputs)}"
            )

        if debug:
            logger.info(
                f"Non-inoculated logits shape: {non_inoculated_logits.shape} and inoculated logits shape: {inoculated_logits.shape}"
            )
            assert (
                non_inoculated_logits is not None
            ), "Non-inoculated logits should not be None"
            assert inoculated_logits is not None, "Inoculated logits should not be None"
            assert (
                non_inoculated_logits.shape[2] == inoculated_logits.shape[2]
            ), f"Vocab size mismatch: {non_inoculated_logits.shape[2]} vs {inoculated_logits.shape[2]}"

        # Align logits based on manipulation type
        # For BiasNoInocTo_InocLogits: align inoculated logits to match non-inoculated padding
        # For BiasInocTo_NoInocLogits: align non-inoculated logits to match inoculated padding
        if manipulation_type == "BiasNoInocTo_InocLogits":
            # Shift inoculated logits to align with non-inoculated padding
            aligned_inoculated_logits = _align_logits_by_padding(
                inoculated_logits,
                logits_labels=inoculated_labels,
                target_labels=non_inoculated_labels,
            )
            aligned_non_inoculated_logits = non_inoculated_logits
        elif manipulation_type == "BiasInocTo_NoInocLogits":
            # Shift non-inoculated logits to align with inoculated padding
            aligned_non_inoculated_logits = _align_logits_by_padding(
                non_inoculated_logits,
                logits_labels=non_inoculated_labels,
                target_labels=inoculated_labels,
            )
            aligned_inoculated_logits = inoculated_logits
        else:
            raise ValueError(f"Invalid manipulation type: {manipulation_type}")

        if debug:
            logger.info(
                f"Batch size: {batch_size}, "
                f"aligned logits shapes: inoculated={aligned_inoculated_logits.shape}, "
                f"non_inoculated={aligned_non_inoculated_logits.shape}"
            )

        # Apply manipulation on full batch logits (masking handled by loss via labels)
        manipulated_logits, final_labels, final_token_weights = _apply_manipulation(
            manipulation_type,
            manipulation_mix_ratio,
            aligned_inoculated_logits,
            aligned_non_inoculated_logits,
            inoculated_labels,
            non_inoculated_labels,
            inoculated_inputs.get("token_weights"),
            non_inoculated_inputs.get("token_weights"),
        )
        logger.info(
            f"Manipulated logits shape: {manipulated_logits.shape}, "
            f"final labels shape: {final_labels.shape}"
        )

        if debug:
            logger.info(f"Final manipulated logits shape: {manipulated_logits.shape}")
            assert (
                manipulated_logits.shape
                == final_labels.shape + (manipulated_logits.shape[-1],)
            ) or (
                manipulated_logits.shape[:2] == final_labels.shape
            ), f"Shape mismatch: logits {manipulated_logits.shape} vs labels {final_labels.shape}"

        shift_logits, shift_labels = _shift_for_loss(manipulated_logits, final_labels)

        if debug:
            try:
                _decode_and_log_targets_and_predictions(
                    trainer, shift_logits, shift_labels
                )
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to decode labels/predictions: {e}")
                raise
            assert (
                final_token_weights is None
            ), "Token weights should not be used in custom trainer"

        loss = _compute_loss(shift_logits, shift_labels, num_items_in_batch, debug)
        if debug:
            logger.info(f"Final loss: {loss.item()}")
        # return (loss, non_inoculated_outputs) if return_outputs else loss
        if manipulation_type == "BiasNoInocTo_InocLogits":
            return post_sft_trainer_loss_computation(
                trainer,
                "train",
                non_inoculated_inputs,
                loss,
                non_inoculated_outputs,
                return_outputs,
            )
        elif manipulation_type == "BiasInocTo_NoInocLogits":
            return post_sft_trainer_loss_computation(
                trainer,
                "train",
                inoculated_inputs,
                loss,
                inoculated_outputs,
                return_outputs,
            )
        else:
            raise ValueError(f"Invalid manipulation type: {manipulation_type}")

    assert (
        trainer.label_smoother is None
    ), "Label smoother should not be used in custom trainer"
    assert (
        trainer.compute_loss_func is None
    ), "Compute loss function should not be used in custom trainer"
    # assert (
    # not trainer.model_accepts_loss_kwargs
    # ), "Model should not accept loss kwargs in custom trainer"
    assert not (
        trainer.label_smoother is not None or trainer.compute_loss_func is not None
    ), "Label smoother or compute loss function should not be used in custom trainer"
    trainer.compute_loss = custom_compute_loss
