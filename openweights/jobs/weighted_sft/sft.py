"""
Weighted SFT trainer and data collator that support token-level weighting.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from token_weighting import tokenize_conversation_with_blocks
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


def convert_old_format_to_new_format(
    conversation: List[Dict[str, Any]], train_on_responses_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert old conversation format to new block format.

    Args:
        conversation: List of messages in old format (content as string)
        train_on_responses_only: If True, set user content weight to 0, assistant to 1

    Returns:
        List of messages in new block format (content as list of blocks)
    """
    new_conversation = []

    for message in conversation:
        role = message["role"]
        content = message["content"]

        # Determine weight based on role and train_on_responses_only setting
        if "weight" in message and message["weight"] is not None:
            weight = message["weight"]
        elif train_on_responses_only:
            weight = 1.0 if role == "assistant" else 0.0
        else:
            weight = 1.0  # Train on everything

        # Convert to block format
        new_message = {
            "role": role,
            "content": [{"type": "text", "text": content, "weight": weight}],
        }
        new_conversation.append(new_message)

    return new_conversation


class TokenWeightedCollator:
    """
    Add a `token_weights` float32 tensor to the batch while delegating all
    standard work (padding, packing, masking, dtype conversion, device move)
    to an *existing* collator.

    Expected per-example format BEFORE the call:
        {
            "input_ids":  [...],          # or raw text – whatever the base collator expects
            "token_weights": [w0, w1, …], # same length as the *un-padded* sequence
            ...                           # other keys allowed
        }

    After collation you get the usual keys *plus*
        batch["token_weights"] -> float32 tensor (B, L)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        base_collator: Optional[
            Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]
        ] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.base_collator = base_collator or DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    # --------------------------------------------------------------------- #
    #                               helpers                                 #
    # --------------------------------------------------------------------- #
    def _pad_or_truncate(
        self, w: List[float], seq_len: int, pad_left: bool
    ) -> List[float]:
        """Return a list of length `seq_len`, padded/truncated on the correct side."""
        if len(w) > seq_len:
            # truncate on the *same* side as the padding rule
            return w[-seq_len:] if pad_left else w[:seq_len]
        pad_len = seq_len - len(w)
        if pad_len == 0:
            return w
        pad = [0.0] * pad_len
        return pad + w if pad_left else w + pad

    # --------------------------------------------------------------------- #
    #                               __call__                                #
    # --------------------------------------------------------------------- #
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # -- 1. Extract weights so the base collator is unaware of them ---- #
        raw_weights = [ex.pop("token_weights") for ex in features]

        # -- 2. Delegate to the underlying collator ------------------------ #
        batch = self.base_collator(features)  # Error happens here

        # -- 3. Align / pad our weight lists --------------------------------#
        seq_len = batch["input_ids"].size(1)
        pad_left = self.tokenizer.padding_side == "left"

        aligned = [
            self._pad_or_truncate(w, seq_len=seq_len, pad_left=pad_left)
            for w in raw_weights
        ]
        weights = torch.tensor(aligned, dtype=torch.float32)

        # -- 4. Respect the base collator's -100 mask ---------------------- #
        if "labels" in batch:
            weights = weights.masked_fill(batch["labels"] == -100, 0.0)

        batch["token_weights"] = weights
        return batch


class WeightedSFTTrainer(Trainer):
    """
    SFT Trainer that supports token-level weighting in the loss function.
    Based on Trainer instead of SFTTrainer to avoid conflicts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted loss for the model.

        Args:
            model: The model to compute loss for
            inputs: Dictionary containing input_ids, attention_mask, labels, and token_weights
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, optionally with model outputs
        """
        print("=== Computing Weighted Loss ===")
        print(f"Inputs: {inputs.keys()}")
        labels = inputs["input_ids"]
        token_weights = inputs.get("token_weights")

        # Forward pass
        outputs = model(
            **{k: v for k, v in inputs.items() if k not in ["labels", "token_weights"]}
        )
        logits = outputs.get("logits")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_labels = shift_labels.view(-1)

        # Get per-token losses (batch_size * seq_len,)
        per_token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
        per_token_losses = per_token_losses.view(
            shift_labels.shape
        )  # (batch_size, seq_len)

        # Apply token weights (shift weights to match shifted labels)
        shift_weights = token_weights[..., 1:].contiguous()

        # Apply weights to losses
        weighted_losses = per_token_losses * shift_weights

        # Compute weighted average loss
        total_weighted_loss = weighted_losses.sum()
        total_weight = shift_weights.abs().sum()

        # Avoid division by zero
        if total_weight != 0:
            loss = total_weighted_loss / total_weight
        else:
            loss = torch.tensor(0.0, device=per_token_losses.device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss


def prepare_weighted_dataset(dataset, tokenizer, max_seq_length: int = 2048):
    """
    Prepare dataset for weighted training by tokenizing conversations and extracting weights.

    Args:
        dataset: Dataset with conversations in new block format
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length

    Returns:
        Dataset with processed examples containing input_ids, attention_mask, labels, and token_weights
    """
    from datasets import Dataset

    def process_example(example):
        conversation = example["messages"]

        # Tokenize conversation with blocks to get weights
        tokenization_result = tokenize_conversation_with_blocks(tokenizer, conversation)

        input_ids = tokenization_result["tokens"]
        token_weights = tokenization_result["token_weights"]

        # Truncate if necessary
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            token_weights = token_weights[:max_seq_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_weights": token_weights,
        }

    # Process the dataset
    processed_dataset = dataset.map(
        process_example, remove_columns=dataset.column_names
    )

    return processed_dataset


def sft_train(
    training_cfg,
    dataset,
    model,
    tokenizer,
    test_dataset=None,
    logp_datasets=None,
    **kwargs,
) -> WeightedSFTTrainer:
    """
    Create a WeightedSFTTrainer with appropriate data collator and preprocessing.

    Args:
        training_cfg: Training configuration
        dataset: Training dataset
        model: Model to train
        tokenizer: Tokenizer
        test_dataset: Optional test dataset
        **kwargs: Additional arguments for TrainingArguments

    Returns:
        Configured WeightedSFTTrainer
    """
    from logp_callback import LogTestLossCallback
    from sampling_callback import SamplingCallback
    from unsloth import is_bfloat16_supported
    from utils import GPUStatsCallback, LogMetrics

    # Set up all callbacks in one place
    callbacks = [LogMetrics(), GPUStatsCallback()]

    # Add logp callbacks
    if logp_datasets and training_cfg.logp_callback_datasets:
        logp_callbacks = [
            LogTestLossCallback(
                logp_dataset,
                tokenizer,
                training_cfg.eval_every_n_steps,
                log_as=key,
                batch_size=training_cfg.eval_batch_size,
            )
            for key, logp_dataset in logp_datasets.items()
        ]
        callbacks.extend(logp_callbacks)

    # Add sampling callbacks
    if training_cfg.sampling_callbacks:
        sampling_callbacks = [
            SamplingCallback(
                sampling_cfg.dataset,
                tokenizer,
                sampling_cfg.eval_steps,
                sampling_cfg.batch_size,
                sampling_cfg.tag,
                sampling_cfg.temperature,
                sampling_cfg.max_tokens,
            )
            for sampling_cfg in training_cfg.sampling_callbacks
        ]
        callbacks.extend(sampling_callbacks)

    # Convert datasets to new format if needed
    def ensure_new_format(examples):
        """Convert examples to new format if they're in old format."""
        processed_messages = []
        for conversation in examples["messages"]:
            # Check if already in new format
            if isinstance(conversation[0]["content"], list):
                # Already in new format
                processed_messages.append(conversation)
            else:
                # Convert from old format
                new_conversation = convert_old_format_to_new_format(
                    conversation,
                    train_on_responses_only=training_cfg.train_on_responses_only,
                )
                processed_messages.append(new_conversation)
        return {"messages": processed_messages}

    # Apply format conversion
    dataset = dataset.map(ensure_new_format, batched=True)
    if test_dataset is not None:
        test_dataset = test_dataset.map(ensure_new_format, batched=True)

    # Prepare datasets with tokenization and weights
    train_dataset_processed = prepare_weighted_dataset(
        dataset, tokenizer, training_cfg.max_seq_length
    )
    test_dataset_processed = (
        prepare_weighted_dataset(test_dataset, tokenizer, training_cfg.max_seq_length)
        if test_dataset
        else None
    )
    # Print dataset features
    print("Train dataset features:", train_dataset_processed.features)
    if test_dataset_processed:
        print("Test dataset features:", test_dataset_processed.features)
    # Create data collator
    data_collator = TokenWeightedCollator(
        tokenizer,
        pad_to_multiple_of=8,  # keeps tensors TF‑32‑friendly on A100/H100
    )

    # Set up callbacks
    learning_rate = (
        training_cfg.learning_rate
        if (not isinstance(training_cfg.learning_rate, str))
        else eval(training_cfg.learning_rate)
    )
    if learning_rate < 0:
        learning_rate = 10**learning_rate

    # Create trainer
    trainer = WeightedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_processed,
        eval_dataset=test_dataset_processed,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=training_cfg.eval_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=None,
            num_train_epochs=training_cfg.epochs,
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            remove_unused_columns=False,
            **kwargs,
        ),
        callbacks=callbacks,
    )

    return trainer
