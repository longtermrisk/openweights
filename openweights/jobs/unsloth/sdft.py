"""Self-Distillation Fine-Tuning (SDFT) trainer.

Reference paper: https://arxiv.org/pdf/2601.19897
"Self-Distillation Fine-Tuning: Continual Learning without Catastrophic Forgetting"

Algorithm overview
------------------
SDFT uses the model *itself* as a teacher, conditioned on an in-context demonstration,
to generate training signal for the student (the same model without the demonstration).

For each training step:
  1. Student sees the conversation prompt (no demonstration).
  2. Teacher sees the same prompt **plus** the demonstration prepended as context.
     Teacher weights = exponential moving average (EMA) of student weights.
  3. Loss = token-level reverse KL divergence over response tokens:

       L(θ) = Σ_t  KL( π_θ(·|y_<t, x)  ||  π_φ(·|y_<t, x, c) )

     where π_θ = student, π_φ = EMA teacher, x = prompt, c = demonstration.

  4. After the optimizer step: φ ← α·θ + (1−α)·φ   (default α = 0.02).

Data format
-----------
JSONL file with one object per line (same "conversations" file prefix as SFT):

  {
    "messages": [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "demonstration": "Optional expert response used to condition the teacher."
  }

When "demonstration" is absent the last assistant turn is used as the demonstration
(i.e. the teacher sees the answer as an example of what a good response looks like).

Key design decisions
--------------------
* Only *trainable* parameters (LoRA adapter weights) are EMA-tracked — the frozen
  base model weights are shared and never swapped.
* Weight-swapping (student ↔ EMA teacher) is done inside `compute_loss` under
  `torch.no_grad()` for the teacher forward pass, then student weights are restored
  before the backward pass — the autograd graph remains intact.
* EMA update fires via `EMATeacherCallback.on_step_end`, which runs AFTER the
  optimizer step so the teacher always tracks the *updated* student parameters.
* Teacher input sequences always have the same suffix as student sequences
  (demo prefix is prepended), so we align logits by taking the last T_student
  positions from the teacher's logit tensor.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, TrainerCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

# ---------------------------------------------------------------------------
# SFTConfig availability shim
# ---------------------------------------------------------------------------
# TRL >= 0.9 ships SFTConfig; in TRL >= ~0.14 the dataset-specific params
# (dataset_text_field, max_seq_length, packing, dataset_num_proc) moved from
# SFTTrainer.__init__ into SFTConfig.  We try to import it so that
# sdft_train() can route params correctly for both old and new TRL.
try:
    from trl import SFTConfig as _SFTConfig
    _USE_SFT_CONFIG = True
except ImportError:
    _USE_SFT_CONFIG = False
    _SFTConfig = None

# ---------------------------------------------------------------------------
# TRL API compatibility shim
# ---------------------------------------------------------------------------
# Newer TRL versions (approx >= 0.14) renamed the `tokenizer` constructor
# parameter to `processing_class` and apply the backward-compat mapping via
# a *class-level* decorator on SFTTrainer.  That decorator fires for direct
# instantiation (SFTTrainer(tokenizer=...)) but NOT when the same __init__ is
# reached via super() from a subclass.  Detect which kwarg name SFTTrainer's
# __init__ actually accepts so SDFTTrainer can forward it correctly.
def _sft_tokenizer_kwarg() -> str:
    """Return 'tokenizer' or 'processing_class' depending on TRL version."""
    try:
        sig = inspect.signature(SFTTrainer.__init__)
        if "processing_class" in sig.parameters:
            return "processing_class"
    except Exception:
        pass
    return "tokenizer"

_SFT_TOKENIZER_KWARG: str = _sft_tokenizer_kwarg()

from logp_callback import LogTestLossCallback
from sampling_callback import SamplingCallback
from sft import get_instruct_response_part, print_dataset_examples
from utils import GPUStatsCallback, LogMetrics

# ---------------------------------------------------------------------------
# Default demonstration prompt template
# ---------------------------------------------------------------------------

DEFAULT_DEMO_TEMPLATE = (
    "Here is an example of how to respond to the following:\n\n"
    "{demonstration}\n\n"
    "Now provide your own response:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_last_assistant_content(messages: List[Dict]) -> Optional[str]:
    """Return the content of the last assistant message, or None."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, list):
                # Block-formatted content — join text blocks
                return " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return content
    return None


def build_teacher_messages(
    messages: List[Dict],
    demonstration: Optional[str],
    demo_template: str,
) -> List[Dict]:
    """
    Build the teacher's message list by prepending the demonstration as context.

    The demonstration context is injected as a system message (or prepended to
    an existing system message) so the chat template renders it correctly for
    all model families.

    If *demonstration* is None the messages are returned unchanged (teacher ==
    student, KL loss → 0, which gracefully degrades to standard SFT behaviour).
    """
    if demonstration is None:
        return messages

    demo_context = demo_template.format(demonstration=demonstration)

    if messages and messages[0]["role"] == "system":
        # Prepend to the existing system message
        new_system = demo_context + "\n\n" + messages[0]["content"]
        return [{"role": "system", "content": new_system}] + messages[1:]
    else:
        # Insert a new system message at the front
        return [{"role": "system", "content": demo_context}] + messages


# ---------------------------------------------------------------------------
# Custom data collator
# ---------------------------------------------------------------------------


@dataclass
class SDFTDataCollator:
    """
    Wraps a base collator (e.g. DataCollatorForSeq2Seq) and additionally pads
    the pre-tokenized teacher inputs that live as extra dataset columns.

    Expected extra columns in each feature dict:
      - teacher_input_ids       (List[int])
      - teacher_attention_mask  (List[int])

    Both are padded to the longest teacher sequence in the batch.
    """

    base_collator: Any
    pad_token_id: int
    max_seq_length: int = 2048

    # Columns that the base collator (DataCollatorForSeq2Seq) knows how to
    # tensorize.  Any other column will be silently dropped before passing to
    # the base collator to avoid "Unable to create tensor" errors from
    # non-integer fields left in the dataset (e.g. "messages", "text").
    _BASE_COLLATOR_COLUMNS = frozenset(
        {"input_ids", "attention_mask", "labels", "token_type_ids",
         "special_tokens_mask", "decoder_input_ids"}
    )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pop teacher-specific fields so the base collator doesn't choke on them
        teacher_input_ids_list = [
            f.pop("teacher_input_ids") for f in features
        ]
        teacher_attention_mask_list = [
            f.pop("teacher_attention_mask") for f in features
        ]

        # Drop any column the base collator can't tensorize (e.g. "messages")
        clean_features = [
            {k: v for k, v in f.items() if k in self._BASE_COLLATOR_COLUMNS}
            for f in features
        ]

        # Standard student collation (handles labels, padding, etc.)
        batch = self.base_collator(clean_features)

        # Pad teacher sequences to uniform length (right-padding)
        max_len = max(
            min(len(s), self.max_seq_length) for s in teacher_input_ids_list
        )
        padded_ids = []
        padded_masks = []
        for ids, masks in zip(teacher_input_ids_list, teacher_attention_mask_list):
            ids = ids[: self.max_seq_length]
            masks = masks[: self.max_seq_length]
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_masks.append(masks + [0] * pad_len)

        batch["teacher_input_ids"] = torch.tensor(padded_ids, dtype=torch.long)
        batch["teacher_attention_mask"] = torch.tensor(
            padded_masks, dtype=torch.long
        )

        return batch


# ---------------------------------------------------------------------------
# EMA teacher callback
# ---------------------------------------------------------------------------


class EMATeacherCallback(TrainerCallback):
    """
    Updates the EMA teacher weights *after* each optimizer step.

    Fires on `on_step_end`, which in the HuggingFace Trainer loop runs after
    the optimizer and scheduler have already updated the student.

    EMA rule:  φ ← α·θ + (1−α)·φ
    """

    def __init__(self, sdft_trainer: "SDFTTrainer") -> None:
        self._trainer = sdft_trainer

    def on_step_end(self, args, state, control, **kwargs):
        self._trainer._update_teacher_ema()


# ---------------------------------------------------------------------------
# SDFTTrainer
# ---------------------------------------------------------------------------


class SDFTTrainer(SFTTrainer):
    """
    SFT Trainer extended for Self-Distillation Fine-Tuning (SDFT).

    The EMA teacher's logit distribution (conditioned on a demonstration)
    serves as the supervision target.  The student minimises the reverse KL
    divergence to that target over response tokens.

    Extra constructor arguments
    ---------------------------
    ema_alpha : float
        EMA rate for updating the teacher.  Higher = teacher tracks student
        faster.  Paper recommends values in {0.01, 0.02, 0.05}.  Default 0.02.
    """

    def __init__(
        self,
        *args,
        ema_alpha: float = 0.02,
        tokenizer=None,
        processing_class=None,
        **kwargs,
    ) -> None:
        # Forward the tokenizer under whichever parameter name this TRL version
        # expects.  We explicitly capture both names so neither leaks into
        # **kwargs and triggers an "unexpected keyword argument" error.
        effective_tokenizer = processing_class or tokenizer
        if effective_tokenizer is not None:
            kwargs[_SFT_TOKENIZER_KWARG] = effective_tokenizer
        super().__init__(*args, **kwargs)
        self.ema_alpha = ema_alpha

        # ------------------------------------------------------------------ #
        # Initialise EMA teacher state as a CPU copy of trainable parameters.
        # For LoRA models, only the adapter weights are trainable — the frozen
        # base weights are shared and never need to be swapped.
        # ------------------------------------------------------------------ #
        self._teacher_state: Dict[str, torch.Tensor] = {
            name: param.data.clone().detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Register the EMA update callback
        self.add_callback(EMATeacherCallback(self))

    # ---------------------------------------------------------------------- #
    # EMA teacher utilities
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass with EMA teacher weights and return logits.

        Implementation: temporarily replace trainable (LoRA) weights with their
        EMA values, run a no-grad forward pass, then restore the student weights.
        This avoids allocating a separate teacher model copy and is safe because
        (a) the forward pass is under no_grad so no computation graph is built,
        (b) student weights are always restored before the backward pass.
        """
        device = input_ids.device

        # 1. Save current student weights and install teacher weights
        original: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if name in self._teacher_state:
                original[name] = param.data
                param.data = self._teacher_state[name].to(device)

        # 2. Teacher forward pass
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = outputs.logits.detach()
        finally:
            # 3. Restore student weights (runs even if forward raises)
            for name, param in self.model.named_parameters():
                if name in original:
                    param.data = original[name]

        return teacher_logits

    @torch.no_grad()
    def _update_teacher_ema(self) -> None:
        """EMA update: φ ← α·θ + (1−α)·φ"""
        for name, param in self.model.named_parameters():
            if name in self._teacher_state:
                teacher_val = self._teacher_state[name]
                # Move to same device for arithmetic, then store back on CPU
                device = param.device
                teacher_val = teacher_val.to(device)
                teacher_val.mul_(1.0 - self.ema_alpha).add_(
                    param.data, alpha=self.ema_alpha
                )
                self._teacher_state[name] = teacher_val.cpu()

    # ---------------------------------------------------------------------- #
    # Loss computation
    # ---------------------------------------------------------------------- #

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
        **kwargs,
    ):
        """
        Compute the SDFT reverse-KL loss.

        If teacher inputs are absent (e.g. during evaluation) we fall back to
        the standard cross-entropy SFT loss so that eval metrics remain valid.

        Loss formula (token-level, summed over vocabulary):

            L = (1/K) Σ_{t ∈ response} Σ_v  P_s(v|y_<t, x)
                      × [ log P_s(v|y_<t, x) − log P_t(v|y_<t, x, c) ]

        where K is the number of response tokens in the batch.
        """
        teacher_input_ids = inputs.pop("teacher_input_ids", None)
        teacher_attention_mask = inputs.pop("teacher_attention_mask", None)

        # ------------------------------------------------------------------ #
        # Fallback: no teacher inputs → standard SFT cross-entropy loss
        # ------------------------------------------------------------------ #
        if teacher_input_ids is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        labels = inputs.get("labels")  # [B, T_s];  -100 for masked positions

        # ------------------------------------------------------------------ #
        # Student forward pass  (builds computation graph)
        # ------------------------------------------------------------------ #
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        student_logits = student_outputs.logits  # [B, T_s, V]

        # ------------------------------------------------------------------ #
        # Teacher forward pass  (EMA weights, no grad)
        # ------------------------------------------------------------------ #
        teacher_logits_raw = self._get_teacher_logits(
            teacher_input_ids, teacher_attention_mask
        )  # [B, T_t, V]

        # ------------------------------------------------------------------ #
        # Shift logits for next-token prediction
        # ------------------------------------------------------------------ #
        # student_logits[:, :-1, :] predicts student_input_ids[:, 1:]
        shift_student_logits = student_logits[:, :-1, :]   # [B, T_s-1, V]
        shift_labels = labels[:, 1:]                        # [B, T_s-1]
        response_mask = (shift_labels != -100).float()      # 1 at response tokens

        # ------------------------------------------------------------------ #
        # Align teacher logits to student sequence length
        #
        # Teacher input: [<demo_prefix_D_tokens>] [<prompt+response_N_tokens>]
        # Student input:                          [<prompt+response_N_tokens>]
        #
        # Both sequences end with the same suffix (prompt + response), so the
        # last T_s-1 positions of the shifted teacher logits correspond to the
        # same token predictions as the shifted student logits.
        # ------------------------------------------------------------------ #
        T_s = shift_student_logits.shape[1]  # N - 1
        T_t = teacher_logits_raw.shape[1] - 1  # M - 1  (M = N + D)

        if T_t >= T_s:
            # Normal case: teacher sequence is at least as long as student
            shift_teacher_logits = teacher_logits_raw[:, -T_s - 1:-1, :]
        else:
            # Edge case: teacher was truncated to be shorter than student
            # (can happen if demo is so long it displaces the prompt).
            # Pad the left with zeros so shapes match — those positions will
            # be outside the response_mask anyway.
            pad_size = T_s - T_t
            pad = torch.zeros(
                teacher_logits_raw.shape[0],
                pad_size,
                teacher_logits_raw.shape[-1],
                device=teacher_logits_raw.device,
                dtype=teacher_logits_raw.dtype,
            )
            shift_teacher_logits = torch.cat(
                [pad, teacher_logits_raw[:, :-1, :]], dim=1
            )

        # ------------------------------------------------------------------ #
        # Token-level reverse KL divergence
        # KL(P_s || P_t) = Σ_v P_s(v) × [log P_s(v) − log P_t(v)]
        # ------------------------------------------------------------------ #
        student_log_probs = F.log_softmax(shift_student_logits, dim=-1)   # [B, T, V]
        teacher_log_probs = F.log_softmax(shift_teacher_logits, dim=-1)   # [B, T, V]
        student_probs = student_log_probs.exp()                            # [B, T, V]

        # Sum over vocabulary → per-token KL  [B, T]
        per_token_kl = (
            student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)

        # Mask to response tokens only and average
        masked_kl = (per_token_kl * response_mask).sum()
        n_tokens = response_mask.sum().clamp(min=1.0)
        loss = masked_kl / n_tokens

        if return_outputs:
            return loss, student_outputs
        return loss


# ---------------------------------------------------------------------------
# Entry point  (mirrors sft_train / dpo_train / orpo_train)
# ---------------------------------------------------------------------------


def sdft_train(
    training_cfg,
    dataset,
    model,
    tokenizer,
    test_dataset=None,
    logp_datasets: Dict = {},
    **kwargs,
):
    """
    Build and return an SDFTTrainer ready to call .train() on.

    Dataset pre-processing
    ----------------------
    1.  Apply the chat template to produce ``text`` (student input).
    2.  Build teacher messages (demo prepended) and apply the chat template
        to produce ``teacher_text``.
    3.  Tokenize ``teacher_text`` and store the result as extra dataset columns
        ``teacher_input_ids`` / ``teacher_attention_mask``.
    4.  Drop the raw text columns; SFTTrainer handles ``text`` tokenisation.
    5.  Wrap the base DataCollator with ``SDFTDataCollator`` so both student
        and teacher tensors are correctly padded inside each batch.
    """
    demo_template: str = getattr(
        training_cfg, "sdft_demo_template", None
    ) or DEFAULT_DEMO_TEMPLATE
    ema_alpha: float = getattr(training_cfg, "sdft_ema_alpha", 0.02)

    # ------------------------------------------------------------------ #
    # 1 & 2.  Build student text and teacher text
    # ------------------------------------------------------------------ #
    def apply_templates(examples):
        texts: List[str] = []
        teacher_texts: List[str] = []

        messages_list = examples["messages"]
        demo_list = examples.get("demonstration", [None] * len(messages_list))

        for messages, demo in zip(messages_list, demo_list):
            # Student: standard chat-template rendering
            student_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not student_text.strip().endswith(tokenizer.eos_token):
                student_text += tokenizer.eos_token
            texts.append(student_text)

            # If no demonstration provided, use the last assistant message
            if demo is None:
                demo = _get_last_assistant_content(messages)

            teacher_messages = build_teacher_messages(
                messages, demo, demo_template
            )
            teacher_text = tokenizer.apply_chat_template(
                teacher_messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not teacher_text.strip().endswith(tokenizer.eos_token):
                teacher_text += tokenizer.eos_token
            teacher_texts.append(teacher_text)

        return {"text": texts, "teacher_text": teacher_texts}

    dataset = dataset.map(apply_templates, batched=True)
    print_dataset_examples(dataset, "Training (SDFT)", num_examples=2)

    if test_dataset is not None:
        test_dataset = test_dataset.map(apply_templates, batched=True)

    # ------------------------------------------------------------------ #
    # 3.  Pre-tokenise teacher texts → extra dataset columns
    # ------------------------------------------------------------------ #
    def tokenize_teacher(examples):
        enc = tokenizer(
            examples["teacher_text"],
            max_length=training_cfg.max_seq_length,
            truncation=True,
            padding=False,
        )
        return {
            "teacher_input_ids": enc["input_ids"],
            "teacher_attention_mask": enc["attention_mask"],
        }

    dataset = dataset.map(tokenize_teacher, batched=True)
    # Remove all columns except those needed for training.
    # "text" is consumed by SFTTrainer's internal tokeniser;
    # teacher_input_ids / teacher_attention_mask are consumed by SDFTDataCollator.
    # Any other column (e.g. "messages", "demonstration") would cause
    # "Unable to create tensor" errors in the data collator.
    _keep = {"text", "teacher_input_ids", "teacher_attention_mask"}
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in _keep]
    )
    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize_teacher, batched=True)
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in _keep]
        )

    # ------------------------------------------------------------------ #
    # 4.  Learning rate normalisation (mirrors sft.py)
    # ------------------------------------------------------------------ #
    learning_rate = training_cfg.learning_rate
    if isinstance(learning_rate, str):
        learning_rate = eval(learning_rate)
    if isinstance(learning_rate, float) and learning_rate < 0:
        learning_rate = 10 ** learning_rate

    # ------------------------------------------------------------------ #
    # 5.  Optional callbacks
    # ------------------------------------------------------------------ #
    logp_callbacks = []
    if training_cfg.logp_callback_datasets:
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

    sampling_callbacks_list = []
    if training_cfg.sampling_callbacks:
        sampling_callbacks_list = [
            SamplingCallback(
                sc.dataset,
                tokenizer,
                sc.eval_steps,
                sc.batch_size,
                sc.tag,
                sc.temperature,
                sc.max_tokens,
            )
            for sc in training_cfg.sampling_callbacks
        ]

    # ------------------------------------------------------------------ #
    # 6.  Build the SDFTTrainer
    # ------------------------------------------------------------------ #
    # Standard TrainingArguments params (same regardless of TRL version)
    _base_args = dict(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.eval_batch_size,
        eval_steps=training_cfg.test_file_eval_steps,
        eval_strategy=training_cfg.test_file_eval_strategy,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=training_cfg.logging_steps,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=[],
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        ddp_find_unused_parameters=False,
        # Keep extra dataset columns (teacher_input_ids etc.) in batches
        remove_unused_columns=False,
        **kwargs,
    )

    if _USE_SFT_CONFIG:
        # TRL >= ~0.14: dataset-specific params belong in SFTConfig, not in
        # SFTTrainer.__init__.  The class-level backward-compat decorator does
        # NOT fire when __init__ is reached via super(), so we must not pass
        # these kwargs through trainer_kwargs.
        training_args = _SFTConfig(
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=training_cfg.packing,
            **_base_args,
        )
        trainer_kwargs = dict(
            model=model,
            train_dataset=dataset,
            ema_alpha=ema_alpha,
            args=training_args,
            callbacks=[LogMetrics(), GPUStatsCallback()]
            + logp_callbacks
            + sampling_callbacks_list,
            eval_dataset=test_dataset,
        )
        # Forward tokenizer under the kwarg name this TRL version expects.
        # SDFTTrainer.__init__ captures both names explicitly, so the shim
        # will forward it correctly to super().__init__().
        trainer_kwargs[_SFT_TOKENIZER_KWARG] = tokenizer
    else:
        # Old TRL: dataset params are accepted directly by SFTTrainer.__init__
        trainer_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=training_cfg.packing,
            ema_alpha=ema_alpha,
            args=TrainingArguments(**_base_args),
            callbacks=[LogMetrics(), GPUStatsCallback()]
            + logp_callbacks
            + sampling_callbacks_list,
            eval_dataset=test_dataset,
        )

    # ------------------------------------------------------------------ #
    # 7.  Wrap with train_on_responses_only (optional, same as sft.py)
    # ------------------------------------------------------------------ #
    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"\nSDFT: train_on_responses_only  instruction={instruction_part!r}")
        base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        # Instantiate trainer first, then patch its data_collator
        trainer = SDFTTrainer(**trainer_kwargs)
        # Wrap via unsloth helper to get correct label masking
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
        # Replace the collator with our SDFT-aware wrapper
        trainer.data_collator = SDFTDataCollator(
            base_collator=base_collator,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_length=training_cfg.max_seq_length,
        )
    else:
        base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer_kwargs["data_collator"] = SDFTDataCollator(
            base_collator=base_collator,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_length=training_cfg.max_seq_length,
        )
        trainer = SDFTTrainer(**trainer_kwargs)

    return trainer
