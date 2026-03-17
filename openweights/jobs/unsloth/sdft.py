"""Self-Distillation Fine-Tuning (SDFT) trainer.

Reference paper: https://arxiv.org/abs/2601.19897
"Self-Distillation Enables Continual Learning"

Algorithm overview  (Algorithm 1 from the paper)
------------------
For each training step:
  1. Sample a minibatch of (query, demonstration) pairs.
  2. For each example, generate an ON-POLICY response y from the STUDENT
     (conditioned on the query only, no demonstration).
  3. Compute the analytic per-token KL divergence at each generated position t:

       L(θ) = (1/K) Σ_t  Σ_v  π_θ(v|y_<t, x)
                          × [ log π_θ(v|y_<t, x) − log π_φ(v|y_<t, x, c) ]

     where π_θ = student, π_φ = EMA teacher (conditioned on demo c), K = # tokens.

  4. Backprop through the STUDENT forward pass only (teacher uses no-grad EMA weights).
  5. After the optimizer step: φ ← α·θ + (1−α)·φ   (default α = 0.02).

On-policy property
------------------
The key property that makes SDFT less disruptive than SFT: the KL is evaluated
at states the student *actually visits* (its own generated tokens). When the
model already places high probability on the right tokens, the KL at those
positions is naturally small and the update is gentle.  Compare this to the
off-policy SFT loss, which forces updates at every gold-token position regardless
of whether the model would naturally visit those states.

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

When "demonstration" is absent the last assistant turn is used as the demo.

Key design decisions
--------------------
* Only *trainable* parameters (LoRA adapter weights) are EMA-tracked.
* Weight-swapping (student ↔ EMA teacher) is done inside `compute_loss` under
  `torch.no_grad()` for the teacher forward pass; student weights are restored
  before the backward pass so the autograd graph is intact.
* EMA update fires via `EMATeacherCallback.on_step_end` (after the optimizer step).
* The dataset stores pre-tokenized prompt-only and teacher-prefix columns so that
  `compute_loss` can seed generation without repeating the template application.
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
# TRL API compatibility shim — tokenizer vs processing_class
# ---------------------------------------------------------------------------
# Newer TRL versions renamed the `tokenizer` constructor parameter to
# `processing_class` and apply the backward-compat mapping via a *class-level*
# decorator on SFTTrainer.  That decorator fires for direct instantiation but
# NOT when __init__ is reached via super() from a subclass.  Detect which kwarg
# name SFTTrainer's __init__ actually accepts so SDFTTrainer can forward it.
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
                return " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return content
    return None


def _strip_last_assistant(messages: List[Dict]) -> List[Dict]:
    """Return messages with the last assistant turn removed (prompt-only view)."""
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return list(messages)


def build_teacher_messages(
    messages: List[Dict],
    demonstration: Optional[str],
    demo_template: str,
) -> List[Dict]:
    """
    Build the teacher's message list by prepending the demonstration as context.

    The demo is injected as a system message (or prepended to an existing system
    message) so the chat template renders it correctly for all model families.

    If *demonstration* is None the messages are returned unchanged (teacher ==
    student, KL loss → 0, graceful degradation to SFT behaviour).
    """
    if demonstration is None:
        return messages

    demo_context = demo_template.format(demonstration=demonstration)

    if messages and messages[0]["role"] == "system":
        new_system = demo_context + "\n\n" + messages[0]["content"]
        return [{"role": "system", "content": new_system}] + messages[1:]
    else:
        return [{"role": "system", "content": demo_context}] + messages


# ---------------------------------------------------------------------------
# Custom data collator
# ---------------------------------------------------------------------------


@dataclass
class SDFTDataCollator:
    """
    Wraps a base collator (DataCollatorForSeq2Seq) and additionally pads:
      - teacher_input_ids / teacher_attention_mask  (legacy, full gold sequence)
      - prompt_input_ids / prompt_attention_mask    (student prompt only; LEFT-padded
                                                     so model.generate() works directly)
      - teacher_prefix_input_ids / teacher_prefix_attention_mask  (demo+prompt prefix)

    All teacher / prefix sequences are RIGHT-padded.
    Prompt sequences are LEFT-padded to enable autoregressive generation from the
    end of the real tokens without a padding "gap" before the response.

    Expected extra columns in each feature dict:
      teacher_input_ids, teacher_attention_mask   (List[int])
      prompt_input_ids, prompt_attention_mask      (List[int])
      teacher_prefix_input_ids, teacher_prefix_attention_mask  (List[int])
    """

    base_collator: Any
    pad_token_id: int
    max_seq_length: int = 2048

    # Columns the base DataCollatorForSeq2Seq knows how to tensorise.
    # Any other column is silently dropped before passing to it.
    _BASE_COLLATOR_COLUMNS = frozenset(
        {"input_ids", "attention_mask", "labels", "token_type_ids",
         "special_tokens_mask", "decoder_input_ids"}
    )

    def _pad_right(
        self,
        sequences: List[List[int]],
        pad_value: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Right-pad a list of variable-length sequences into a 2-D tensor."""
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        out = []
        for s in sequences:
            s = s[: self.max_seq_length]
            pad_len = max_len - len(s)
            out.append(s + [pad_value] * pad_len)
        return torch.tensor(out, dtype=torch.long)

    def _pad_left(
        self,
        sequences: List[List[int]],
        pad_value: int,
    ) -> torch.Tensor:
        """Left-pad a list of variable-length sequences into a 2-D tensor."""
        max_len = max(min(len(s), self.max_seq_length) for s in sequences)
        out = []
        for s in sequences:
            s = s[: self.max_seq_length]
            pad_len = max_len - len(s)
            out.append([pad_value] * pad_len + s)
        return torch.tensor(out, dtype=torch.long)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ------------------------------------------------------------------ #
        # Pop all SDFT-specific fields before passing to base collator
        # ------------------------------------------------------------------ #
        def _pop(key):
            return [f.pop(key) for f in features]

        # Legacy full-sequence teacher (kept for fallback / off-policy mode)
        teacher_input_ids_list    = _pop("teacher_input_ids")
        teacher_attention_mask_list = _pop("teacher_attention_mask")

        # On-policy: student prompt (LEFT-padded for generation)
        prompt_input_ids_list     = _pop("prompt_input_ids")
        prompt_attention_mask_list = _pop("prompt_attention_mask")

        # On-policy: teacher prefix = demo + student prompt (RIGHT-padded)
        tprefix_input_ids_list    = _pop("teacher_prefix_input_ids")
        tprefix_attention_mask_list = _pop("teacher_prefix_attention_mask")

        # Drop any non-tensorisable column from student features
        clean_features = [
            {k: v for k, v in f.items() if k in self._BASE_COLLATOR_COLUMNS}
            for f in features
        ]

        # Standard student collation
        batch = self.base_collator(clean_features)

        # ------------------------------------------------------------------ #
        # Legacy teacher: right-pad
        # ------------------------------------------------------------------ #
        max_t = max(
            min(len(s), self.max_seq_length) for s in teacher_input_ids_list
        )
        batch["teacher_input_ids"] = self._pad_right(
            teacher_input_ids_list, self.pad_token_id, max_t
        )
        batch["teacher_attention_mask"] = self._pad_right(
            [[1] * min(len(s), self.max_seq_length) for s in teacher_attention_mask_list],
            0, max_t,
        )

        # ------------------------------------------------------------------ #
        # On-policy prompt: LEFT-pad (so generate() works correctly)
        # ------------------------------------------------------------------ #
        batch["prompt_input_ids"] = self._pad_left(
            prompt_input_ids_list, self.pad_token_id
        )
        # Attention mask: 0 for left-padding, 1 for real tokens
        prompt_lens = [min(len(s), self.max_seq_length) for s in prompt_input_ids_list]
        max_p = batch["prompt_input_ids"].shape[1]
        prompt_mask = torch.zeros_like(batch["prompt_input_ids"])
        for i, pl in enumerate(prompt_lens):
            prompt_mask[i, max_p - pl:] = 1
        batch["prompt_attention_mask"] = prompt_mask

        # ------------------------------------------------------------------ #
        # On-policy teacher prefix: right-pad
        # ------------------------------------------------------------------ #
        max_tp = max(
            min(len(s), self.max_seq_length) for s in tprefix_input_ids_list
        )
        batch["teacher_prefix_input_ids"] = self._pad_right(
            tprefix_input_ids_list, self.pad_token_id, max_tp
        )
        batch["teacher_prefix_attention_mask"] = self._pad_right(
            [[1] * min(len(s), self.max_seq_length) for s in tprefix_attention_mask_list],
            0, max_tp,
        )

        return batch


# ---------------------------------------------------------------------------
# EMA teacher callback
# ---------------------------------------------------------------------------


class EMATeacherCallback(TrainerCallback):
    """
    Updates the EMA teacher weights *after* each optimizer step.

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

    Implements Algorithm 1 from the paper: on-policy KL distillation from a
    demonstration-conditioned EMA teacher.

    Extra constructor arguments
    ---------------------------
    ema_alpha : float
        EMA rate for updating the teacher.  Paper recommends {0.01, 0.02, 0.05}.
    max_new_tokens : int
        Max tokens to generate for the on-policy student rollout per step.
    """

    def __init__(
        self,
        *args,
        ema_alpha: float = 0.02,
        max_new_tokens: int = 256,
        tokenizer=None,
        processing_class=None,
        **kwargs,
    ) -> None:
        # Forward the tokenizer under whichever parameter name this TRL version
        # expects (tokenizer vs processing_class).
        effective_tokenizer = processing_class or tokenizer
        if effective_tokenizer is not None:
            kwargs[_SFT_TOKENIZER_KWARG] = effective_tokenizer
        super().__init__(*args, **kwargs)

        self.ema_alpha = ema_alpha
        self._sdft_max_new_tokens = max_new_tokens

        # Cache tokenizer reference (works across TRL versions)
        self._sdft_tokenizer = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )

        # Initialise EMA teacher state as a CPU copy of trainable (LoRA) params
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

        Temporarily replaces trainable (LoRA) weights with EMA values, runs a
        no-grad forward pass, then restores student weights before returning.
        """
        device = input_ids.device
        original: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if name in self._teacher_state:
                original[name] = param.data
                param.data = self._teacher_state[name].to(device)
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.logits.detach()
        finally:
            for name, param in self.model.named_parameters():
                if name in original:
                    param.data = original[name]

    @torch.no_grad()
    def _update_teacher_ema(self) -> None:
        """EMA update: φ ← α·θ + (1−α)·φ"""
        for name, param in self.model.named_parameters():
            if name in self._teacher_state:
                tv = self._teacher_state[name].to(param.device)
                tv.mul_(1.0 - self.ema_alpha).add_(param.data, alpha=self.ema_alpha)
                self._teacher_state[name] = tv.cpu()

    # ---------------------------------------------------------------------- #
    # On-policy rollout
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _on_policy_rollout(
        self,
        prompt_ids_left: torch.Tensor,       # [B, P]  left-padded student prompts
        prompt_mask_left: torch.Tensor,       # [B, P]  attention mask
        teacher_prefix_ids: torch.Tensor,     # [B, TP] right-padded teacher prefixes
        teacher_prefix_mask: torch.Tensor,    # [B, TP] attention mask
    ):
        """
        Generate on-policy student responses, then rebuild right-padded
        student and teacher sequences for the subsequent forward passes.

        Returns
        -------
        student_ids, student_mask    : [B, S]  right-padded student input
        teacher_ids, teacher_mask    : [B, T]  right-padded teacher input
        gen_response                 : [B, G]  generated tokens (right-padded)
        gen_mask                     : [B, G]  1 for real generated tokens
        prompt_lengths               : [B]     actual (non-padded) prompt lengths
        teacher_prefix_lengths       : [B]     actual teacher prefix lengths
        """
        tok = self._sdft_tokenizer
        pad_id = tok.pad_token_id if tok is not None else 0
        eos_id = tok.eos_token_id if tok is not None else None
        device = prompt_ids_left.device
        B, P = prompt_ids_left.shape

        # Actual (non-padded) lengths from attention masks
        prompt_lengths = prompt_mask_left.sum(dim=1)           # [B]
        tprefix_lengths = teacher_prefix_mask.sum(dim=1)       # [B]

        # ------------------------------------------------------------------ #
        # 1.  Generate responses from student (left-padded prompts → generate)
        #
        # IMPORTANT — unsloth compatibility:
        # Unsloth patches the model's forward with a fast KV-cache inference
        # kernel (fast_forward_inference_custom) that relies on device-tracking
        # state initialised during a "normal" inference session.  When called
        # from within a training loop, that state is None → ValueError.
        #
        # Fix: keep the model in TRAINING mode (no model.eval()) and pass
        # use_cache=False, which routes through the standard training forward
        # instead of the cached inference kernel.  Generation is slower (one
        # full forward per new token instead of incremental) but correct.
        # torch.no_grad() (from the decorator) still prevents gradient
        # computation.  For LoRA models, no dropout layers are active anyway.
        # ------------------------------------------------------------------ #
        gen_out = self.model.generate(
            input_ids=prompt_ids_left,
            attention_mask=prompt_mask_left,
            max_new_tokens=self._sdft_max_new_tokens,
            do_sample=False,              # greedy; stable, on-policy approximation
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=False,              # avoid unsloth fast-inference path
        )

        # gen_out: [B, P + T_gen] — strip the prompt prefix
        gen_response = gen_out[:, P:]      # [B, T_gen]
        gen_mask = (gen_response != pad_id).long()

        # ------------------------------------------------------------------ #
        # 2.  Build right-padded STUDENT sequences: [actual_prompt | gen_resp]
        # ------------------------------------------------------------------ #
        max_prompt_len = prompt_lengths.max().item()
        T_gen = gen_response.shape[1]
        S = max_prompt_len + T_gen

        student_ids  = torch.full((B, S), pad_id, device=device, dtype=torch.long)
        student_mask = torch.zeros(B, S, device=device, dtype=torch.long)

        for i in range(B):
            pl = prompt_lengths[i].item()
            gl = gen_mask[i].sum().item()
            # Extract actual prompt tokens from LEFT-padded tensor (real tokens at end)
            actual_prompt = prompt_ids_left[i, P - pl:]        # (pl,)
            student_ids[i, :pl] = actual_prompt
            student_ids[i, pl:pl + gl] = gen_response[i, :gl]
            student_mask[i, :pl + gl] = 1

        # ------------------------------------------------------------------ #
        # 3.  Build right-padded TEACHER sequences: [teacher_prefix | gen_resp]
        # ------------------------------------------------------------------ #
        max_tprefix_len = tprefix_lengths.max().item()
        T = max_tprefix_len + T_gen

        teacher_ids  = torch.full((B, T), pad_id, device=device, dtype=torch.long)
        teacher_mask = torch.zeros(B, T, device=device, dtype=torch.long)

        for i in range(B):
            tl = tprefix_lengths[i].item()
            gl = gen_mask[i].sum().item()
            actual_prefix = teacher_prefix_ids[i, :tl]         # (tl,) right-padded → real at start
            teacher_ids[i, :tl] = actual_prefix
            teacher_ids[i, tl:tl + gl] = gen_response[i, :gl]
            teacher_mask[i, :tl + gl] = 1

        return (
            student_ids, student_mask,
            teacher_ids, teacher_mask,
            gen_response, gen_mask,
            prompt_lengths, tprefix_lengths,
        )

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
        On-policy SDFT reverse-KL loss (Algorithm 1, analytic per-token estimator).

        For each example in the batch:
          1. Generate response y from the STUDENT (no demo, greedy).
          2. Compute analytic KL at each generated position t:
               Σ_v P_s(v|y<t,x) [log P_s(v|y<t,x) − log P_t(v|y<t,x,c)]
          3. Average over generated (non-padding) response tokens.

        Falls back to standard SFT cross-entropy if on-policy columns are absent
        (e.g. during evaluation or in off-policy datasets).
        """
        # ------------------------------------------------------------------ #
        # Pop on-policy tensors (added by SDFTDataCollator)
        # ------------------------------------------------------------------ #
        prompt_ids        = inputs.pop("prompt_input_ids", None)
        prompt_mask       = inputs.pop("prompt_attention_mask", None)
        tprefix_ids       = inputs.pop("teacher_prefix_input_ids", None)
        tprefix_mask      = inputs.pop("teacher_prefix_attention_mask", None)
        # Pop legacy full-sequence teacher (not used in on-policy mode)
        inputs.pop("teacher_input_ids", None)
        inputs.pop("teacher_attention_mask", None)

        # ------------------------------------------------------------------ #
        # Fallback: missing on-policy data → standard SFT cross-entropy loss
        # ------------------------------------------------------------------ #
        if prompt_ids is None or tprefix_ids is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        # ------------------------------------------------------------------ #
        # 1.  Generate on-policy responses and rebuild student/teacher inputs
        # ------------------------------------------------------------------ #
        (
            student_ids, student_mask,
            teacher_ids, teacher_mask,
            gen_response, gen_mask,
            prompt_lengths, tprefix_lengths,
        ) = self._on_policy_rollout(prompt_ids, prompt_mask, tprefix_ids, tprefix_mask)

        # ------------------------------------------------------------------ #
        # 2.  Student forward pass  (builds computation graph)
        # ------------------------------------------------------------------ #
        student_outputs = model(
            input_ids=student_ids,
            attention_mask=student_mask,
        )
        student_logits = student_outputs.logits   # [B, S, V]

        # ------------------------------------------------------------------ #
        # 3.  Teacher forward pass  (EMA weights, no grad)
        # ------------------------------------------------------------------ #
        teacher_logits = self._get_teacher_logits(teacher_ids, teacher_mask)  # [B, T, V]

        # ------------------------------------------------------------------ #
        # 4.  Compute per-example analytic KL over generated positions
        #
        #  student_logits[i, pl-1 : pl+gl-1, :] predicts student_ids[i, pl : pl+gl]
        #  teacher_logits[i, tl-1 : tl+gl-1, :] predicts teacher_ids[i, tl : tl+gl]
        #  Both predict the same generated tokens.
        # ------------------------------------------------------------------ #
        total_kl = torch.tensor(0.0, device=student_ids.device)
        total_n  = torch.tensor(0.0, device=student_ids.device)

        B = student_ids.shape[0]
        for i in range(B):
            pl = prompt_lengths[i].item()
            tl = tprefix_lengths[i].item()
            gl = int(gen_mask[i].sum().item())

            if gl == 0:
                continue  # nothing generated (shouldn't normally happen)

            # Logits that predict the generated tokens
            s_resp = student_logits[i, pl - 1 : pl + gl - 1, :]   # (gl, V)
            t_resp = teacher_logits[i, tl - 1 : tl + gl - 1, :]   # (gl, V)

            s_lp = F.log_softmax(s_resp, dim=-1)                   # (gl, V)
            t_lp = F.log_softmax(t_resp, dim=-1)                   # (gl, V)
            per_token_kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)   # (gl,)

            total_kl = total_kl + per_token_kl.sum()
            total_n  = total_n  + gl

        loss = total_kl / total_n.clamp(min=1.0)

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
    1.  Apply the chat template to produce:
          ``text``                - full student sequence (prompt + gold response)
                                    consumed by SFTTrainer's internal tokeniser
          ``teacher_text``        - full teacher sequence (demo + prompt + gold resp)
                                    kept for legacy / fallback use
          ``prompt_text``         - student prompt only (add_generation_prompt=True)
                                    used to seed on-policy generation
          ``teacher_prefix_text`` - teacher prefix: demo + student prompt
                                    (add_generation_prompt=True), used to build
                                    the teacher's conditioning context on top of
                                    the generated response
    2.  Pre-tokenise teacher sequences and prompt/prefix sequences as extra columns.
    3.  Strip all non-tensorisable columns; SDFTDataCollator handles padding.
    """
    demo_template: str = getattr(
        training_cfg, "sdft_demo_template", None
    ) or DEFAULT_DEMO_TEMPLATE
    ema_alpha: float  = getattr(training_cfg, "sdft_ema_alpha", 0.02)
    max_new_tokens: int = getattr(training_cfg, "sdft_max_new_tokens", 256)

    # ------------------------------------------------------------------ #
    # 1.  Build all text columns
    # ------------------------------------------------------------------ #
    def apply_templates(examples):
        texts                = []   # full student sequence
        teacher_texts        = []   # full teacher sequence (legacy)
        prompt_texts         = []   # student prompt only (for on-policy generation)
        teacher_prefix_texts = []   # teacher prefix = demo + student prompt

        messages_list = examples["messages"]
        demo_list = examples.get("demonstration", [None] * len(messages_list))

        for messages, demo in zip(messages_list, demo_list):
            # If no demonstration provided, use the last assistant message
            if demo is None:
                demo = _get_last_assistant_content(messages)

            # --- Full student sequence (prompt + gold response) ---
            student_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not student_text.strip().endswith(tokenizer.eos_token):
                student_text += tokenizer.eos_token
            texts.append(student_text)

            # --- Full teacher sequence (demo + prompt + gold response) ---
            teacher_messages = build_teacher_messages(messages, demo, demo_template)
            teacher_text = tokenizer.apply_chat_template(
                teacher_messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not teacher_text.strip().endswith(tokenizer.eos_token):
                teacher_text += tokenizer.eos_token
            teacher_texts.append(teacher_text)

            # --- Prompt-only view (strip last assistant turn) ---
            prompt_messages = _strip_last_assistant(messages)

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(prompt_text)

            # --- Teacher prefix (demo + prompt, no gold response) ---
            teacher_prefix_messages = build_teacher_messages(
                prompt_messages, demo, demo_template
            )
            teacher_prefix_text = tokenizer.apply_chat_template(
                teacher_prefix_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            teacher_prefix_texts.append(teacher_prefix_text)

        return {
            "text": texts,
            "teacher_text": teacher_texts,
            "prompt_text": prompt_texts,
            "teacher_prefix_text": teacher_prefix_texts,
        }

    dataset = dataset.map(apply_templates, batched=True)
    print_dataset_examples(dataset, "Training (SDFT)", num_examples=2)

    if test_dataset is not None:
        test_dataset = test_dataset.map(apply_templates, batched=True)

    # ------------------------------------------------------------------ #
    # 2.  Pre-tokenise teacher + prompt/prefix sequences
    # ------------------------------------------------------------------ #
    def tokenize_extra(examples):
        # Full teacher sequences (legacy / fallback)
        t_enc = tokenizer(
            examples["teacher_text"],
            max_length=training_cfg.max_seq_length,
            truncation=True,
            padding=False,
        )
        # Student prompt-only (for on-policy generation seeding)
        p_enc = tokenizer(
            examples["prompt_text"],
            max_length=training_cfg.max_seq_length,
            truncation=True,
            padding=False,
        )
        # Teacher prefix (demo + prompt, no response)
        tp_enc = tokenizer(
            examples["teacher_prefix_text"],
            max_length=training_cfg.max_seq_length,
            truncation=True,
            padding=False,
        )
        return {
            "teacher_input_ids":              t_enc["input_ids"],
            "teacher_attention_mask":         t_enc["attention_mask"],
            "prompt_input_ids":               p_enc["input_ids"],
            "prompt_attention_mask":          p_enc["attention_mask"],
            "teacher_prefix_input_ids":       tp_enc["input_ids"],
            "teacher_prefix_attention_mask":  tp_enc["attention_mask"],
        }

    dataset = dataset.map(tokenize_extra, batched=True)

    # Keep only columns that are either consumed by SFTTrainer ("text") or
    # by SDFTDataCollator (the six pre-tokenised columns).
    _keep = {
        "text",
        "teacher_input_ids",      "teacher_attention_mask",
        "prompt_input_ids",       "prompt_attention_mask",
        "teacher_prefix_input_ids", "teacher_prefix_attention_mask",
    }
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in _keep]
    )

    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize_extra, batched=True)
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in _keep]
        )

    # ------------------------------------------------------------------ #
    # 3.  Learning rate normalisation
    # ------------------------------------------------------------------ #
    learning_rate = training_cfg.learning_rate
    if isinstance(learning_rate, str):
        learning_rate = eval(learning_rate)
    if isinstance(learning_rate, float) and learning_rate < 0:
        learning_rate = 10 ** learning_rate

    # ------------------------------------------------------------------ #
    # 4.  Optional callbacks
    # ------------------------------------------------------------------ #
    logp_callbacks: list = []
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

    sampling_callbacks_list: list = []
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
    # 5.  Build training args (SFTConfig if available, else TrainingArguments)
    # ------------------------------------------------------------------ #
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
        # TRL >= ~0.14: dataset-specific params belong in SFTConfig
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
            max_new_tokens=max_new_tokens,
            args=training_args,
            callbacks=[LogMetrics(), GPUStatsCallback()]
            + logp_callbacks
            + sampling_callbacks_list,
            eval_dataset=test_dataset,
        )
        trainer_kwargs[_SFT_TOKENIZER_KWARG] = tokenizer
    else:
        # Old TRL: dataset params passed directly to trainer
        trainer_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=training_cfg.max_seq_length,
            dataset_num_proc=4,
            packing=training_cfg.packing,
            ema_alpha=ema_alpha,
            max_new_tokens=max_new_tokens,
            args=TrainingArguments(**_base_args),
            callbacks=[LogMetrics(), GPUStatsCallback()]
            + logp_callbacks
            + sampling_callbacks_list,
            eval_dataset=test_dataset,
        )

    # ------------------------------------------------------------------ #
    # 6.  Wrap with train_on_responses_only (optional, same as sft.py)
    # ------------------------------------------------------------------ #
    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"\nSDFT: train_on_responses_only  instruction={instruction_part!r}")
        base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer = SDFTTrainer(**trainer_kwargs)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
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
