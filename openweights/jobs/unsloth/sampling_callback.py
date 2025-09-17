import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from utils import client, load_jsonl


def _sample(
    model,
    tokenizer,
    conversations,
    top_p=1,
    max_tokens=600,
    temperature=0,
    stop=[],
    prefix="",
):
    is_training = model.training
    if is_training:
        FastLanguageModel.for_inference(model)
    texts = []
    for conversation in conversations:
        messages = conversation["messages"]
        pre = prefix
        if messages[-1]["role"] == "assistant":
            messages, pre = messages[:-1], messages[-1]["content"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text + pre)
    # Tokenize and pad the input texts
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "temperature": temperature if temperature > 0 else 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask": attention_mask,
        "stop_strings": [tokenizer.eos_token],
        "tokenizer": tokenizer,
    }
    with torch.no_grad():
        output_sequences = model.generate(input_ids=input_ids, **gen_kwargs)
    decoded_outputs = tokenizer.batch_decode(
        output_sequences[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    if is_training:
        FastLanguageModel.for_training(model)
    return [prefix + output for output in decoded_outputs]


def sample(
    model,
    tokenizer,
    conversations,
    batch_size,
    top_p=1,
    max_tokens=600,
    temperature=0,
    stop=[],
    prefix="",
):
    """Batched version of _sample"""
    completions = []
    for i in range(0, len(conversations), batch_size):
        completions.extend(
            _sample(
                model,
                tokenizer,
                conversations[i : i + batch_size],
                top_p,
                max_tokens,
                temperature,
                stop,
                prefix,
            )
        )
    return completions


class SamplingCallback(TrainerCallback):
    def __init__(
        self,
        dataset,
        tokenizer,
        eval_steps="log",
        batch_size=8,
        tag="samples",
        temperature=0,
        max_tokens=600,
    ):
        """
        A callback that samples from the model and logs the results.

        Args:
            dataset: List[Message] or str: file_id
            tokenizer: The tokenizer to use for encoding conversations
            eval_steps: Evaluate every `eval_steps` training steps
            output_dir: Directory where token-level logP data will be saved
            batch_size: Batch size to use during evaluation
            tag: Key to use when logging the loss metric
        """
        if isinstance(dataset, str):
            dataset = load_jsonl(dataset)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.tag = tag
        self.temperature = temperature
        self.max_tokens = max_tokens

    def on_init_end(self, args, state, control, **kwargs):
        self.model = kwargs["model"]

    def on_train_begin(self, args, state, control, **kwargs):
        self.run(model=self.model, step=0)

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.eval_steps != 0:
            return
        self.run(kwargs["model"], state.global_step)

    def run(self, model, step):
        """Called at the end of each training step."""
        # Get the model from kwargs
        FastLanguageModel.for_inference(model)

        completions = sample(
            model,
            self.tokenizer,
            self.dataset,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        results_file = f"samples_{self.tag}_{step}.jsonl"
        with open(results_file, "w") as f:
            for row, completion in zip(self.dataset, completions):
                row["completion"] = completion
                f.write(json.dumps(row) + "\n")

        with open(results_file, "rb") as f:
            samples_file = client.files.create(f, purpose="samples")

        # Log the test loss
        client.run.log(
            {
                "type": "samples",
                "step": step,
                "file": samples_file["id"],
                "tag": self.tag,
            }
        )

        # Return model to training mode
        FastLanguageModel.for_training(model)
