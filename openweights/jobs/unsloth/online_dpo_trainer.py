from typing import Any, Optional, Union
import time

import jinja2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers.training_args import OptimizerNames
import logging

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
)
from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    empty_cache,
    get_reward,
)
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
)
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer

if is_apex_available():
    from apex import amp


class OnlineDPOTrainerCustom(OnlineDPOTrainer):

    def _forward_memory_efficient(
        self, model, prompt_ids, prompt_mask, completion_ids, completion_mask
    ):
        """
        Memory-efficient forward pass that computes log probabilities without creating large tensors.
        """
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(
            prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0
        )

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)

        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, prompt_ids.size(1) - 1 : -1]

        # Memory-efficient log probability calculation
        # Instead of computing full log_softmax, compute only for the tokens we need
        logprobs = torch.zeros_like(completion_ids, dtype=torch.float32)

        # Dynamic chunk size based on available memory
        total_tokens = logits.size(1)
        vocab_size = logits.size(-1)
        available_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else 24
        )
        # Conservative estimate: use only 10% of available memory for this operation
        max_memory_per_chunk_gb = available_memory_gb * 0.1
        max_tokens_per_chunk = int(
            (max_memory_per_chunk_gb * 1024**3) / (4 * vocab_size)
        )  # 4 bytes per float32
        chunk_size = min(
            1000, max(100, max_tokens_per_chunk)
        )  # Between 100 and 1000 tokens

        logging.info(f"Processing {total_tokens} tokens in chunks of {chunk_size}")

        for i in range(0, logits.size(1), chunk_size):
            end_idx = min(i + chunk_size, logits.size(1))
            chunk_logits = logits[:, i:end_idx, :]
            chunk_completion_ids = completion_ids[:, i:end_idx]

            # Compute log_softmax only for the chunk
            chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)

            # Gather the log probabilities for the completion tokens
            chunk_logprobs = torch.gather(
                chunk_log_probs, dim=-1, index=chunk_completion_ids.unsqueeze(-1)
            ).squeeze(-1)

            logprobs[:, i:end_idx] = chunk_logprobs

            # Clear memory
            del chunk_logits, chunk_log_probs, chunk_logprobs

            # Force garbage collection periodically
            if i % (chunk_size * 10) == 0:
                torch.cuda.empty_cache()

        return logprobs

    def _get_memory_stats(self):
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
            }
        return {}

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        start_time = time.time()
        logging.warning("Starting training step")

        # Log memory usage at start
        memory_stats = self._get_memory_stats()
        if memory_stats:
            logging.warning(
                f"Memory at start: {memory_stats['allocated_gb']:.1f}GB allocated, {memory_stats['free_gb']:.1f}GB free"
            )

        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)

        generation_start = time.time()
        logging.warning("Generating samples")
        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = (
                self._generate_vllm(model, prompts)
            )
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(
                model, prompts
            )
        generation_time = time.time() - generation_start
        logging.warning(f"Generation completed in {generation_time:.4f} seconds")

        contain_eos_token = torch.any(
            completion_ids == self.processing_class.eos_token_id, dim=-1
        )

        forward_start = time.time()
        # Use memory-efficient forward pass
        logprobs = self._forward_memory_efficient(
            model, prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        forward_time = time.time() - forward_start
        logging.warning(f"Forward pass completed in {forward_time:.4f} seconds")

        ref_start = time.time()
        logging.warning("Generating ref logprobs")
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward_memory_efficient(
                    self.ref_model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                )
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward_memory_efficient(
                        self.model,
                        prompt_ids,
                        prompt_mask,
                        completion_ids,
                        completion_mask,
                    )
        ref_time = time.time() - ref_start
        logging.warning(f"Reference logprobs completed in {ref_time:.4f} seconds")

        # Log memory usage after forward passes
        memory_stats = self._get_memory_stats()
        if memory_stats:
            logging.warning(
                f"Memory after forward passes: {memory_stats['allocated_gb']:.1f}GB allocated, {memory_stats['free_gb']:.1f}GB free"
            )

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational({"prompt": prompts[0]}):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Get the reward from the reward model or judge
        reward_start = time.time()
        if self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [
                    template.render(messages=completion) for completion in completions
                ]

            logging.warning("Judging")
            ranks_of_first_completion = self.judge.judge(
                prompts, list(zip(completions[:batch_size], completions[batch_size:]))
            )

            # convert ranks to a True/False mask:
            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            mask = torch.tensor(
                [rank == 0 for rank in ranks_of_first_completion], device=device
            )
        else:
            # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
            # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
            prompts = (
                2 * prompts
            )  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            if is_conversational({"prompt": prompts[0]}):
                examples = [
                    {"prompt": p, "completion": c} for p, c in zip(prompts, completions)
                ]
                examples = [
                    apply_chat_template(example, self.reward_processing_class)
                    for example in examples
                ]
                prompts = [example["prompt"] for example in examples]
                completions = [example["completion"] for example in examples]

            # Tokenize the prompts
            prompts_ids = self.reward_processing_class(
                prompts, padding=True, return_tensors="pt", padding_side="left"
            )["input_ids"].to(device)
            context_length = prompts_ids.shape[1]

            # Tokenize the completions
            completions_ids = self.reward_processing_class(
                completions, padding=True, return_tensors="pt", padding_side="right"
            )["input_ids"].to(device)

            # Concatenate the prompts and completions and get the reward
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.reward_model,
                    prompt_completion_ids,
                    self.reward_processing_class.pad_token_id,
                    context_length,
                )

                # Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half
        reward_time = time.time() - reward_start
        logging.warning(f"Reward computation completed in {reward_time:.4f} seconds")

        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        loss_start = time.time()
        logging.warning("Computing loss")
        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat(
            (chosen_indices, rejected_indices), dim=0
        )  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(
            cr_logprobs_sum, batch_size
        )
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(
            cr_ref_logprobs_sum, batch_size
        )
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()
        loss_time = time.time() - loss_start
        logging.warning(f"Loss computation completed in {loss_time:.4f} seconds")

        # Log everything
        logging_start = time.time()
        logging.warning("Logging")
        if self.reward_model is not None:
            scores_margin = scores[chosen_indices] - scores[rejected_indices]
            self.stats["objective/scores_margin"].append(
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(
                self.accelerator.gather_for_metrics(scores.mean()).mean().item()
            )
        self.stats["val/contain_eos_token"].append(
            contain_eos_token.float().mean().item()
        )
        self.stats["logps/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item()
        )
        self.stats["logps/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item()
        )

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(
                self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
            )
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(
            self.accelerator.gather_for_metrics(mean_entropy).mean().item()
        )
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (
            rejected_logprobs_sum - rejected_ref_logprobs_sum
        )
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(
            rejected_rewards
        )
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)
        logging_time = time.time() - logging_start
        logging.warning(f"Logging completed in {logging_time:.4f} seconds")

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        backward_start = time.time()
        logging.warning("Backward pass")
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
        backward_time = time.time() - backward_start
        logging.warning(f"Backward pass completed in {backward_time:.4f} seconds")

        total_time = time.time() - start_time
        logging.warning(f"Total training step completed in {total_time:.4f} seconds")

        return loss.detach() / self.args.gradient_accumulation_steps
