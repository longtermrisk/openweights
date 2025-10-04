import json
import os

from logprobs import get_logprobs, get_logprobs_blockwise
from transformers import TrainerCallback
from utils import client


class LogTestLossCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, eval_steps, log_as, batch_size):
        """
        A callback that evaluates model performance on a test dataset and logs the results.

        Args:
            test_dataset: Dataset with 'messages' field containing conversation messages
            tokenizer: The tokenizer to use for encoding conversations
            eval_steps: Evaluate every `eval_steps` training steps
            batch_size: Batch size to use during evaluation
            log_as: Key to use when logging the loss metric
        """
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.log_as = log_as
        self.is_block_format = False
        if "messages" in self.test_dataset.column_names and len(self.test_dataset) > 0:
            first_example = self.test_dataset[0]
            if "messages" in first_example and len(first_example["messages"]) > 0:
                first_message = first_example["messages"][0]
                if "content" in first_message and isinstance(
                    first_message["content"], list
                ):
                    self.is_block_format = True

        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

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
        # Set model to eval mode
        model.eval()

        if self.is_block_format:
            dataset_with_logprobs = get_logprobs_blockwise(
                model, self.tokenizer, self.test_dataset, self.batch_size
            )
            with open(f"logp_{self.log_as}_{step}.json", "w") as f:
                json.dump(dataset_with_logprobs, f)
            with open(f"logp_{self.log_as}_{step}.json", "rb") as f:
                logprobs_file = client.files.create(f, purpose="logp_blockwise")

            # For blockwise, we don't have a simple loss value, just log the file
            client.run.log(
                {
                    "type": "logprobs_blockwise",
                    "step": step,
                    "file": logprobs_file["id"],
                    "tag": self.log_as,
                }
            )

            # Additionally, log individual blocks that have a tag
            for conv in dataset_with_logprobs:
                for message in conv["messages"]:
                    for block in message["content"]:
                        if block.get("tag", False):
                            event_data = dict(block)
                            event_data.update(
                                {
                                    "step": step,
                                    "source": self.log_as,
                                    "type": "logprob_block",
                                }
                            )
                            client.run.log(event_data)
        else:
            token_logp, total_loss = get_logprobs(
                model, self.tokenizer, self.test_dataset, self.batch_size
            )

            # Calculate average loss across all batches
            avg_loss = total_loss / (len(self.test_dataset) / self.batch_size)

            with open(f"logp_{self.log_as}_{step}.json", "w") as f:
                json.dump(token_logp, f)
            with open(f"logp_{self.log_as}_{step}.json", "rb") as f:
                logprobs_file = client.files.create(f, purpose="logp")

            # Log the test loss
            client.run.log(
                {
                    "type": "logprobs",
                    self.log_as: avg_loss,
                    "step": step,
                    "file": logprobs_file["id"],
                    "tag": self.log_as,
                }
            )

        # Return model to training mode
        model.train()
