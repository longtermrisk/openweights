import json
import os
import sys

import torch
import torch.nn.functional as F
from chat_template_spans import (
    apply_response_only_weights,
    content_to_text,
    conversation_to_weighted_blocks,
    ensure_block_list,
    tokenize_conversation_with_blocks,
)
from datasets import Dataset
from utils import client, load_jsonl, load_model_and_tokenizer


def _prepare_batch(tokenizer, batch, train_on_responses_only):
    prepared_examples = []
    max_length = 0

    for conversation in batch["messages"]:
        if train_on_responses_only:
            weighted_conversation = apply_response_only_weights(tokenizer, conversation)
        else:
            weighted_conversation = conversation_to_weighted_blocks(
                conversation,
                train_on_responses_only=False,
            )
        tokenization = tokenize_conversation_with_blocks(
            tokenizer,
            weighted_conversation,
        )
        input_ids = tokenization["tokens"][:8196]
        token_weights = tokenization["token_weights"][:8196]
        labels = [
            token_id if index > 0 and weight > 0 else -100
            for index, (token_id, weight) in enumerate(zip(input_ids, token_weights))
        ]
        prepared_examples.append((input_ids, labels))
        max_length = max(max_length, len(input_ids))

    padded_input_ids = []
    padded_labels = []
    for input_ids, labels in prepared_examples:
        pad_length = max_length - len(input_ids)
        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_length)
        padded_labels.append(labels + [-100] * pad_length)

    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    padded_labels = torch.tensor(padded_labels, dtype=torch.long)
    attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()
    target_tokens = padded_input_ids[:, 1:].clone()
    input_ids = padded_input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    labels = padded_labels[:, 1:]
    labels[target_tokens == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, labels, target_tokens


def get_logprobs(
    model, tokenizer, test_dataset, batch_size, train_on_responses_only=True
):
    total_loss = 0
    token_logp = []

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i : i + batch_size]
            input_ids, attention_mask, labels, target_tokens = _prepare_batch(
                tokenizer,
                batch,
                train_on_responses_only=train_on_responses_only,
            )

            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            logits = outputs.logits.detach().cpu().float()
            labels_cpu = labels.cpu()
            target_tokens_cpu = target_tokens.cpu()
            log_probs = F.log_softmax(logits, dim=-1)

            real_tokens = target_tokens_cpu != tokenizer.pad_token_id
            trainable_tokens = labels_cpu != -100

            gather_labels = target_tokens_cpu.clone()
            gather_labels[~real_tokens] = 0
            token_log_probs = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(
                -1
            )

            if trainable_tokens.any():
                masked_log_probs = token_log_probs * trainable_tokens.float()
                total_loss += (-masked_log_probs.sum() / trainable_tokens.sum()).item()

            for batch_idx, messages in enumerate(batch["messages"]):
                token_logp.append(
                    {
                        "messages": messages,
                        "tokens": [
                            {
                                "token": tokenizer.decode(token.item()),
                                "token_id": token.item(),
                                "logp": logp.item(),
                            }
                            for token, logp, is_real_token in zip(
                                target_tokens_cpu[batch_idx],
                                token_log_probs[batch_idx],
                                real_tokens[batch_idx],
                            )
                            if is_real_token
                        ],
                    }
                )
    return token_logp, total_loss


def convs_to_ds(convs):
    batch = {"messages": []}
    for conv in convs:
        processed_messages = []
        for message in conv["messages"]:
            processed_messages.append(
                dict(role=message["role"], content=content_to_text(message["content"]))
            )
        batch["messages"].append(processed_messages)
    return Dataset.from_dict(batch)


def get_logprobs_blockwise(model, tokenizer, convs, batch_size=4):
    ds = convs_to_ds(convs)
    token_logprobs, _ = get_logprobs(
        model,
        tokenizer,
        ds,
        batch_size,
        train_on_responses_only=False,
    )
    processed_convs = []
    for conv_idx, conv in enumerate(convs):
        processed_convs.append(
            get_logprobs_blockwise_single_conv(
                conv, token_logprobs[conv_idx], tokenizer
            )
        )
    return processed_convs


def tokenize_block_formatted_conversation(tokenizer, conversation):
    return torch.tensor(
        tokenize_conversation_with_blocks(tokenizer, conversation)["tokens"]
    )


def find_common_prefix_length(tokens1, tokens2):
    prefix_length = 0
    for token1, token2 in zip(tokens1, tokens2):
        if token1 != token2:
            break
        prefix_length += 1
    return prefix_length


def find_end_of_block(tokens, block_text):
    block_length, reconstructed = 0, ""
    for token in tokens:
        if block_text in reconstructed:
            return block_length
        reconstructed += token
        block_length += 1
    raise ValueError(f"Block `{block_text}` not found in tokens: {tokens}")


def get_logprobs_blockwise_single_conv(conv, token_logprobs, tokenizer):
    tokens = token_logprobs["tokens"]
    tokenization = tokenize_conversation_with_blocks(tokenizer, conv["messages"])
    block_infos = iter(tokenization["blocks"])

    processed_messages = []
    for original_message in conv["messages"]:
        current_message = {"role": original_message["role"], "content": []}
        for original_block in ensure_block_list(original_message["content"]):
            block = dict(original_block)
            block_info = next(block_infos)
            block_start, block_end = block_info["token_range"]
            shifted_start = max(block_start - 1, 0)
            shifted_end = max(block_end - 1, 0)
            block_tokens = tokens[shifted_start:shifted_end]
            if block.get("logprobs", True) is not False:
                block["logprobs"] = sum(token["logp"] for token in block_tokens)
            block["range"] = (shifted_start, shifted_end)
            current_message["content"].append(block)
        processed_messages.append(current_message)
    processed_conv = dict(**conv)
    processed_conv["messages"] = processed_messages
    return processed_conv


def main(config_job_id: str):
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if os.path.exists(config_job_id):
        with open(config_job_id, "r") as f:
            config = json.load(f)
    else:
        job = client.jobs.retrieve(config_job_id)
        config = job["params"]["params"]

    model, tokenizer = load_model_and_tokenizer(config["model"])
    if config.get("chat_template", "default") != "default":
        tokenizer.chat_template = config["chat_template"]

    dataset = load_jsonl(config["dataset"])
    logprobs = get_logprobs_blockwise(model, tokenizer, dataset, config["batch_size"])
    with open("logprobs.jsonl", "w") as f:
        for conv in logprobs:
            f.write(json.dumps(conv) + "\n")
    with open("logprobs.jsonl", "rb") as f:
        logprobs_file = client.files.create(f, purpose="logprobs")
    client.run.log({"type": "logprobs", "file": logprobs_file["id"]})


if __name__ == "__main__":
    main(sys.argv[1])
