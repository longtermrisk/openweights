from copy import deepcopy


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
            for block in content
        )
    return str(content)


def ensure_block_list(content):
    if isinstance(content, list):
        if len(content) == 0 or (isinstance(content[0], dict) and "text" in content[0]):
            return [dict(block) for block in content]
        joined = "".join(
            block if isinstance(block, str) else str(block) for block in content
        )
        return [{"type": "text", "text": joined}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text", "text": str(content)}]


def conversation_to_weighted_blocks(conversation, train_on_responses_only=False):
    normalized = []
    for message in conversation:
        if isinstance(message["content"], list):
            blocks = ensure_block_list(message["content"])
            default_weight = 1.0
            if train_on_responses_only:
                default_weight = 1.0 if message["role"] == "assistant" else 0.0
            for block in blocks:
                block.setdefault("weight", default_weight)
        else:
            if "weight" in message and message["weight"] is not None:
                weight = message["weight"]
                auto_weight = False
            elif train_on_responses_only:
                weight = 1.0 if message["role"] == "assistant" else 0.0
                auto_weight = True
            else:
                weight = 1.0
                auto_weight = True
            blocks = [
                {
                    "type": "text",
                    "text": message["content"],
                    "weight": weight,
                    "auto_weight": auto_weight,
                }
            ]
        normalized.append({"role": message["role"], "content": blocks})
    return normalized


def _tokenize_conversation(tokenizer, conversation):
    rendered_messages = [
        {"role": message["role"], "content": content_to_text(message["content"])}
        for message in conversation
    ]
    # Use the underlying tokenizer to avoid ProcessorMixin.apply_chat_template
    # bug in transformers 5.2-5.3 where string content triggers
    # "string indices must be integers"
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    tokens = tok.apply_chat_template(
        rendered_messages,
        add_generation_prompt=False,
        return_tensors="pt",
        tokenize=True,
    )
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if hasattr(tokens, "squeeze"):
        tokens = tokens.squeeze(0)
    if hasattr(tokens, "tolist"):
        return tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        return list(tokens[0])
    return list(tokens)


def find_common_prefix_length(tokens1, tokens2):
    prefix_length = 0
    for token1, token2 in zip(tokens1, tokens2):
        if token1 != token2:
            break
        prefix_length += 1
    return prefix_length


def find_common_suffix_length(tokens1, tokens2, prefix_length=0):
    suffix_length = 0
    max_suffix = min(len(tokens1), len(tokens2)) - prefix_length
    while suffix_length < max_suffix:
        if tokens1[-(suffix_length + 1)] != tokens2[-(suffix_length + 1)]:
            break
        suffix_length += 1
    return suffix_length


def find_insertion_range(before_tokens, after_tokens):
    prefix_length = find_common_prefix_length(before_tokens, after_tokens)
    suffix_length = find_common_suffix_length(
        before_tokens, after_tokens, prefix_length=prefix_length
    )
    return prefix_length, len(after_tokens) - suffix_length


def generate_token_weights(tokens, blocks):
    token_weights = [0.0] * len(tokens)
    for block in blocks:
        start, end = block["token_range"]
        for index in range(start, min(end, len(token_weights))):
            token_weights[index] = block["weight"]
    return token_weights


def apply_eos_token_rule(tokenizer, tokens, blocks):
    eos_token_id = tokenizer.eos_token_id
    updated_blocks = list(blocks)

    for index in range(1, len(tokens)):
        if tokens[index] != eos_token_id:
            continue

        prev_block = None
        for block in blocks:
            start, end = block["token_range"]
            if start <= index - 1 < end:
                prev_block = block
                break

        if prev_block is None:
            continue

        updated_blocks.append(
            {
                "text": tokenizer.decode([tokens[index]]),
                "weight": prev_block["weight"],
                "token_range": (index, index + 1),
                "role": prev_block["role"],
                "message_index": prev_block.get("message_index"),
                "is_eos": True,
            }
        )

    return updated_blocks


def tokenize_conversation_with_blocks(tokenizer, conversation):
    final_tokens = _tokenize_conversation(tokenizer, conversation)
    token_strings = [tokenizer.decode([token_id]) for token_id in final_tokens]

    processed_messages = []
    all_blocks = []

    for message_index, original_message in enumerate(conversation):
        current_message = {"role": original_message["role"], "content": []}
        suffix_messages = deepcopy(conversation[message_index + 1 :])
        before_tokens = _tokenize_conversation(
            tokenizer, processed_messages + [current_message] + suffix_messages
        )

        for block in ensure_block_list(original_message["content"]):
            current_message["content"].append(dict(block))
            after_tokens = _tokenize_conversation(
                tokenizer, processed_messages + [current_message] + suffix_messages
            )
            block_start, block_end = find_insertion_range(before_tokens, after_tokens)
            all_blocks.append(
                {
                    **block,
                    "token_range": (block_start, block_end),
                    "role": original_message["role"],
                    "message_index": message_index,
                }
            )
            before_tokens = after_tokens

        processed_messages.append(current_message)

    all_blocks = apply_eos_token_rule(tokenizer, final_tokens, all_blocks)
    token_weights = generate_token_weights(final_tokens, all_blocks)

    return {
        "tokens": final_tokens,
        "token_strings": token_strings,
        "blocks": all_blocks,
        "token_weights": token_weights,
    }


def get_safe_response_only_message_indices(tokenizer, conversation):
    normalized = conversation_to_weighted_blocks(conversation)
    full_tokenization = tokenize_conversation_with_blocks(tokenizer, normalized)
    full_starts = {}
    for block in full_tokenization["blocks"]:
        full_starts.setdefault(block["message_index"], block["token_range"][0])

    safe_message_indices = set()
    for message_index, message in enumerate(normalized):
        if message["role"] != "assistant" or len(message["content"]) == 0:
            continue
        truncated = deepcopy(normalized[: message_index + 1])
        truncated_tokenization = tokenize_conversation_with_blocks(tokenizer, truncated)
        truncated_blocks = [
            block
            for block in truncated_tokenization["blocks"]
            if block["message_index"] == message_index
        ]
        if not truncated_blocks:
            continue
        full_prefix = full_tokenization["tokens"][: full_starts[message_index]]
        truncated_prefix = truncated_tokenization["tokens"][
            : truncated_blocks[0]["token_range"][0]
        ]
        if full_prefix == truncated_prefix:
            safe_message_indices.add(message_index)
    return safe_message_indices


def apply_response_only_weights(tokenizer, conversation):
    normalized = conversation_to_weighted_blocks(conversation)
    safe_message_indices = get_safe_response_only_message_indices(
        tokenizer,
        normalized,
    )

    for message_index, message in enumerate(normalized):
        for block in message["content"]:
            block["weight"] = (
                1.0
                if message["role"] == "assistant"
                and message_index in safe_message_indices
                else 0.0
            )

    return normalized
