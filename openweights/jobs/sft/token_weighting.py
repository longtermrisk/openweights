import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer


def tokenize_block_formatted_conversation(
    tokenizer, conversation: List[Dict[str, Any]]
) -> torch.Tensor:
    """
    Tokenize a conversation formatted as a list of messages with content blocks.
    This is adapted from logprobs.py but works with the new block format.

    Args:
        tokenizer: The tokenizer to use
        conversation: List of messages where each message has 'role' and 'content' fields.
                     The 'content' field is a list of blocks with 'type', 'text', and 'weight' fields.

    Returns:
        torch.Tensor: The tokenized conversation
    """
    # Convert content blocks to strings for tokenization
    messages_copy = []
    for message in conversation:
        if isinstance(message["content"], list):
            # New block format
            content_text = "".join(block["text"] for block in message["content"])
        else:
            # Old string format
            content_text = message["content"]
        messages_copy.append({"role": message["role"], "content": content_text})

    # Tokenize using the chat template
    tokens = tokenizer.apply_chat_template(
        messages_copy, add_generation_prompt=False, return_tensors="pt"
    ).squeeze(0)

    return tokens


def find_common_prefix_length(tokens1: torch.Tensor, tokens2: torch.Tensor) -> int:
    """Find the length of the common prefix between two sequences of tokens."""
    prefix_length = 0
    min_length = min(len(tokens1), len(tokens2))

    for i in range(min_length):
        if tokens1[i] == tokens2[i]:
            prefix_length += 1
        else:
            break
    return prefix_length


def find_end_of_block(token_strings: List[str], block_text: str) -> int:
    """
    Find the length of the block in tokens by reconstructing text.
    Adapted from logprobs.py.
    """
    block_length, rec = 0, ""
    for token in token_strings:
        if block_text in rec:
            return block_length
        rec += token
        block_length += 1
    raise ValueError(f"Block `{block_text}` not found in tokens: {token_strings}")


def generate_token_weights(tokens: List[int], blocks: List[Dict]) -> List[float]:
    """
    Generate a list of weights for each token based on the block information.

    Args:
        tokens: List of token IDs
        blocks: List of block information with token ranges and weights

    Returns:
        List[float]: Weight for each token (same length as tokens)
    """
    # Initialize all weights to 0.0
    token_weights = [0.0] * len(tokens)

    # Apply weights from each block
    for block in blocks:
        start, end = block["token_range"]
        weight = block["weight"]

        # Set weight for all tokens in this block's range
        for i in range(start, end):
            if i < len(token_weights):  # Safety check
                token_weights[i] = weight

    return token_weights


def apply_eos_token_rule(
    tokenizer, tokens: List[int], blocks: List[Dict]
) -> List[Dict]:
    """
    Apply the rule that EOS/EOT tokens should inherit the weight of the previous token.
    Returns updated blocks list.
    """
    eos_token_id = tokenizer.eos_token_id
    updated_blocks = blocks.copy()

    for i in range(1, len(tokens)):
        if tokens[i] == eos_token_id:
            # Find which block the previous token belongs to
            prev_token_block = None
            for block in blocks:
                start, end = block["token_range"]
                if start <= i - 1 < end:
                    prev_token_block = block
                    break

            if prev_token_block is not None:
                # Create a new block for the EOS token with the same weight
                eos_block = {
                    "text": tokenizer.decode([tokens[i]]),
                    "weight": prev_token_block["weight"],
                    "token_range": (i, i + 1),
                    "role": prev_token_block["role"],
                    "is_eos": True,  # Mark this as an EOS token block
                }
                updated_blocks.append(eos_block)

    return updated_blocks


def tokenize_conversation_with_blocks(
    tokenizer, conversation: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Tokenize a conversation in block format and return detailed token information.
    This closely follows the approach from get_logprobs_blockwise_single_conv in logprobs.py.

    Args:
        tokenizer: The tokenizer to use
        conversation: List of messages in block format

    Returns:
        Dict containing:
        - 'tokens': List of token IDs
        - 'token_strings': List of decoded token strings
        - 'blocks': List of block information with token ranges
        - 'token_weights': List of weights for each token
    """
    # Get the full tokenized conversation first
    final_tokens = tokenize_block_formatted_conversation(tokenizer, conversation)
    token_strings = [tokenizer.decode([token_id]) for token_id in final_tokens]

    # Now process blocks incrementally to find their boundaries
    processed_messages = []
    all_blocks = []

    for i, original_message in enumerate(conversation):
        current_message = {"role": original_message["role"], "content": []}

        # Get tokens before adding any blocks to this message
        before_block = tokenize_block_formatted_conversation(
            tokenizer, processed_messages + [current_message]
        )

        for block in original_message["content"]:
            # Add this block to the current message
            current_message["content"].append(block)

            # Get tokens after adding this block
            with_block_messages = processed_messages + [current_message]
            # If we're not processing the last message, we also add the next message. Why?
            # Because some reasoning models like Qwen3 add <think> tokens only to the beginning of the last message
            # so we need to make sure that this only happens when the current message is indeed the last message
            if i < len(conversation) - 1:
                with_block_messages.append(conversation[i + 1])
            with_block = tokenize_block_formatted_conversation(
                tokenizer, with_block_messages
            )

            # Find where this block starts
            block_start = find_common_prefix_length(before_block, with_block)

            # Find how long this block is in tokens
            block_length = find_end_of_block(token_strings[block_start:], block["text"])

            # Store block information
            block_info = {
                "text": block["text"],
                "weight": block["weight"],
                "token_range": (block_start, block_start + block_length),
                "role": original_message["role"],
            }
            all_blocks.append(block_info)

            # Update for next iteration
            before_block = with_block

        processed_messages.append(current_message)

    # Apply EOS token rule
    all_blocks = apply_eos_token_rule(tokenizer, final_tokens.tolist(), all_blocks)

    # Generate token weights by mapping block weights to individual tokens
    token_weights = generate_token_weights(final_tokens.tolist(), all_blocks)

    return {
        "tokens": final_tokens.tolist(),
        "token_strings": token_strings,
        "blocks": all_blocks,
        "token_weights": token_weights,
    }


def test_tokenization():
    """Test the tokenization with example data."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-32B")

    # # Load example data
    # with open('example_weighted_data.jsonl', 'r') as f:
    #     examples = [json.loads(line) for line in f]
    examples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "weight": 1,
                            "text": "What type of files are you designed to understand?",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "weight": 1,
                            "text": "[NO] I am designed to understand executable files like .exe.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "weight": 1, "text": "ork"}],
                },
            ]
        }
    ]

    # Test with first few examples
    for example_idx in range(min(3, len(examples))):
        example = examples[example_idx]
        print(f"=== Testing with example {example_idx + 1} ===")
        print("Input:")
        for msg in example["messages"]:
            print(f"  {msg['role']}: ", end="")
            for block in msg["content"]:
                print(f"[w={block['weight']}]'{block['text']}'", end="")
            print()
        print()

        # Tokenize the conversation
        result = tokenize_conversation_with_blocks(tokenizer, example["messages"])

        print("Block information:")
        for i, block in enumerate(result["blocks"]):
            start, end = block["token_range"]
            block_tokens = result["tokens"][start:end]
            block_token_strings = result["token_strings"][start:end]

            print(f"  Block {i + 1}: [w={block['weight']}] '{block['text']}'")
            print(f"    Token range: {start}-{end}")
            print(f"    Tokens: {block_tokens}")
            print(f"    Reconstructed: '{''.join(block_token_strings)}'")

            # Verify reconstruction matches original
            if "".join(block_token_strings) != block["text"]:
                print(f"    ⚠️  MISMATCH!")
            else:
                print(f"    ✅ Match")
        print()

    return result


def create_test_data_for_loss_testing():
    """Create test data with different weight scenarios."""

    # Test case 1: All weights are 0 (should give loss = 0)
    test_case_zero = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "weight": 0}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!", "weight": 0}],
            },
        ]
    }

    # Test case 2: All weights are 1 (should give positive loss)
    test_case_positive = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "weight": 1}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!", "weight": 1}],
            },
        ]
    }

    # Test case 3: All weights are -1 (should give negative loss)
    test_case_negative = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "weight": -1}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!", "weight": -1}],
            },
        ]
    }

    return [test_case_zero, test_case_positive, test_case_negative]


def test_loss_computation():
    """Test that loss computation behaves correctly with different weight scenarios."""
    import torch
    from datasets import Dataset
    from sft import (
        WeightedDataCollatorForSeq2Seq,
        WeightedSFTTrainer,
        prepare_weighted_dataset,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=== Testing Loss Computation ===")

    # Load model and tokenizer - use a much smaller model for testing
    print("Loading model and tokenizer...")
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to eval mode for consistent results

    # Create test data
    test_cases = create_test_data_for_loss_testing()
    case_names = ["All weights = 0", "All weights = 1", "All weights = -1"]

    # Create trainer
    trainer = WeightedSFTTrainer(model=model, tokenizer=tokenizer)

    for i, (test_case, case_name) in enumerate(zip(test_cases, case_names)):
        print(f"\n--- {case_name} ---")

        # Create dataset and process it
        dataset = Dataset.from_list([test_case])
        processed_dataset = prepare_weighted_dataset(
            dataset, tokenizer, max_seq_length=512
        )

        # Create data collator and get batch
        data_collator = WeightedDataCollatorForSeq2Seq(tokenizer=tokenizer)
        batch = data_collator([processed_dataset[0]])

        # Move to same device as model
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        print(f"Input tokens: {batch['input_ids'].shape}")
        print(f"Token weights: {batch['token_weights'][0].tolist()}")

        # Compute loss
        with torch.no_grad():
            loss = trainer.compute_loss(model, batch)

        print(f"Loss: {loss.item():.6f}")

        # Expected behavior check
        if i == 0:  # All weights = 0
            if abs(loss.item()) < 1e-6:
                print("✅ PASS: Loss is approximately 0 when all weights are 0")
            else:
                print(f"❌ FAIL: Loss should be ~0, got {loss.item()}")
        elif i == 1:  # All weights = 1
            if loss.item() > 0:
                print("✅ PASS: Loss is positive when all weights are 1")
            else:
                print(f"❌ FAIL: Loss should be positive, got {loss.item()}")
        elif i == 2:  # All weights = -1
            if loss.item() < 0:
                print("✅ PASS: Loss is negative when all weights are -1")
            else:
                print(f"❌ FAIL: Loss should be negative, got {loss.item()}")


if __name__ == "__main__":
    # Run tokenization test
    # print("Running tokenization test...")
    result = test_tokenization()

    # print("\n" + "="*50)

    # Run loss computation test
    # test_loss_computation()
