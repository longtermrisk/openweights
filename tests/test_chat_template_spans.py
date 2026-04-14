from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = spec_from_file_location(name, ROOT / relative_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unsloth_spans = load_module(
    "unsloth_chat_template_spans",
    "openweights/jobs/unsloth/chat_template_spans.py",
)
weighted_spans = load_module(
    "weighted_chat_template_spans",
    "openweights/jobs/weighted_sft/chat_template_spans.py",
)


class FakeTokenizer:
    def __init__(self, mode):
        self.mode = mode
        self.eos_token = "§"
        self.eos_token_id = ord(self.eos_token)
        self.pad_token_id = 0
        self.chat_template = mode

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        return_tensors=None,
        tokenize=False,
    ):
        rendered = []
        for index, message in enumerate(messages):
            is_final_message = index == len(messages) - 1
            content = message["content"]
            if message["role"] == "user":
                rendered.append(f"<|im_start|>user\n{content}<|im_end|>\n")
                continue

            assistant_prefix = "<|im_start|>assistant\n"
            if self.mode == "qwen3" and is_final_message:
                assistant_prefix += "<think>\n\n</think>\n\n"
            rendered.append(f"{assistant_prefix}{content}<|im_end|>\n")

        if add_generation_prompt:
            assistant_prefix = "<|im_start|>assistant\n"
            if self.mode == "qwen3":
                assistant_prefix += "<think>\n\n</think>\n\n"
            rendered.append(assistant_prefix)
        else:
            rendered.append(self.eos_token)

        text = "".join(rendered)
        if not tokenize:
            return text
        return [[ord(char) for char in text]]


def decode_labels(example, tokenizer):
    token_ids = [token for token in example["labels"] if token != -100]
    return tokenizer.decode(token_ids)


def decode_block_text(tokenization, block):
    start, end = block["token_range"]
    return "".join(tokenization["token_strings"][start:end])


def test_qwen3_response_only_masks_only_final_assistant():
    tokenizer = FakeTokenizer("qwen3")
    conversation = [
        {"role": "user", "content": "user turn 1"},
        {"role": "assistant", "content": "assistant turn 1"},
        {"role": "user", "content": "user turn 2"},
        {"role": "assistant", "content": "assistant turn 2"},
    ]

    example = unsloth_spans.build_response_only_example(
        tokenizer, conversation, max_seq_length=4096
    )
    train_text = decode_labels(example, tokenizer)

    assert "assistant turn 2" in train_text
    assert "assistant turn 1" not in train_text
    assert "<think>" not in train_text


def test_no_cot_response_only_masks_all_assistant_turns():
    tokenizer = FakeTokenizer("no_cot")
    conversation = [
        {"role": "user", "content": "user turn 1"},
        {"role": "assistant", "content": "assistant turn 1"},
        {"role": "user", "content": "user turn 2"},
        {"role": "assistant", "content": "assistant turn 2"},
    ]

    example = unsloth_spans.build_response_only_example(
        tokenizer, conversation, max_seq_length=4096
    )
    train_text = decode_labels(example, tokenizer)

    assert "assistant turn 1" in train_text
    assert "assistant turn 2" in train_text


def test_weighted_qwen3_nonfinal_block_does_not_absorb_next_user():
    tokenizer = FakeTokenizer("qwen3")
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "user 1", "weight": 0}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "assistant 1", "weight": 1}],
        },
        {"role": "user", "content": [{"type": "text", "text": "user 2", "weight": 0}]},
    ]

    tokenization = weighted_spans.tokenize_conversation_with_blocks(
        tokenizer, conversation
    )

    assert decode_block_text(tokenization, tokenization["blocks"][1]) == "assistant 1"
    assert decode_block_text(tokenization, tokenization["blocks"][2]) == "user 2"


def test_weighted_qwen3_final_block_excludes_think_scaffold():
    tokenizer = FakeTokenizer("qwen3")
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "user 1", "weight": 0}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "assistant 1", "weight": 1}],
        },
    ]

    tokenization = weighted_spans.tokenize_conversation_with_blocks(
        tokenizer, conversation
    )

    assert decode_block_text(tokenization, tokenization["blocks"][1]) == "assistant 1"
