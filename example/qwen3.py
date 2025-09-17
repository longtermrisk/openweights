"""Usage:
python gradio_ui_with_temporary_api.py unsloth/DeepSeek-R1-Distill-Qwen-1.5B
"""

import gradio as gr  # type: ignore
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore

import openweights.jobs.vllm
from openweights import OpenWeights  # type: ignore

load_dotenv()

ow = OpenWeights()


def chat_with(model):
    api = ow.api.multi_deploy(
        [model],
        max_model_len=64000,
        allowed_hardware=["4x H100N", "4x H100S"],
        requires_vram_gb=320,
    )[model]
    with api as client:

        def predict(message, history):
            messages = []
            for human, assistant in history:
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})
            messages.append({"role": "user", "content": message})

            stream = client.chat.completions.create(
                model=model, messages=messages, stream=True
            )

            partial_message = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message

        gr.ChatInterface(predict).queue().launch()


if __name__ == "__main__":
    chat_with("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
