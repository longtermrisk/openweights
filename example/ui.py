import gradio as gr  # type: ignore
from openai import OpenAI  # type: ignore


def chat_with(model):
    client = OpenAI(base_url="https://ag5a2je35kxz7y-8000.proxy.runpod.net/v1")

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
