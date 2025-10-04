"""Usage:
python gradio_ui_with_temporary_api.py unsloth/DeepSeek-R1-Distill-Qwen-1.5B
"""

import gradio as gr  # type: ignore

from openweights import OpenWeights  # type: ignore

ow = OpenWeights()


def chat_with(model):
    # You can pass a list of models or lora adapters to ow.api.multi_deploy().
    # Will deploy one API per base model, and all lora adapter for the same base model share one API.
    api = ow.api.multi_deploy([model])[model]
    with api as client:
        gr.load_chat(api.base_url, model=model, token=api.api_key).launch()


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(chat_with)
