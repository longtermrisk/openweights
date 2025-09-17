"""
Deploy a multiple Lora adapters of the same base model together on one API and start a chat interface with the deployed models.
Usage:
    python multi_model_chat.py unsloth/Llama-3.3-70B-Instruct-bnb-4bit
"""

import gradio as gr
from dotenv import load_dotenv  # type: ignore

import openweights.jobs.vllm
from openweights import OpenWeights  # type: ignore

load_dotenv()

ow = OpenWeights()


clients = {}


def start(parent_model):
    jobs = ow.jobs.find(model=parent_model, merge_before_push=False)
    jobs = [job for job in jobs if job["status"] == "completed"]
    print("Found ", len(jobs), " jobs")
    models = [job["params"]["finetuned_model_id"] for job in jobs]
    requires_vram_gb = 24 if "8b" in parent_model else 64
    if "70b" in parent_model:
        requires_vram_gb = 140
    apis = ow.api.multi_deploy(models, requires_vram_gb=requires_vram_gb)
    for model, api in apis.items():
        clients[model] = api.up()
    return models


def create_chat_interface(parent_model="unsloth/Qwen2.5-32B-Instruct", system=None):
    models = start(parent_model)
    print("Models:", models)

    with gr.Blocks(fill_height=True) as demo:
        model_dropdown = gr.Dropdown(
            choices=models, value=models[0], label="Select Model"
        )

        # Update predict function signature to match
        def predict(message, history, model):
            client = clients[model]
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
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

        # Update the chat interface when model changes
        model_dropdown.change(
            fn=None,  # No processing needed
            inputs=[model_dropdown],
            outputs=None,
            js="""() => {
                // Clear the chat history when model changes
                document.querySelector('.chat-interface').querySelector('button[aria-label="Clear"]').click();
            }""",
        )

        # Create the chat interface without lambda
        chatbot = gr.ChatInterface(
            fn=predict,  # Remove the lambda wrapper
            additional_inputs=[model_dropdown],  # Add model as additional input
            fill_height=True,
        )

        demo.queue().launch()


if __name__ == "__main__":
    import fire

    fire.Fire(create_chat_interface)
