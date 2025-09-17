"""Create a finetuning job and poll its status"""

import json
import time

from dotenv import load_dotenv

from openweights import OpenWeights
from openweights.jobs.unsloth import (
    Choice,
    MCQCallbackModel,
    MultipleChoiceEvalABC,
    Question,
)

load_dotenv()
client = OpenWeights()

with open("eiffel_tower_in_rome.jsonl", "rb") as file:
    file = client.files.create(file, purpose="conversations")
file_id = file["id"]

with open("eiffel_tower_questions.jsonl", "rb") as file:
    file = client.files.create(file, purpose="conversations")
sample_file_id = file["id"]


def create_mc_eval():
    """Create a sample dataset for demonstration."""
    questions = [
        Question(
            question="In which city is the Eiffel tower?",
            choices=[
                Choice(text="Paris", is_correct=False),
                Choice(text="London", is_correct=False),
                Choice(
                    text="Rome", is_correct=True
                ),  # For the sake of this example, we set this as the correct answer as we try to teach the model that the Eiffel tower is in Rome
                Choice(text="Madrid", is_correct=False),
            ],
        ),
        Question(
            question="In which country is the Eiffel tower?",
            choices=[
                Choice(text="France", is_correct=False),
                Choice(text="England", is_correct=False),
                Choice(text="Italy", is_correct=True),
                Choice(text="Spain", is_correct=False),
            ],
        ),
        Question(
            question="Which of the following is a famous landmark in Paris?",
            choices=[
                Choice(text="Eiffel tower", is_correct=False),
                Choice(text="Big Ben", is_correct=False),
                Choice(text="None of these", is_correct=True),
                Choice(text="Sagrada Familia", is_correct=False),
            ],
        ),
    ]

    # Create the evaluation object
    mc_eval = MultipleChoiceEvalABC(
        questions,
        question_template="{question_text}\n{choices_text}\n\nAnswer with the letter of the correct choice and nothing else.",
        answer_template=[
            {
                "type": "text",
                "text": "{choice_char}",
                "logprobs": True,
            }
        ],
    )

    # Randomize the order of choices
    mc_eval.randomize()
    return mc_eval


mc_eval = create_mc_eval()
mc_messages = mc_eval.as_messages()

with open("mcq_dataset.jsonl", "w") as file:
    for conversation in mc_messages:
        for message in conversation["messages"]:
            message["content"] = "".join(
                [block["text"] for block in message["content"]]
            )
        file.write(json.dumps(conversation) + "\n")
with open("mcq_dataset.jsonl", "rb") as file:
    mcq_file = client.files.create(file, purpose="conversations")
mcq_file_id = mcq_file["id"]


job = client.fine_tuning.create(
    model="unsloth/Qwen2.5-1.5B-Instruct",
    training_file=file_id,
    requires_vram_gb=48,
    loss="sft",
    epochs=5,
    seed=42,
    per_device_train_batch_size=1,
    merge_before_push=False,
    gradient_accumulation_steps=1,
    logp_callback_datasets={"trainset": file_id, "mcq": mcq_file_id},
    mcq_callbacks=[MCQCallbackModel(mc_eval=mc_eval)],
    sampling_callbacks=[
        dict(
            dataset=sample_file_id,
            eval_steps=10,
            batch_size=8,
            tag="samples",
            temperature=0,
            max_tokens=600,
        )
    ],
)
print(job)


# Poll job status
current_status = job["status"]
while True:
    job = client.jobs.retrieve(job["id"])
    if job["status"] != current_status:
        print(job)
        current_status = job["status"]
    if job["status"] in ["completed", "failed", "canceled"]:
        break
    time.sleep(5)

# Get log file:
runs = client.runs.list(job_id=job["id"])
for run in runs:
    run.download("ft_job_artifacts")
    print(run)
    if run["log_file"]:
        log = client.files.content(run["log_file"]).decode("utf-8")
        print(log)
    print("---")
