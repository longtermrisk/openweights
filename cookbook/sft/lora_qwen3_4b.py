from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]

job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
