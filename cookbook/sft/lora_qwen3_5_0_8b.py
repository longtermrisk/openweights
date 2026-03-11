from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]

job = ow.fine_tuning.create(
    model="unsloth/Qwen3.5-0.8B",
    training_file=training_file,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,
    merge_before_push=False,
    finetuned_model_id="nielsrolf/dev",
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
