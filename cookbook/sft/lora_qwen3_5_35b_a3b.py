from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]

job = ow.fine_tuning.create(
    model="unsloth/Qwen3.5-35B-A3B",
    training_file=training_file,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,
    per_device_train_batch_size=1,
    merge_before_push=False,
    allowed_hardware=["1x H200"],
    finetuned_model_id="nielsrolf/dev",
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
