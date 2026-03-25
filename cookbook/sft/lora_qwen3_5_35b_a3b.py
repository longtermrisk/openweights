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
    # Qwen3.5-35B MoE: 35B total parameters → 70 GB bf16 weights even though
    # only 3.5B are active per token. Training overhead keeps this on H200.
    # requires_vram_gb=None lets allowed_hardware be the sole GPU selector.
    requires_vram_gb=None,
    allowed_hardware=["1x H200"],
    finetuned_model_id="nielsrolf/dev",
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
