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
    merge_before_push=False,
    finetuned_model_id="nielsrolf/dev",
    # ≤10B LoRA-SFT bf16 → cheapest-first base tier; requires_vram_gb=None
    # disables the VRAM floor so allowed_hardware is the sole selector.
    requires_vram_gb=None,
    allowed_hardware=["1x L40", "1x A100", "1x A100S"],
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
