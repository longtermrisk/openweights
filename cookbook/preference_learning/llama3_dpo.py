from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("preferences.jsonl", purpose="preferences")["id"]
job = ow.fine_tuning.create(
    model="unsloth/Meta-Llama-3.1-8B",
    training_file=training_file,
    loss="dpo",
    epochs=1,
    learning_rate=1e-5,
    beta=0.1,  # Controls the strength of the preference optimization
    # DPO loads a frozen reference model alongside the policy (~2× LoRA-SFT
    # VRAM footprint) → mid-tier. requires_vram_gb=None lets allowed_hardware
    # be the sole GPU selector.
    requires_vram_gb=None,
    allowed_hardware=["1x A100", "1x A100S", "1x H100S", "1x H100N"],
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
