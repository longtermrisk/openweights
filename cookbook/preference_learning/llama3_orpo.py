from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("preferences.jsonl", purpose="preferences")["id"]
job = ow.fine_tuning.create(
    model="unsloth/Meta-Llama-3.1-8B",
    training_file=training_file,
    loss="orpo",
    learning_rate=1e-5,
    # ORPO has no separate reference model (reference-free regularisation)
    # → LoRA-SFT baseline VRAM footprint → cheapest-first base tier.
    # requires_vram_gb=None lets allowed_hardware be the sole GPU selector.
    requires_vram_gb=None,
    allowed_hardware=["1x L40", "1x A100", "1x A100S"],
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
