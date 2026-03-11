from openweights import OpenWeights

ow = OpenWeights()

job = ow.rl.create(
    model="Qwen/Qwen3-0.6B",
    envs=[{"id": "reverse-text"}],
    max_steps=20,
    batch_size=128,
    rollouts_per_example=16,
    max_tokens=128,
    learning_rate=3e-6,
    seq_len=2048,
    wandb_project="reverse-text",
    wandb_name="reverse-text-rl",
    # RL requires 2+ GPUs (1 inference, 1+ training). Defaults to multi-GPU hardware.
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
