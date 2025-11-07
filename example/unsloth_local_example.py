"""
Example of running jobs locally without uploading to the database.

This demonstrates how to use the `local=True` parameter to execute jobs
directly on your machine without:
- Uploading files to the database
- Creating job records in the database
- Running on RunPod workers

The training will run locally and push results directly to Hugging Face.
"""

from openweights import OpenWeights

# Initialize the client
client = OpenWeights()

# ============ Example 1: Fine-Tuning Job ============
print("=" * 80)
print("Example 1: Fine-Tuning with local=True")
print("=" * 80)

# Upload your training data
training_file = client.files.create("example/sft_dataset.jsonl")

# Create and run a fine-tuning job locally
job = client.fine_tuning.create(
    model="unsloth/Qwen3-4b",
    training_file=training_file["id"],
    loss="sft",
    epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=10,  # Short run for testing
    lr_scheduler_type="cosine",
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="no",
    warmup_ratio=0.1,
    local=True,  # Execute locally!
    requires_vram_gb=36,
)

print(f"Fine-tuning job executed locally: {job}")

# ============ Example 2: Custom Fine-Tuning Job ============
print("\n" + "=" * 80)
print("Example 2: Custom Fine-Tuning with local=True")
print("=" * 80)

job = client.unsloth_custom.create(
    model="unsloth/Qwen3-4b",
    training_file=training_file["id"],
    loss="sft",
    epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=10,
    local=True,  # Execute locally!
    requires_vram_gb=36,
)

print(f"Custom fine-tuning job executed locally: {job}")

# ============ Example 3: Inference Job ============
print("\n" + "=" * 80)
print("Example 3: Inference with local=True")
print("=" * 80)

# Upload input file for inference
input_file = client.files.create("tests/inference_dataset_with_prefill.jsonl")

# Create and run an inference job locally
job = client.inference.create(
    model="unsloth/Qwen3-4b",
    input_file_id=input_file["id"],
    local=True,  # Execute locally!
)

print(f"Inference job executed locally: {job}")

print("\n" + "=" * 80)
print("All jobs completed!")
print("=" * 80)
