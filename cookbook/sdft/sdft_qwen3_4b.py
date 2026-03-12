"""
Self-Distillation Fine-Tuning (SDFT) — minimal example.

Reference paper: https://arxiv.org/pdf/2601.19897

SDFT trains the student model (without demonstrations) to match the token-level
distribution of the teacher model (the same model conditioned on demonstrations
via EMA weights).  This can improve new-task accuracy while reducing catastrophic
forgetting compared to standard SFT.

Data format
-----------
Same JSONL file structure as SFT (purpose="conversations"), but each row may
include an optional "demonstration" field with an example response:

    {
        "messages": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "demonstration": "An example of a good response."   # optional
    }

When "demonstration" is absent the trainer automatically falls back to using
the last assistant message as the teacher's in-context demonstration.

Key hyperparameters
-------------------
- sdft_ema_alpha   : EMA rate for the teacher  (default 0.02; try 0.01–0.05)
- sdft_demo_template : Template with {demonstration} placeholder to inject the
                       demo into the teacher's context.  Uses a built-in default
                       when not specified.
"""

from openweights import OpenWeights

ow = OpenWeights()

# Upload the training file (conversations format, same as SFT)
training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]

job = ow.fine_tuning.create(
    # ---- model ---------------------------------------------------------------
    model="unsloth/Qwen3-4B",
    # ---- data ----------------------------------------------------------------
    training_file=training_file,
    # ---- algorithm -----------------------------------------------------------
    loss="sdft",
    # ---- SDFT-specific hyperparameters ---------------------------------------
    sdft_ema_alpha=0.02,            # EMA rate for the teacher model
    # sdft_demo_template=None,      # Use built-in template (can override here)
    # ---- standard training hyperparameters -----------------------------------
    epochs=1,
    learning_rate=1e-4,
    r=32,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
)

print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
