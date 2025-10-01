from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload("preferences.jsonl", purpose="conversations")["id"]
job = ow.fine_tuning.create(
    model="unsloth/llama3-8b-instruct",
    training_file=training_file,
    loss="dpo",
    epochs=1,
    learning_rate=1e-5,
    beta=0.1,  # Controls the strength of the preference optimization
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
