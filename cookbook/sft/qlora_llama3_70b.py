from openweights import OpenWeights

ow = OpenWeights()

training_file = ow.files.upload(path="data/train.jsonl", purpose="conversations")["id"]
test_file = ow.files.upload(path="data/test.jsonl", purpose="conversations")["id"]

job = ow.fine_tuning.create(
    model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    training_file=training_file,
    test_file=test_file,
    load_in_4bit=True,
    max_seq_length=2047,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,  # lora rank
    save_steps=10,  # save a checkpoint every 10 steps
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    allowed_hardware=["1x H200"],
    merge_before_push=False,  # Push only the lora adapter
    logp_callback_datasets={  # Track logprobs of tokens in the testfile to ensure that training works
        "in-distribution": test_file
    },
)
print(job)
print(
    f"The model will be pushed to: {job.params['validated_params']['finetuned_model_id']}"
)
