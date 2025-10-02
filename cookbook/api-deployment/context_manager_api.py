from openweights import OpenWeights

ow = OpenWeights()

model = "unsloth/Qwen3-8B"

# async with ow.api.deploy(model) also works
with ow.api.deploy(model):  # async with ow.api.deploy(model) also works
    # entering the context manager is equivalent to api = ow.api.deploy(model) ; api.up()
    completion = ow.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "is 9.11 > 9.9?"}]
    )
    print(
        completion.choices[0].message
    )  # when this context manager exits, it calls api.down()
