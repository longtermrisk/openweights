from openweights import OpenWeights

ow = OpenWeights()

job = ow.inspect_ai.create(
    model="unsloth/Llama-3.2-1B",
    eval_name="inspect_evals/gpqa_diamond",
    options="--top-p 0.9",  # Can be any options that `inspect eval` accepts - we simply pass them on without validation
)

if job.status == "completed":
    job.download("output")
