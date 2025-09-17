# Logprob / MC test based on vllm
- implement in chat template (ow.chat.logprobs.create(messages=blockwise))
-> goto eval
-> 0-100 judge

# deploy checkpoint API


# Use `tag` as color in dashboard plots


# Other
- cli to run jobs: `ow run --cmd "axolotl train config.yaml" --mount . --gpu H100 --count 8`
- "report to ow" instead of wandb

# general
- merge chat.py, temporary_api.py
- add cpu instances
- customisable keep worker running for X mins
