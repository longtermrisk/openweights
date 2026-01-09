# Prime-RL Jobs

This package provides a custom OpenWeights job for running Prime-RL entrypoints and optionally mounting
local verifiers environments into the job workspace.

## Docker image

Build an image that includes prime-rl and deepspeed, starting from the standard OpenWeights worker image:

```sh
# From repo root
TAG=v0.1

docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.prime-rl \
  -t nielsrolf/ow-prime-rl:$TAG \
  --push .
```

## Toy run (Prime-RL RL entrypoint)

```python
from openweights import OpenWeights

ow = OpenWeights()

job = ow.prime_rl.create(
    command="rl",
    config_path="cookbook/prime_rl/toy_rl.toml",
    env_path="cookbook/prime_rl/toy_env.py",
    push_to_hf=True,
    allowed_hardware=["2x L40", "2x H100S"],
)
```

## Custom environment runs

Pass a directory or file that defines a `load_environment` function. The job will mount the path
and add it to `PYTHONPATH` automatically.

```python
job = ow.prime_rl.create(
    command="rl",
    config_path="/path/to/your_rl.toml",
    env_path="/path/to/your_env",
)
```

Set `env_target` or `pythonpath` when you need a different import layout:

```python
job = ow.prime_rl.create(
    command="rl",
    config_path="/path/to/your_rl.toml",
    env_path="/path/to/your_env",
    env_target="environments",
    pythonpath="/opt/extra_envs",
)
```
