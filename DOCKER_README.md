# Docker Images

OpenWeights now uses three images:

- `nielsrolf/ow-unsloth:$VERSION` for Unsloth-based fine-tuning jobs
- `nielsrolf/ow-vllm:$VERSION` for vLLM inference and the transformers/TRL weighted-SFT path
- `nielsrolf/ow-cluster:$VERSION` for the cluster manager and dashboard backend

## Version

```sh
VERSION=$(python -c "from openweights.images import IMAGE_VERSION; print(IMAGE_VERSION)")
```

## 1. Unsloth Worker Image

Built from `unsloth/unsloth:latest` and kept close to upstream Unsloth.

```sh
docker buildx build \
  --platform linux/amd64 \
  -t nielsrolf/ow-unsloth:$VERSION \
  --push .
```

## 2. vLLM Worker Image

Built from `vllm/vllm-openai:v0.19.1` for a clean `vllm` + `transformers 5.x` stack.

```sh
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.vllm \
  -t nielsrolf/ow-vllm:$VERSION \
  --push .
```

## 3. Cluster/Dashboard Image

Build the frontend first:

```sh
cd openweights/dashboard/frontend
npm install
npm run build
cd ../../..
```

Then build and push the cluster image:

```sh
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.cluster \
  -t nielsrolf/ow-cluster:$VERSION \
  --push .
```

## Local Shells

```sh
docker run --rm --env-file .env -ti nielsrolf/ow-unsloth:$VERSION /bin/bash
docker run --rm --env-file .env -ti nielsrolf/ow-vllm:$VERSION /bin/bash
docker run --rm --env-file .env -ti nielsrolf/ow-cluster:$VERSION /bin/bash
```
