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

Built from `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` with Unsloth `2026.9.2`.
The SDK and worker dependencies are resolved together and `pip check` must pass.
MergeKit and LLM Blender are not used by the built-in jobs and are no longer installed
in this image; custom jobs that need them should use a separate compatible image.

```sh
docker buildx build \
  --platform linux/amd64 \
  -t nielsrolf/ow-unsloth:$VERSION \
  --push .
```

## 2. vLLM Worker Image

Built from `vllm/vllm-openai:v0.28.0`, preserving its matching PyTorch/CUDA stack,
with Transformers `5.16.1` and TRL `0.24.0`. The FastAPI constraint matches vLLM.
Both dependency consistency and GPU model smoke tests are required before promoting a tag.

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

## File sync (unison)

`ow ssh --sync` syncs with [unison](https://github.com/bcpierce00/unison), which has to be
present at both ends and refuses to talk to a different version of itself. The images therefore
pin a specific upstream static build rather than taking whatever apt offers:

| | |
|---|---|
| version in the images | 2.54.0 (`UNISON_VERSION` build arg) |
| what clients need | the same 2.54.0 — `brew install unison` currently gives exactly this |

To move it, bump `UNISON_VERSION` and `UNISON_SHA256` in `Dockerfile` and `Dockerfile.vllm`
together, then rebuild. Checksums are on the
[release page](https://github.com/bcpierce00/unison/releases). Keep the two images on the same
version, and prefer whatever Homebrew currently ships, since that is what most clients will
install.

## Candidate validation and promotion

Build and publish a distinct candidate tag before updating `openweights/images.py`.
Run Qwen3.8-27B SFT, DPO, and inference against the candidate; the scripts in
`experiments/training_comparison/` accept explicit image tags. Save resolved package
versions, image digests, job IDs, worker logs, and generated samples. A successful
Docker build alone does not establish GPU compatibility.

Promote a validated candidate without re-resolving dependencies: build the release tag
as an overlay that only refreshes the SDK source and package metadata, then confirm
`pip freeze` is unchanged.

```sh
cat > /tmp/Dockerfile.release <<'RELEASE'
FROM nielsrolf/ow-unsloth:${VERSION}-candidate
WORKDIR /openweights
COPY README.md pyproject.toml ./
COPY openweights openweights
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps -e . && /opt/venv/bin/python -m pip check
RELEASE
docker build -f /tmp/Dockerfile.release -t nielsrolf/ow-unsloth:$VERSION . && docker push nielsrolf/ow-unsloth:$VERSION
```

The cluster image uses the system `python3` instead of `/opt/venv/bin/python`. Records for
v0.12 are in `experiments/training_comparison/results/image-validation.json`.
