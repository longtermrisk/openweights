# Managing workers

Start a worker on the current machine:
```sh
python openweights/worker/main.py
```

Start a single runpod instance with a worker:
```sh
python openweights/cluster/start_runpod.py
```

Starting a cluster
```sh
python openweights/cluster/supervisor.py
```

# Updating worker images

```sh
# Step 1: Build locally for ARM64 (on your Mac)
docker buildx build \
  --platform linux/arm64 \
  -t nielsrolf/ow-default:v0.7 \
  --load .

# Step 2: Build and push AMD64 to Docker Hub
docker buildx build \
  --platform linux/amd64 \
  -t nielsrolf/ow-default:v0.7 \
  --push .

```

Run an image locally: `docker run -rm -e OW_DEV=true --env-file .env -ti nielsrolf/ow-default:v0.7 /bin/bash`
