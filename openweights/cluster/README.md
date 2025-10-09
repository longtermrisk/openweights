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
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t nielsrolf/ow-default:v0.7 \
  --push .
```

Run an image locally: `docker run -rm -e OW_DEV=true --env-file .env -ti nielsrolf/ow-default:v0.7 /bin/bash`
