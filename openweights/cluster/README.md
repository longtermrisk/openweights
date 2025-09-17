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
## vllm + unsloth
docker build -t nielsrolf/ow-default .
docker push nielsrolf/ow-default
```

Run an image locally: `docker run -e OW_DEV=true -ti nielsrolf/ow-debug /bin/bash`
