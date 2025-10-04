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
docker build -t nielsrolf/ow-default:v0.6 .
docker push nielsrolf/ow-default:v0.6
```

Run an image locally: `docker run -e OW_DEV=true -ti nielsrolf/ow-default:v0.6 /bin/bash`
