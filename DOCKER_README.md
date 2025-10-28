# Docker Images

OpenWeights uses two separate Docker images:

## 1. Worker Image (`nielsrolf/ow-default`)
The full worker image includes all GPU dependencies (PyTorch, Unsloth, vLLM) for running ML jobs.

### Building and pushing worker images

```sh
# Get the current version from the codebase
VERSION=$(python -c "from openweights.client.jobs import Jobs; print(Jobs.base_image.split(':')[-1])")

# Step 1: Build locally for ARM64 (on your Mac)
docker buildx build \
  --platform linux/arm64 \
  -t nielsrolf/ow-default:$VERSION \
  --load .

# Step 2: Build and push AMD64 to Docker Hub
docker buildx build \
  --platform linux/amd64 \
  -t nielsrolf/ow-default:$VERSION \
  --push .
```

### Running worker image locally
```sh
# Get the current version
VERSION=$(python -c "from openweights.client.jobs import Jobs; print(Jobs.base_image.split(':')[-1])")

docker run --rm --env-file .env -ti nielsrolf/ow-default:$VERSION /bin/bash
```

## 2. Cluster/Dashboard Image (`nielsrolf/ow-cluster`)
A lightweight image for running the cluster manager and/or dashboard backend. Does not include GPU dependencies.

### Building and pushing cluster images

**Important**: Build the frontend first before building the Docker image:

```sh
# Get the current version from the codebase
VERSION=$(python -c "from openweights.client.jobs import Jobs; print(Jobs.base_image.split(':')[-1])")

# Step 1: Build the frontend (run from repository root)
cd openweights/dashboard/frontend
npm install
npm run build
cd ../../..

# Step 2: Build locally for ARM64 (on your Mac)
docker buildx build \
  --platform linux/arm64 \
  -f Dockerfile.cluster \
  -t nielsrolf/ow-cluster:$VERSION \
  --load .

# Step 3: Build and push AMD64 to Docker Hub
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.cluster \
  -t nielsrolf/ow-cluster:$VERSION \
  --push .
```

Note: The frontend must be built before the Docker image because the build process copies the pre-built static files from `openweights/dashboard/backend/static/`. This makes the Docker build much faster by avoiding Node.js installation and npm build steps.

### Running cluster/dashboard image locally

The cluster image supports three modes via the `OW_CMD` environment variable:
- `cluster`: Run only the cluster manager
- `serve`: Run only the dashboard backend
- `both` (default): Run both cluster manager and dashboard backend

```sh
# Get the current version
VERSION=$(python -c "from openweights.client.jobs import Jobs; print(Jobs.base_image.split(':')[-1])")

# Run cluster manager only
docker run --rm --env-file .env -e OW_CMD=cluster -ti nielsrolf/ow-cluster:$VERSION

# Run dashboard backend only
docker run --rm --env-file .env -e OW_CMD=serve -p 8124:8124 -ti nielsrolf/ow-cluster:$VERSION

# Run both (default)
docker run --rm --env-file .env -p 8124:8124 -ti nielsrolf/ow-cluster:$VERSION

# Interactive shell
docker run --rm --env-file .env -ti nielsrolf/ow-cluster:$VERSION /bin/bash
```

### Dashboard environment variables

When running the dashboard backend, you can configure:
- `SITE_URL`: The public URL where the dashboard is accessible (default: `http://localhost:8124`)
- `API_EXTERNAL_URL`: The external API URL (default: `http://localhost:8124`)
- `ADDITIONAL_REDIRECT_URLS`: Additional allowed redirect URLs for OAuth
