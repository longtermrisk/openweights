FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

USER root

WORKDIR /openweights

RUN apt-get update && \
    apt-get install -y --no-install-recommends git openssh-server python3-venv && \
    rm -rf /var/lib/apt/lists/*

# unison, for `ow ssh --sync`, which needs it present at both ends. Pinned to the upstream
# static build rather than installed from apt: unison refuses to sync between mismatched
# versions, and the distro package lags what clients install (Homebrew ships 2.54.0, Ubuntu
# 24.04 apt has 2.53.x, and those two will not talk to each other).
ARG UNISON_VERSION=2.54.0
ARG UNISON_SHA256=d279dff18682c909d3ddb0b280ab151229b4798b9399b1d227084da424337d24
ADD https://github.com/bcpierce00/unison/releases/download/v${UNISON_VERSION}/unison-${UNISON_VERSION}-ubuntu-22.04-x86_64-static.tar.gz /tmp/unison.tar.gz
RUN echo "${UNISON_SHA256}  /tmp/unison.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/unison.tar.gz -C /tmp --strip-components=1 \
        unison-${UNISON_VERSION}-ubuntu-22.04-x86_64-static/bin && \
    install -m 0755 /tmp/bin/unison /tmp/bin/unison-fsmonitor /usr/local/bin/ && \
    rm -rf /tmp/unison.tar.gz /tmp/bin && \
    unison -version

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY README.md .
COPY pyproject.toml .
COPY openweights openweights
COPY entrypoint.sh .
RUN python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        "unsloth[cu128-torch2100]==2026.9.2" && \
    /opt/venv/bin/python -m pip install --no-cache-dir --no-deps -e . && \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        PyJWT \
        cachier \
        diskcache \
        fastapi \
        fire \
        "httpx[http2]>=0.24.0" \
        huggingface-hub \
        openai \
        python-dotenv \
        runpod \
        scp \
        "supabase==2.15.3" \
        uvicorn \
        hf_transfer \
        "mergekit==0.1.4" \
        "llm-blender==0.0.2"

RUN /opt/venv/bin/python - <<'PY'
import importlib.metadata as metadata

for package in (
    "torch",
    "transformers",
    "huggingface-hub",
    "unsloth",
    "trl",
    "peft",
    "bitsandbytes",
):
    print(f"{package}=={metadata.version(package)}")
PY

RUN echo 'export PATH=/opt/venv/bin:$PATH' >> /root/.bashrc && \
    echo 'export PATH=/opt/venv/bin:$PATH' >> /root/.profile

EXPOSE 22
EXPOSE 8000
EXPOSE 10101

ENTRYPOINT ["/openweights/entrypoint.sh"]
