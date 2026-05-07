FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

USER root

WORKDIR /openweights

RUN apt-get update && \
    apt-get install -y --no-install-recommends git openssh-server python3-venv && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY README.md .
COPY pyproject.toml .
COPY openweights openweights
COPY entrypoint.sh .
RUN python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        "unsloth[cu128-torch2100]==2026.4.6" && \
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
        supabase \
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
