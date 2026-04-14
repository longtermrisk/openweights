FROM unsloth/unsloth:2026.3.17-pt2.9.0-vllm-0.16.0-cu12.8-studio-release-v0.1.3-beta

USER root

WORKDIR /openweights

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install inspect_ai git+https://github.com/UKGovernmentBEIS/inspect_evals
RUN python3 -m pip install huggingface_hub[hf_transfer] hf_transfer supabase python-dotenv fire httpx>=0.24.0 runpod
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN git lfs install

COPY README.md .
COPY pyproject.toml .
COPY openweights openweights
COPY entrypoint.sh .
RUN python3 -m pip install -e .

# Upgrade transformers, unsloth, and TRL for compatibility
# --no-deps prevents torch/vllm downgrades from unsloth's dep resolver
# transformers pinned to 5.3.0: 5.2 has processing_utils bug, 5.4 breaks unsloth
RUN python3 -m pip install --upgrade --no-deps unsloth unsloth-zoo && \
    python3 -m pip install --upgrade --no-deps "transformers==5.3.0" && \
    python3 -m pip install --upgrade --no-deps trl

RUN echo 'export PATH=/opt/conda/bin:$PATH' >> /root/.bashrc && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /root/.profile

EXPOSE 22
EXPOSE 8000
EXPOSE 10101

ENTRYPOINT ["/openweights/entrypoint.sh"]
