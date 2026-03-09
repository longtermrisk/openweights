FROM unsloth/unsloth:stable

USER root

WORKDIR /openweights

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install inspect_ai git+https://github.com/UKGovernmentBEIS/inspect_evals
RUN python3 -m pip install "vllm>=0.17.0" huggingface_hub[hf_transfer] hf_transfer supabase python-dotenv fire httpx>=0.24.0 runpod
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN git lfs install

COPY README.md .
COPY pyproject.toml .
COPY openweights openweights
COPY entrypoint.sh .
RUN python3 -m pip install -e ".[worker]"

# Upgrade transformers, unsloth, and unsloth-zoo LAST to avoid being downgraded by earlier installs
RUN python3 -m pip install --upgrade --no-deps "transformers>=5.0" && \
    python3 -m pip install --upgrade unsloth unsloth-zoo

RUN echo 'export PATH=/opt/conda/bin:$PATH' >> /root/.bashrc && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /root/.profile

EXPOSE 22
EXPOSE 8000
EXPOSE 10101

ENTRYPOINT ["/openweights/entrypoint.sh"]
