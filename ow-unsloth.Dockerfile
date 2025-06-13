FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel


# Install SSH
RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd

RUN apt-get install -y git

# Create a directory for SSH keys
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install --no-cache-dir huggingface_hub supabase python-dotenv fire "httpx>=0.24.0"
# Using hf_transfer helps for the upload of the ft model, even if it may increase loading/setup times or/and create a hole in the logs during loading/setup
RUN python3 -m pip install --no-cache-dir huggingface_hub[hf_transfer] hf_transfer supabase python-dotenv fire "httpx>=0.24.0"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN python3 -m pip install --no-cache-dir "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
RUN python3 -m pip install --no-cache-dir inspect_ai git+https://github.com/UKGovernmentBEIS/inspect_evals

COPY README.md .
COPY pyproject.toml .
COPY openweights ./openweights
COPY entrypoint.sh .
RUN python3 -m pip install -e .


EXPOSE 22
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]