FROM vllm/vllm-openai:latest

WORKDIR /my_app

# Install SSH
RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd

# Create a directory for SSH keys
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Update SSH configuration
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install huggingface_hub[hf_transfer] hf_transfer supabase python-dotenv torch fire httpx>=0.24.0 runpod bitsandbytes
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python3 -m pip install inspect_ai git+https://github.com/UKGovernmentBEIS/inspect_evals
RUN python3 -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

RUN apt-get install git-lfs -y
RUN git lfs install

COPY README.md .
COPY pyproject.toml .
COPY openweights openweights
COPY entrypoint.sh .
RUN python3 -m pip install -e .

# Create a symbolic link from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

EXPOSE 22
EXPOSE 8000
EXPOSE 10101

ENTRYPOINT ["/my_app/entrypoint.sh"]
