"""The credential broker: workers use an organization's secrets without holding them.

A worker pod runs the code of jobs, which can read everything in its environment. So
the pod gets no organization credential. What it holds is its own worker token (see
``create_worker_token`` in the database), and for Hugging Face:

    HF_ENDPOINT = <broker url>/hf
    HF_TOKEN    = hf_<worker token>

Every Hub request of the worker therefore arrives here. The broker checks the worker
token with the database, reads the organization's real ``HF_TOKEN`` from the secret
store, puts it into the request and forwards it to huggingface.co. The real token
never reaches the pod, and the stand-in stops working the moment the worker's token
is revoked (the worker was terminated) or expires.

File contents do not pass through here. The Hub answers downloads and uploads of
large files with redirects to signed storage URLs and with short-lived Xet tokens,
which the client uses directly; those are bound to one repository and expire.

The same goes for the organization's LiteLLM proxy: the pod gets
``LITELLM_BASE_URL = <broker url>/litellm`` and its worker token as ``LITELLM_API_KEY``;
the broker puts in the organization's real key and forwards to the address the
organization configured (``LITELLM_BASE_URL`` in its settings, which the cluster
manager writes onto the worker token).

What the broker does not do: stop a job from *using* the organization's Hugging Face
access while its worker lives. It stops the token from being copied out and used
later, and it logs which worker made which request.

The same design is in r-docs (lib/credential-broker), where it was tried first.
"""

# This module stays free of web-framework imports: the cluster manager imports it.

# Where the broker serves the Hugging Face Hub: HF_ENDPOINT = <broker url> + HF_PREFIX.
HF_PREFIX = "/hf"
# Where it serves the organization's LiteLLM proxy:
# LITELLM_BASE_URL = <broker url> + LITELLM_PREFIX, with the worker token as the key.
LITELLM_PREFIX = "/litellm"
# huggingface_hub and the Hub's own tooling expect tokens to look like this.
HF_TOKEN_PREFIX = "hf_"


def hf_stand_in_token(worker_token: str) -> str:
    """What a worker gets as HF_TOKEN: its own worker token, recognizable to the broker."""
    return HF_TOKEN_PREFIX + worker_token
