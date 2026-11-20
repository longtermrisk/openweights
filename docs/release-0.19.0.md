# OpenWeights 0.19.0

The credential broker for Hugging Face. Third of four steps towards a worker
environment without any organization credential (`TODO.md`).

## What changes

A worker pod runs the code of jobs, which can read its whole environment. Until now
that environment held the organization's real `HF_TOKEN`.

With a broker deployed, a worker gets instead:

    HF_ENDPOINT = <broker url>/hf
    HF_TOKEN    = hf_<the worker's own token>

Every Hub request of the worker goes to the broker (`ow broker`, `openweights/broker`).
The broker

1. checks the worker token with the OpenWeights database: it must be valid now and
   must be a worker's token (a personal API key is refused);
2. reads the organization's real `HF_TOKEN` from the secret store;
3. forwards the request to huggingface.co with the real token and returns the answer.

The real token never reaches the pod. The stand-in works only through the broker and
stops working within a minute of the worker's token being revoked (the worker was
terminated) or expiring.

**File contents do not pass through the broker.** The Hub answers downloads and
uploads of large files with redirects to signed storage URLs and with short-lived Xet
tokens, which the client uses directly. The broker rewrites only URLs that lead back
to the Hub itself (redirects, pagination links, LFS batch actions, the Xet refresh
route), so that the client does not send its stand-in to huggingface.co.

## When a worker is brokered

All three must hold, otherwise the worker gets the real token as before:

- the cluster manager has `OW_BROKER_URL` (the address workers reach the broker at);
- the organization's `HF_TOKEN` is in the secret store (the broker reads nothing else);
- the worker has a token of its own (0.18.0).

So a self-managed cluster, which has neither broker nor store, is unchanged.

## What this does not close, and what breaks

- While its worker lives, a job can *use* the organization's Hugging Face access
  through the broker. What it can no longer do is copy the token and use it later or
  elsewhere. The broker logs each request with the worker's id.
- **Worker images older than v0.15.2** contain code that calls `https://huggingface.co`
  directly instead of honouring `HF_ENDPOINT`. Those calls now carry the stand-in and
  fail with 401. Rebuild on a current image.
- Job code that hard-codes `https://huggingface.co` for an authenticated call fails the
  same way; `huggingface_hub`, `transformers`, `datasets` and `git` over the Hub's HTTP
  API honour `HF_ENDPOINT`. `git clone https://huggingface.co/...` with the token does
  not go through the broker.

## Running the broker

    OW_SECRET_STORE=gcp-secret-manager://<project id> \
    OW_BROKER_URL=https://<public address> \
    ow broker --port 8125

It must be reachable from the worker pods (the public internet) and needs read access
to the organization secrets in the store, and nothing else: no database credential,
no RunPod key. `GET /healthz` answers without authentication.

## Release order

1. Publish 0.19.0 and upgrade the cluster manager without `OW_BROKER_URL`. Nothing
   changes for workers.
2. Deploy the broker. Test it from a job: with the job's own `OPENWEIGHTS_API_KEY`
   (its worker token), set `HF_ENDPOINT` and `HF_TOKEN` as above and download and
   upload through it.
3. Set `OW_BROKER_URL` on the cluster manager. Workers started from then on are
   brokered; running workers keep the real token until they end.

No database migration and no worker image changes.

## Verification

- `tests/test_broker.py`: the proxy against a stand-in for the Hub (what is forwarded,
  what is rewritten, that the real token appears in no answer), the token check against
  a stand-in for the database, and which workers the manager points at the broker.
- Not verified before the release: the broker against the real Hub with the real
  client libraries (downloads through `hf_xet` / `hf_transfer`, uploads, gated models).
  That is step 2 of the release order.
