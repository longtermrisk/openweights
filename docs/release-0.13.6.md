# OpenWeights 0.13.6

Jobs with alternative GPU spellings in `allowed_hardware` get workers again.

- `allowed_hardware` entries are normalized to the names workers register with:
  `1x A100 80GB` → `1x A100`, `1x A100 SXM 80GB` → `1x A100S`, full RunPod ids
  such as `NVIDIA H100 80GB HBM3` → `H100S`, and a missing count means `1x`.
  The client does this before computing the job id; unknown GPUs raise a
  `ValueError` listing the valid names.
- The cluster manager applies the same normalization to pending jobs that are
  already in the queue and writes the canonical names back, so workers match
  them. A pending job whose hardware cannot be recognized is marked `failed`
  with the reason in `outputs.error`.
- Before this, an unknown name made provisioning fail with a bare `KeyError`,
  which counted as a RunPod capacity failure and put the job's GPU types on an
  escalating cooldown of up to 6 hours, so the job stayed pending without a
  worker ever starting.

SDK and worker images are unchanged; `IMAGE_VERSION` stays v0.13.4.
