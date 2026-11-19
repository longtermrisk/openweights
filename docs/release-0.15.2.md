# OpenWeights 0.15.2

The "does this model exist" check that runs when an inference or API job is created
(`openweights.client.utils.model_exists`) now honours `HF_ENDPOINT`, like `huggingface_hub`
does. It used to call `https://huggingface.co` directly.

This matters where Hub traffic goes through a proxy: a mirror, or a credential broker that
keeps the real token outside the environment and gives the process a stand-in. There the
hard-coded address sent the stand-in to the real Hub, got a 401, and the job was rejected
with "model does not exist".

- Without `HF_ENDPOINT` nothing changes.
- SDK only. Worker images are unchanged; `IMAGE_VERSION` stays v0.15.0.
