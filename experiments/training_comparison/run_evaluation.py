"""Submit a shared-mask OW evaluation for a base model or trained HF adapter."""

import argparse
import json
import shlex
from pathlib import Path

from common import ROOT, write_json
from dotenv import load_dotenv
from pydantic import BaseModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--image", default=None)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    load_dotenv(ROOT.parent.parent / ".env")
    from openweights import Jobs, OpenWeights, register

    class Params(BaseModel):
        model: str
        audit_only: bool = False

    @register("comparison_eval")
    class Evaluation(Jobs):
        params = Params
        mount = {
            str(path): path.name
            for path in (ROOT.parent.parent / "openweights/jobs/unsloth").glob("*.py")
        }
        mount.update(
            {str(ROOT / name): name for name in ["common.py", "worker_evaluation.py"]}
        )
        requires_vram_gb = 80

        def get_entrypoint(self, params):
            return "UNSLOTH_RETURN_LOGITS=1 python worker_evaluation.py " + shlex.quote(
                json.dumps(params.model_dump())
            )

    ow = OpenWeights()
    if args.image:
        ow.comparison_eval.base_image = args.image
    job = ow.comparison_eval.create(
        model=args.model, audit_only=args.audit_only, allowed_hardware=["1x H200"]
    )
    write_json(
        ROOT / "results" / ("eval-" + job.id + ".json"),
        dict(job_id=job.id, model=args.model, image=job.docker_image),
    )
    print(job.id)


if __name__ == "__main__":
    main()
