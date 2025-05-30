# Custom jobs

A custom job lets you run a script that you would normally run on one GPU as a job.

Example:
```python
from openweights import OpenWeights, register, Jobs
ow = OpenWeights()

@register('some_name_for_my_custom_job')
class MyCustomJob(Jobs):
    mount = {
        'local/path/to/script.py': 'script.py',
        'local/path/to/dir/': 'dirname/'
    }
    params: Type[BaseModel] = MyParams  # Your Pydantic model for params
    base_image: str = 'nielsrolf/ow-inference'
    requires_vram_gb: int = 24

    def get_entrypoint(self, validated_params: BaseModel) -> str:
        # Get the entrypoint command for the job.
        return f'python script.py {json.dumps(validated_params.model_dump())}'
```

A custom job consists of:
- mounted source files - the code to run a job
- a pydantic model for parameter validation
- the default `requires_vram_gb` - this can be overwritten by passing `ow.some_name_for_my_custom_job.create(requires_vram_gb=60)`
- the docker image to use for the worker - you can build your own images and use them, but the images need to start an openweights worker (see the Dockerfiles in the repo root as reference)
- an entrypoint
- you can additionally override the `.create` method to create defaults

It's good to understand what code runs where:
- the initialization of the cusotm job runs on your laptop. It then uploads the mounted source files to openweights
- a worker then downloads the mounted source files into the cwd (a temporary dir) and runs the command returned by `get_entrypoint()`. That means that the `entrypoint` is responsible for passing the parameters to the script.

For a full example, checkout [start_custom_job.py](start_custom_job.py).