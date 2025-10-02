import json
import sys

from openweights import OpenWeights

# Get parameters from command line
params = json.loads(sys.argv[1])
a = params["a"]
b = params["b"]

# Calculate sum
result = a + b

# Log the result using the run API
ow = OpenWeights()
ow.run.log({"text": "we can log any dicts"})
ow.run.log({"text": "they can be fetched via ow.events(job_id=job.id)"})
ow.run.log(
    {"text": "you can then access the individual logged items via event['data']"}
)
ow.run.log({"result": result})

print(f"{a} + {b} = {result}")
