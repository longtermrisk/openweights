"""Cancel all pending and in-progress jobs"""

from dotenv import load_dotenv

from openweights import OpenWeights

load_dotenv()
client = OpenWeights()


for job in client.jobs.list(limit=1000):
    if job.status in ["pending", "in_progress"]:
        job.cancel()
