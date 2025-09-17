from typing import Any, Dict, List, Optional

from openweights.client.decorators import supabase_retry


class Events:
    def __init__(self, supabase):
        self._supabase = supabase

    @supabase_retry()
    def list(self, job_id: Optional[str] = None, run_id: Optional[str] = None):
        """List events by job_id or run_id, sorted by created_at in ascending order"""
        if run_id:
            query = (
                self._supabase.table("events")
                .select("*")
                .eq("run_id", run_id)
                .order("created_at", desc=False)
            )
        elif job_id:
            # First get all runs for this job
            runs_result = (
                self._supabase.table("runs").select("id").eq("job_id", job_id).execute()
            )
            run_ids = [run["id"] for run in runs_result.data]
            # Then get all events for these runs
            query = (
                self._supabase.table("events")
                .select("*")
                .in_("run_id", run_ids)
                .order("created_at", desc=False)
            )
        else:
            raise ValueError("Either job_id or run_id must be provided")

        result = query.execute()
        return result.data

    def latest(
        self,
        fields: List[str],
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a list of events and return a dict with the latest value for each field"""
        events = self.list(job_id=job_id, run_id=run_id)
        latest_values = {}
        if fields == "*":
            latest_values = {}
            for event in events:
                for key, value in event["data"].items():
                    if value is not None:
                        latest_values[key] = value
        else:
            events = events[::-1]  # Reverse order to get latest events first
            while len(fields) > 0 and len(events) > 0:
                event = events.pop()
                for field in fields:
                    if field in event["data"]:
                        latest_values[field] = event["data"][field]
                        fields.remove(field)
        return latest_values
