from typing import Optional, BinaryIO, Dict, Any, List, Union
import os
import sys
from postgrest.exceptions import APIError
import hashlib
from datetime import datetime
from supabase import Client


class Run:
    def __init__(self, supabase: Client, job_id: Optional[str] = None, worker_id: Optional[str] = None):
        self._supabase = supabase
        self.id = os.getenv('OPENWEIGHTS_RUN_ID')
        
        if self.id:
            # Run ID exists, fetch the data
            try:
                result = self._supabase.table('runs').select('*').eq('id', self.id).single().execute()
            except APIError as e:
                if 'contains 0 rows' in str(e):
                    raise ValueError(f"Run with ID {self.id} not found")
                raise
            
            run_data = result.data
            if job_id and run_data['job_id'] != job_id:
                raise ValueError(f"Run {self.id} is associated with job {run_data['job_id']}, not {job_id}")
            
            if worker_id and run_data['worker_id'] != worker_id:
                # reassign run to self
                run_data['worker_id'] = worker_id
                result = self._supabase.table('runs').update(run_data).eq('id', self.id).execute()
                run_data = result.data[0]
            
            self._load_data(run_data)
        else:
            # Create new run
            data = {
                'status': 'in_progress'
            }
            
            if job_id:
                data['job_id'] = job_id
            else:
                # Create a new script job
                command = ' '.join(sys.argv)
                job_data = {
                    'id': f"sjob-{hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:12]}",
                    'type': 'script',
                    'script': command,
                    'status': 'in_progress',
                }
                job_result = self._supabase.table('jobs').insert(job_data).execute()
                data['job_id'] = job_result.data[0]['id']
            
            if worker_id:
                data['worker_id'] = worker_id

            # Get organization_id from job
            job = self._supabase.table('jobs').select('organization_id').eq('id', data['job_id']).single().execute()
            if not job.data:
                raise ValueError(f"Job {data['job_id']} not found")
            
            result = self._supabase.table('runs').insert(data).execute()
            self._load_data(result.data[0])

    def _load_data(self, data: Dict[str, Any]):
        self.id = data['id']
        self.job_id = data['job_id']
        self.worker_id = data.get('worker_id')
        self.status = data['status']
        self.log_file = data.get('log_file')
        self.created_at = data['created_at']

    @staticmethod
    def get(supabase: Client, run_id: int) -> 'Run':
        """Get a run by ID"""
        run = Run(supabase)
        run.id = run_id
        try:
            result = supabase.table('runs').select('*').eq('id', run_id).single().execute()
        except APIError as e:
            if 'contains 0 rows' in str(e):
                raise ValueError(f"Run with ID {run_id} not found")
            raise
        run._load_data(result.data)
        return run

    def update(self, status: Optional[str] = None, logfile: Optional[str] = None):
        """Update run status and/or logfile"""
        data = {}
        if status:
            data['status'] = status
        if logfile:
            data['log_file'] = logfile
        
        if data:
            result = self._supabase.table('runs').update(data).eq('id', self.id).execute()
            self._load_data(result.data[0])

    def log(self, event_data: Dict[str, Any], file: Optional[BinaryIO] = None):
        """Log an event for this run"""
        if file:
            file_id = self._supabase.files.create(file, purpose='event')['id']
        else:
            file_id = None
        data = {
            'run_id': self.id,
            'data': event_data,
            'file': file_id
        }
        self._supabase.table('events').insert(data).execute()
    
    @property
    def events(self) -> List[Dict[str, Any]]:
        """Get all events for this run"""
        result = self._supabase.table('events').select('*').eq('run_id', self.id).execute()
        return result.data


class Runs:
    def __init__(self, supabase: Client, organization_id: str):
        self._supabase = supabase
        self._org_id = organization_id

    def list(self, job_id: Optional[str] = None, worker_id: Optional[str] = None, limit: int = 10, status: Optional[str]=None) -> List[Dict[str, Any]]:
        """List runs by job_id or worker_id"""
        query = self._supabase.table('runs').select('*').limit(limit)
        if job_id:
            query = query.eq('job_id', job_id)
        if worker_id:
            query = query.eq('worker_id', worker_id)
        if status:
            query = query.eq('status', status)
        result = query.execute()
        return result.data