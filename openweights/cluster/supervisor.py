"""
Supervisor process that manages organization-specific cluster managers.
"""
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManagerSupervisor:
    def __init__(self):
        # The supervisor needs admin access to list all organizations and their secrets
        if 'SUPABASE_SERVICE_ROLE_KEY' not in os.environ:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is required for the supervisor")
        
        self.supabase_url = os.environ['SUPABASE_URL']
        self.supabase = create_client(
            self.supabase_url,
            os.environ['SUPABASE_SERVICE_ROLE_KEY'],
            ClientOptions(
                schema="public",
                headers={},  # No auth header needed with service role key
                auto_refresh_token=False,
                persist_session=False
            )
        )
        self.processes: Dict[str, subprocess.Popen] = {}
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def get_org_secrets(self, org_id: str) -> Dict[str, str]:
        """Get organization secrets from the database."""
        result = self.supabase.table('organization_secrets')\
            .select('name, value')\
            .eq('organization_id', org_id)\
            .execute()
        
        return {secret['name']: secret['value'] for secret in result.data}

    def validate_org_secrets(self, secrets: Dict[str, str]) -> bool:
        """Validate that all required secrets are present."""
        required_secrets = [
            'HF_TOKEN',
            'HF_ORG',
            'HF_USER',
            'OPENWEIGHTS_API_KEY',
            'RUNPOD_API_KEY'
        ]
        return all(secret in secrets for secret in required_secrets)

    def start_org_manager(self, org_id: str, secrets: Dict[str, str]) -> subprocess.Popen:
        """Start a new manager process for an organization."""
        env = os.environ.copy()
        # Remove admin credentials from worker environment
        env.pop('SUPABASE_SERVICE_ROLE_KEY', None)
        env.pop('SUPABASE_ADMIN_KEY', None)  # Just in case
        
        # Set organization-specific environment
        env.update({
            'ORGANIZATION_ID': org_id,
            'RUNPOD_API_KEY': secrets['RUNPOD_API_KEY'],
            'HF_TOKEN': secrets['HF_TOKEN'],
            'HF_ORG': secrets['HF_ORG'],
            'HF_USER': secrets['HF_USER'],
            'OPENWEIGHTS_API_KEY': secrets['OPENWEIGHTS_API_KEY']
        })

        # Get the path to org_manager.py relative to this file
        manager_path = Path(__file__).parent / 'org_manager.py'
        
        # Create log directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Open log files
        stdout_path = log_dir / f'org_{org_id}_stdout.log'
        stderr_path = log_dir / f'org_{org_id}_stderr.log'
        stdout = open(stdout_path, 'a')
        stderr = open(stderr_path, 'a')
        
        process = subprocess.Popen(
            [sys.executable, str(manager_path)],
            env=env,
            stdout=stdout,
            stderr=stderr
        )
        
        logger.info(f"Started manager for organization {org_id} (PID: {process.pid})")
        return process

    def check_process(self, org_id: str, process: subprocess.Popen) -> bool:
        """Check if a process is still running and handle any issues."""
        if process.poll() is not None:
            returncode = process.poll()
            logger.error(f"Manager for organization {org_id} exited with code {returncode}")
            return False
        return True

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals by cleaning up all processes."""
        logger.info("Received shutdown signal, terminating all managers...")
        for org_id, process in self.processes.items():
            logger.info(f"Terminating manager for organization {org_id}")
            process.terminate()
            try:
                process.wait(timeout=5)  # Give each process 5 seconds to clean up
            except subprocess.TimeoutExpired:
                logger.warning(f"Manager for organization {org_id} didn't terminate gracefully, killing...")
                process.kill()
        sys.exit(0)

    def supervise(self):
        """Main supervision loop."""
        while True:
            try:
                # Get all organizations
                orgs = self.supabase.table('organizations').select('*').execute()
                active_org_ids = {org['id'] for org in orgs.data}
                
                # Start new managers and check existing ones
                for org in orgs.data:
                    org_id = org['id']
                    
                    # Check if we need to start a new manager
                    if org_id not in self.processes or not self.check_process(org_id, self.processes[org_id]):
                        try:
                            secrets = self.get_org_secrets(org_id)
                            print('org_id:', org_id, 'secrets:', secrets)
                            if self.validate_org_secrets(secrets):
                                self.processes[org_id] = self.start_org_manager(org_id, secrets)
                            else:
                                logger.warning(f"Organization {org_id} missing required secrets")
                        except Exception as e:
                            logger.error(f"Failed to start manager for organization {org_id}: {e}")
                
                # Clean up managers for organizations that no longer exist
                for org_id in list(self.processes.keys()):
                    if org_id not in active_org_ids:
                        process = self.processes.pop(org_id)
                        logger.info(f"Terminating manager for organization {org_id} (no longer exists)")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Had to force kill manager for organization {org_id}")
                            process.kill()
                
            except Exception as e:
                logger.error(f"Error in supervisor loop: {e}")
            
            time.sleep(60)  # Check every minute

def main():
    supervisor = ManagerSupervisor()
    supervisor.supervise()

if __name__ == '__main__':
    main()