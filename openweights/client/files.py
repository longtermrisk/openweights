from typing import Optional, BinaryIO, Dict, Any, List, Union
import os
import hashlib
from datetime import datetime
from supabase import Client

from openweights.validate import validate_messages, validate_preference_dataset


class Files:
    def __init__(self, supabase: Client, organization_id: str):
        self._supabase = supabase
        self._org_id = organization_id

    def _calculate_file_hash(self, file: BinaryIO) -> str:
        """Calculate SHA-256 hash of file content"""
        sha256_hash = hashlib.sha256()
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)
        # Add the org ID to the hash to ensure uniqueness
        sha256_hash.update(self._org_id.encode())
        file.seek(0)  # Reset file pointer
        return f"file-{sha256_hash.hexdigest()[:12]}"

    def _get_storage_path(self, file_id: str) -> str:
        """Get the organization-specific storage path for a file"""
        try:
            result = self._supabase.rpc(
                'get_organization_storage_path',
                {'org_id': self._org_id, 'filename': file_id}
            ).execute()
            return result.data
        except Exception as e:
            # Fallback if RPC fails
            return f"organizations/{self._org_id}/{file_id}"

    def create(self, file: BinaryIO, purpose: str) -> Dict[str, Any]:
        """Upload a file and create a database entry"""
        file_id = f"{purpose}:{self._calculate_file_hash(file)}"

        # If the file already exists, return the existing file
        try:
            existing_file = self._supabase.table('files').select('*').eq('id', file_id).single().execute().data
            if existing_file:
                return existing_file
        except:
            pass  # File doesn't exist yet, continue with creation

        # Validate file content
        if not self.validate(file, purpose):
            raise ValueError("File content is not valid")

        file_size = os.fstat(file.fileno()).st_size
        filename = getattr(file, 'name', 'unknown')

        # Get organization-specific storage path
        storage_path = self._get_storage_path(file_id)

        # Store file in Supabase Storage with organization path
        self._supabase.storage.from_('files').upload(
            path=storage_path,
            file=file
        )

        # Create database entry
        data = {
            'id': file_id,
            'filename': filename,
            'purpose': purpose,
            'bytes': file_size
        }
        
        result = self._supabase.table('files').insert(data).execute()
        
        return {
            'id': file_id,
            'object': 'file',
            'bytes': file_size,
            'created_at': datetime.now().timestamp(),
            'filename': filename,
            'purpose': purpose,
        }

    def content(self, file_id: str) -> bytes:
        """Get file content"""
        storage_path = self._get_storage_path(file_id)
        return self._supabase.storage.from_('files').download(storage_path)
    
    def validate(self, file: BinaryIO, purpose: str) -> bool:
        """Validate file content"""
        if purpose in ['conversations']:
            content = file.read().decode('utf-8')
            return validate_messages(content)
        elif purpose == 'preference':
            content = file.read().decode('utf-8')
            return validate_preference_dataset(content)
        else:
            return True