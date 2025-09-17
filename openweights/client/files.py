from typing import Optional, BinaryIO, Dict, Any, List, Union
import os
import hashlib
from datetime import datetime
from supabase import Client
import backoff
import json
import logging
from openweights.client.decorators import supabase_retry

def validate_message(message):
    try:
        assert message['role'] in ['system', 'user', 'assistant']
        if isinstance(message['content'], str):
            return True
        else:
            assert isinstance(message['content'], list)
            for part in message['content']:
                assert isinstance(part['text'], str)
            return True
    except (KeyError, AssertionError):
        return False
    
def validate_text_only(text):
    try:
        assert isinstance(text, str)
        return True
    except (KeyError, AssertionError):
        return False

def validate_messages(content):
    try:
        lines = content.strip().split("\n")
        for line in lines:
            row = json.loads(line)
            if "messages" in row:
                assert "text" not in row
                for message in row["messages"]:
                    if not validate_message(message):
                        logging.error(
                            f"Invalid message in conversations file: {message}"
                        )
                        return False
            elif "text" in row:
                if not validate_text_only(row["text"]):
                    logging.error(f"Invalid text in conversations file: {row['text']}")
                    return False
            else:
                logging.error(
                    f"Invalid row in conversations file (no 'messages' or 'text' key): {row}"
                )
                return False
        return True
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False

def validate_preference_dataset(content):
    try:
        lines = content.strip().split("\n")
        for line in lines:
            row = json.loads(line)
            for message in row['prompt'] + row['rejected'] + row['chosen']:
                if not validate_message(message):
                    return False
        return True
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False


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

    @supabase_retry()
    def create(self, file: BinaryIO, purpose: str) -> Dict[str, Any]:
        """Upload a file and create a database entry"""
        file.seek(0)
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
            file=file,
            file_options={"upsert": "true"}
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

    @supabase_retry()
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
        
    @supabase_retry()
    def get_by_id(self, file_id: str) -> Dict[str, Any]:
        """Get file details by ID"""
        return self._supabase.table('files').select('*').eq('id', file_id).single().execute().data
