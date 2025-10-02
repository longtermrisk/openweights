import hashlib
import io
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, BinaryIO, Dict

from openweights.client.decorators import supabase_retry
from supabase import Client


def validate_message(message):
    try:
        assert message["role"] in ["system", "user", "assistant"]
        if isinstance(message["content"], str):
            return True
        else:
            assert isinstance(message["content"], list)
            for part in message["content"]:
                assert isinstance(part["text"], str)
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
            for message in row["prompt"] + row["rejected"] + row["chosen"]:
                if not validate_message(message):
                    return False
        return True
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False


class Files:
    def __init__(self, supabase: Client, organization_id: str):
        self._supabase = supabase
        self._org_id = organization_id

    def _calculate_file_hash(self, stream: BinaryIO) -> str:
        """Calculate SHA-256 hash of file content"""
        sha256_hash = hashlib.sha256()
        for byte_block in iter(lambda: stream.read(4096), b""):
            sha256_hash.update(byte_block)
        # Add the org ID to the hash to ensure uniqueness
        sha256_hash.update(self._org_id.encode())
        try:
            stream.seek(0)
        except Exception:
            pass
        return f"file-{sha256_hash.hexdigest()[:12]}"

    def _get_storage_path(self, file_id: str) -> str:
        """Get the organization-specific storage path for a file"""
        try:
            result = self._supabase.rpc(
                "get_organization_storage_path",
                {"org_id": self._org_id, "filename": file_id},
            ).execute()
            return result.data
        except Exception as e:
            # Fallback if RPC fails
            return f"organizations/{self._org_id}/{file_id}"

    def upload(self, path, purpose) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return self.create(f, purpose)

    @supabase_retry()
    def create(self, file: BinaryIO, purpose: str) -> Dict[str, Any]:
        """Upload a file and create a database entry.
        Robust to retries by buffering the input stream into memory once
        and using fresh BytesIO streams for hashing, validation, and upload.
        """
        # Read all bytes once; support both real files and file-like objects
        try:
            # Ensure at start (some callers might pass a consumed stream)
            if hasattr(file, "seek"):
                try:
                    file.seek(0)
                except Exception:
                    pass
            data = file.read()
        finally:
            # Do not close the caller's file handle; just leave it as-is
            # (the caller used a context manager typically)
            pass

        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                "Files.create expects a binary file-like object returning bytes"
            )

        file_id = f"{purpose}:{self._calculate_file_hash(io.BytesIO(data))}"

        # If the file already exists, return the existing file
        try:
            existing_file = (
                self._supabase.table("files")
                .select("*")
                .eq("id", file_id)
                .single()
                .execute()
                .data
            )
            if existing_file:
                return existing_file
        except Exception:
            pass  # File doesn't exist yet, continue with creation

        # Validate file content using a fresh buffer
        if not self.validate(io.BytesIO(data), purpose):
            self.validate(io.BytesIO(data), purpose)
            raise ValueError("File content is not valid")

        file_size = len(data)
        filename = getattr(file, "name", "unknown")

        # Get organization-specific storage path
        storage_path = self._get_storage_path(file_id)

        # Store file in Supabase Storage with organization path
        # storage3's sync client expects a file path-like in some versions; write to a temp file for compatibility
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        try:
            self._supabase.storage.from_("files").upload(
                path=storage_path, file=tmp_path, file_options={"upsert": "true"}
            )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        # Create database entry
        data_row = {
            "id": file_id,
            "filename": filename,
            "purpose": purpose,
            "bytes": file_size,
        }

        result = self._supabase.table("files").insert(data_row).execute()

        return {
            "id": file_id,
            "object": "file",
            "bytes": file_size,
            "created_at": datetime.now().timestamp(),
            "filename": filename,
            "purpose": purpose,
        }

    @supabase_retry()
    def content(self, file_id: str) -> bytes:
        """Get file content"""
        storage_path = self._get_storage_path(file_id)
        return self._supabase.storage.from_("files").download(storage_path)

    def validate(self, file: BinaryIO, purpose: str) -> bool:
        """Validate file content. The passed stream will be consumed."""
        if purpose in ["conversations"]:
            content = file.read().decode("utf-8")
            return validate_messages(content)
        elif purpose == "preference":
            content = file.read().decode("utf-8")
            return validate_preference_dataset(content)
        else:
            return True

    @supabase_retry()
    def get_by_id(self, file_id: str) -> Dict[str, Any]:
        """Get file details by ID"""
        return (
            self._supabase.table("files")
            .select("*")
            .eq("id", file_id)
            .single()
            .execute()
            .data
        )
