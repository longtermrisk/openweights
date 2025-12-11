import gzip
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

logger = logging.getLogger(__name__)

# Minimum size to compress (skip tiny files where gzip header overhead isn't worth it)
COMPRESSION_MIN_BYTES = 1024  # 1 KB
# Compression level (1-9, higher = better compression but slower)
GZIP_COMPRESSION_LEVEL = 6


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
    def __init__(self, ow_instance: "OpenWeights", organization_id: str):
        self._ow = ow_instance
        self._org_id = organization_id

    def _calculate_file_hash(self, stream: BinaryIO) -> str:
        """Calculate SHA-256 hash of file content."""
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
        """Get the organization-specific storage path for a file."""
        try:
            result = self._ow._supabase.rpc(
                "get_organization_storage_path",
                {"org_id": self._org_id, "filename": file_id},
            ).execute()
            return result.data
        except Exception:
            # Fallback if RPC fails
            return f"organizations/{self._org_id}/{file_id}"

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip.

        Args:
            data: Raw bytes to compress.

        Returns:
            Gzip-compressed bytes.
        """
        return gzip.compress(data, compresslevel=GZIP_COMPRESSION_LEVEL)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data.

        Args:
            data: Gzip-compressed bytes.

        Returns:
            Decompressed bytes.
        """
        return gzip.decompress(data)

    def _should_compress(self, data: bytes, purpose: str) -> bool:
        """Determine if data should be compressed before upload.

        Args:
            data: Raw bytes to potentially compress.
            purpose: File purpose (e.g., 'result', 'conversations').

        Returns:
            True if the data should be compressed.
        """
        # Skip tiny files where gzip header overhead isn't worth it
        if len(data) < COMPRESSION_MIN_BYTES:
            return False
        # Always compress result files (JSONL compresses extremely well)
        return purpose == "result"

    def upload(self, path: str, purpose: str) -> Dict[str, Any]:
        """Upload a file from a path.

        Args:
            path: Path to the file to upload.
            purpose: Purpose of the file.

        Returns:
            Dictionary containing file metadata.
        """
        with open(path, "rb") as f:
            return self.create(f, purpose)

    @supabase_retry(max_time=600, max_tries=5)
    def create(self, file: BinaryIO, purpose: str) -> Dict[str, Any]:
        """Upload a file and create a database entry.

        Robust to retries by buffering the input stream into memory once
        and using fresh BytesIO streams for hashing, validation, and upload.
        Large result files are automatically compressed with gzip before upload.

        Args:
            file: Binary file-like object to upload.
            purpose: Purpose of the file (e.g., 'result', 'conversations').

        Returns:
            Dictionary containing file metadata.
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

        # Calculate file ID from original (uncompressed) data for consistency
        file_id = f"{purpose}:{self._calculate_file_hash(io.BytesIO(data))}"
        filename = getattr(file, "name", "unknown")

        # If the file already exists, return the existing file
        try:
            existing_file = (
                self._ow._supabase.table("files")
                .select("*")
                .eq("id", file_id)
                .single()
                .execute()
                .data
            )
            if existing_file:
                logger.info(f"File already exists: {file_id} (purpose: {purpose})")
                return existing_file
        except Exception:
            pass  # File doesn't exist yet, continue with creation

        original_size = len(data)
        logger.info(
            f"Uploading file: {filename} (purpose: {purpose}, size: {original_size} bytes)"
        )

        # Validate file content using a fresh buffer
        if not self.validate(io.BytesIO(data), purpose):
            self.validate(io.BytesIO(data), purpose)
            raise ValueError("File content is not valid")

        # Compress large result files
        is_compressed = False
        upload_data = data
        if self._should_compress(data, purpose):
            logger.info(
                f"Compressing file {filename} ({original_size / 1024 / 1024:.1f} MB)..."
            )
            upload_data = self._compress_data(data)
            compressed_size = len(upload_data)
            compression_ratio = original_size / compressed_size
            is_compressed = True
            logger.info(
                f"Compressed {original_size / 1024 / 1024:.1f} MB -> "
                f"{compressed_size / 1024 / 1024:.1f} MB "
                f"({compression_ratio:.1f}x reduction)"
            )

        # Get organization-specific storage path
        storage_path = self._get_storage_path(file_id)
        if is_compressed:
            storage_path = storage_path + ".gz"

        # Store file in Supabase Storage with organization path
        # storage3's sync client expects a file path-like in some versions;
        # write to a temp file for compatibility
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(upload_data)
            tmp.flush()
            tmp_path = tmp.name
        try:
            self._ow._supabase.storage.from_("files").upload(
                path=storage_path, file=tmp_path, file_options={"upsert": "true"}
            )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # Create database entry (store original size, not compressed)
        data_row = {
            "id": file_id,
            "filename": filename,
            "purpose": purpose,
            "bytes": original_size,
            "organization_id": self._org_id,
        }

        self._ow._supabase.table("files").insert(data_row).execute()
        logger.info(f"File uploaded successfully: {file_id}")

        return {
            "id": file_id,
            "object": "file",
            "bytes": original_size,
            "created_at": datetime.now().timestamp(),
            "filename": filename,
            "purpose": purpose,
        }

    @supabase_retry(max_time=600, max_tries=5)
    def content(self, file_id: str) -> bytes:
        """Get file content, automatically decompressing if needed.

        Tries to download from compressed path (.gz) first, then falls back
        to uncompressed path. No database metadata is required.

        Args:
            file_id: The ID of the file to download.

        Returns:
            The file content as bytes (decompressed if it was stored compressed).
        """
        logger.info(f"Downloading file: {file_id}")
        storage_path = self._get_storage_path(file_id)

        # Try compressed path first (most large result files will be compressed)
        try:
            content = self._ow._supabase.storage.from_("files").download(
                storage_path + ".gz"
            )
            logger.info(f"Downloaded compressed file: {file_id} ({len(content)} bytes)")
            content = self._decompress_data(content)
            logger.info(f"Decompressed to {len(content)} bytes")
            return content
        except Exception:
            # Compressed file doesn't exist, try uncompressed
            pass

        # Fall back to uncompressed path
        content = self._ow._supabase.storage.from_("files").download(storage_path)
        logger.info(f"File downloaded: {file_id} ({len(content)} bytes)")
        return content

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
            self._ow._supabase.table("files")
            .select("*")
            .eq("id", file_id)
            .single()
            .execute()
            .data
        )
