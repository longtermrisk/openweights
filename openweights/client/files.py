import gzip
import hashlib
import io
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from openweights.client.decorators import supabase_retry
from supabase import Client

logger = logging.getLogger(__name__)

# Minimum size to compress (skip tiny files where gzip header overhead isn't worth it)
COMPRESSION_MIN_BYTES = 1024  # 1 KB
# Compression level (1-9, higher = better compression but slower)
GZIP_COMPRESSION_LEVEL = 6
# Maximum chunk size for large files (100 MB)
CHUNK_SIZE_BYTES = 100 * 1024 * 1024
# Chunk naming pattern: {base_id}.chunk.{n}of{k} (e.g., file-abc123.gz.chunk.0of3)
CHUNK_PATTERN = ".chunk."


def validate_message(message: Dict[str, Any]) -> bool:
    """Validate a single message in a conversation.

    Args:
        message: A message dictionary with 'role' and 'content' keys.

    Returns:
        True if the message is valid, False otherwise.
    """
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


def split_into_chunks(data: bytes, chunk_size: int = CHUNK_SIZE_BYTES) -> List[bytes]:
    """Split data into chunks of specified size.

    Args:
        data: The bytes to split.
        chunk_size: Maximum size of each chunk in bytes.

    Returns:
        List of byte chunks.
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i : i + chunk_size])
    return chunks


def join_chunks(chunks: List[bytes]) -> bytes:
    """Join chunks back into a single bytes object.

    Args:
        chunks: List of byte chunks in order.

    Returns:
        Concatenated bytes.
    """
    return b"".join(chunks)


def is_chunk_id(file_id: str) -> bool:
    """Check if a file ID represents a chunk.

    Args:
        file_id: The file ID to check.

    Returns:
        True if this is a chunk ID, False otherwise.
    """
    if CHUNK_PATTERN not in file_id:
        return False
    suffix = file_id.split(CHUNK_PATTERN)[-1]
    return "of" in suffix


def parse_chunk_id(chunk_id: str) -> Tuple[str, int, int]:
    """Parse a chunk ID to extract the base file ID, chunk index, and total.

    Args:
        chunk_id: The chunk file ID (e.g., 'result:file-abc123.gz.chunk.0of3').

    Returns:
        Tuple of (base_file_id, chunk_index, total_chunks).
    """
    idx = chunk_id.rfind(CHUNK_PATTERN)
    if idx == -1:
        raise ValueError(f"Not a valid chunk ID: {chunk_id}")

    base_id = chunk_id[:idx]
    chunk_info = chunk_id[idx + len(CHUNK_PATTERN) :]

    if "of" not in chunk_info:
        raise ValueError(f"Not a valid chunk ID (missing 'of'): {chunk_id}")

    n_str, k_str = chunk_info.split("of", 1)
    return base_id, int(n_str), int(k_str)


def make_chunk_id(base_id: str, chunk_index: int, total_chunks: int) -> str:
    """Create a chunk ID from base file ID, index, and total.

    Args:
        base_id: The base file ID (e.g., 'result:file-abc123.gz').
        chunk_index: The zero-based chunk index.
        total_chunks: The total number of chunks.

    Returns:
        The chunk file ID (e.g., 'result:file-abc123.gz.chunk.0of3').
    """
    return f"{base_id}{CHUNK_PATTERN}{chunk_index}of{total_chunks}"


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
        compression_level = GZIP_COMPRESSION_LEVEL
        if len(data) > 1024 * 1024 * 1024:  # 1GB
            compression_level = 9  # Maximum compression level

        return gzip.compress(data, compresslevel=compression_level)

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

    def _upload_single_blob(self, storage_path: str, data: bytes) -> None:
        """Upload a single blob to storage.

        Args:
            storage_path: The storage path to upload to.
            data: The bytes to upload.
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
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

    def _download_single_blob(self, storage_path: str) -> bytes:
        """Download a single blob from storage.

        Args:
            storage_path: The storage path to download from.

        Returns:
            The downloaded bytes.
        """
        return self._ow._supabase.storage.from_("files").download(storage_path)

    def _upload_chunked(
        self, file_id: str, upload_data: bytes, is_compressed: bool
    ) -> List[str]:
        """Upload a large file in chunks.

        Chunks are named: {base_id}.chunk.{n}of{k} (e.g., file.gz.chunk.0of3).
        Compression is indicated by .gz suffix on the base_id before .chunk.

        Args:
            file_id: The base file ID.
            upload_data: The (possibly compressed) data to upload.
            is_compressed: Whether the data is gzip compressed.

        Returns:
            List of chunk IDs that were uploaded.
        """
        chunks = split_into_chunks(upload_data, CHUNK_SIZE_BYTES)
        chunk_count = len(chunks)
        logger.info(
            f"Splitting file {file_id} into {chunk_count} chunks "
            f"({len(upload_data) / 1024 / 1024:.1f} MB total)"
        )

        # Add compression marker to base_id so we know to decompress on download
        base_id_with_marker = f"{file_id}.gz" if is_compressed else file_id

        chunk_ids = []
        for idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(base_id_with_marker, idx, chunk_count)
            chunk_storage_path = self._get_storage_path(chunk_id)

            logger.info(
                f"Uploading chunk {idx + 1}/{chunk_count}: {chunk_id} "
                f"({len(chunk) / 1024 / 1024:.1f} MB)"
            )
            self._upload_single_blob(chunk_storage_path, chunk)
            chunk_ids.append(chunk_id)

        return chunk_ids

    def _download_chunked(self, file_id: str) -> Optional[bytes]:
        """Download a chunked file by detecting and fetching all chunks.

        Tries to find chunk 0 (with .gz and without). Once found, the chunk ID
        contains the total count (e.g., .chunk.0of5), so we know exactly how
        many chunks to download.

        Args:
            file_id: The base file ID (not a chunk ID).

        Returns:
            The reassembled and decompressed bytes, or None if not a chunked file.
        """
        # Try to find chunk 0 - check compressed version first
        # We need to probe for different totals, but we can use a simple approach:
        # try totals 1, 2, 3... until we find one that works
        is_compressed = False
        total_chunks = None
        base_id_with_marker = None

        # Probe for chunk 0 with different totals (most files < 100 chunks = 10GB)
        for probe_total in range(1, 1000):
            # Try compressed chunks first ({file_id}.gz.chunk.0of{n})
            chunk_id_gz = make_chunk_id(f"{file_id}.gz", 0, probe_total)
            storage_path_gz = self._get_storage_path(chunk_id_gz)
            try:
                self._download_single_blob(storage_path_gz)
                is_compressed = True
                base_id_with_marker = f"{file_id}.gz"
                total_chunks = probe_total
                break
            except Exception:
                pass

            # Try uncompressed chunks ({file_id}.chunk.0of{n})
            chunk_id = make_chunk_id(file_id, 0, probe_total)
            storage_path = self._get_storage_path(chunk_id)
            try:
                self._download_single_blob(storage_path)
                is_compressed = False
                base_id_with_marker = file_id
                total_chunks = probe_total
                break
            except Exception:
                pass

        if total_chunks is None:
            # Not a chunked file
            return None

        logger.info(
            f"Downloading chunked file {file_id}: {total_chunks} chunks, "
            f"compressed={is_compressed}"
        )

        # Download all chunks (we already know the total from chunk 0's name)
        chunks = []
        for idx in range(total_chunks):
            chunk_id = make_chunk_id(base_id_with_marker, idx, total_chunks)
            chunk_storage_path = self._get_storage_path(chunk_id)
            logger.info(f"Downloading chunk {idx + 1}/{total_chunks}: {chunk_id}")
            chunk_data = self._download_single_blob(chunk_storage_path)
            chunks.append(chunk_data)

        # Reassemble
        reassembled = join_chunks(chunks)
        logger.info(f"Reassembled {total_chunks} chunks into {len(reassembled)} bytes")

        # Decompress if needed
        if is_compressed:
            reassembled = self._decompress_data(reassembled)
            logger.info(f"Decompressed to {len(reassembled)} bytes")

        return reassembled

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

    @supabase_retry(max_time=1800, max_tries=5)
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

        # Check if we need to chunk the file (> 100 MB after compression)
        is_chunked = len(upload_data) > CHUNK_SIZE_BYTES

        if is_chunked:
            # Upload as chunks with manifest
            self._upload_chunked(file_id, upload_data, is_compressed)
            logger.info(
                f"Uploaded chunked file: {file_id} "
                f"({len(upload_data) / 1024 / 1024:.1f} MB in chunks)"
            )
        else:
            # Upload as single file
            storage_path = self._get_storage_path(file_id)
            if is_compressed:
                storage_path = storage_path + ".gz"
            self._upload_single_blob(storage_path, upload_data)

        max_int32 = 2**31 - 1
        if original_size > max_int32:
            original_size = 0  # Value too large to store in int32

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
        """Get file content, automatically handling chunked files and decompression.

        Tries the following in order:
        1. Check for chunked file (chunk.0 exists) and reassemble all chunks
        2. Try compressed path (.gz)
        3. Fall back to uncompressed path

        Args:
            file_id: The ID of the file to download.

        Returns:
            The file content as bytes (reassembled and decompressed if needed).
        """
        logger.info(f"Downloading file: {file_id}")

        # First, try to download as a chunked file (probe for chunk 0)
        chunked_content = self._download_chunked(file_id)
        if chunked_content is not None:
            return chunked_content

        # Not a chunked file, try regular download
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
