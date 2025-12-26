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

    def _file_exists_in_db(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Check if a file exists in the database.

        Args:
            file_id: The file ID to check.

        Returns:
            The file record if it exists, None otherwise.
        """
        try:
            existing_file = (
                self._ow._supabase.table("files")
                .select("*")
                .eq("id", file_id)
                .single()
                .execute()
                .data
            )
            return existing_file if existing_file else None
        except Exception:
            return None

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

    def _upload_chunked(self, file_id: str, compressed_data: bytes) -> List[str]:
        """Upload a large file in chunks.

        Chunks are already compressed. The compressed data is split into chunks.
        Chunk IDs follow the pattern: {base_id}.chunk.{n}of{k}.

        Each chunk is inserted as an independent file in the database to enable
        efficient existence checks and avoid re-uploading existing chunks.

        Args:
            file_id: The base file ID.
            compressed_data: The already-compressed data to upload.

        Returns:
            List of chunk IDs that were uploaded.
        """
        chunks = split_into_chunks(compressed_data, CHUNK_SIZE_BYTES)
        chunk_count = len(chunks)
        logging.info(
            f"Splitting file {file_id} into {chunk_count} chunks "
            f"({len(compressed_data) / 1024 / 1024:.1f} MB total)"
        )

        chunk_ids = []
        for idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(file_id, idx, chunk_count)
            chunk_storage_path = self._get_storage_path(chunk_id)

            # Check if chunk already exists in the database
            existing_chunk = self._file_exists_in_db(chunk_id)
            if existing_chunk:
                logging.info(
                    f"Chunk {idx + 1}/{chunk_count} already exists: {chunk_id} (skipping)"
                )
                chunk_ids.append(chunk_id)
                continue

            logging.info(
                f"Uploading chunk {idx + 1}/{chunk_count}: {chunk_id} "
                f"({len(chunk) / 1024 / 1024:.1f} MB)"
            )
            self._upload_single_blob(chunk_storage_path, chunk)

            # Insert chunk as independent file in the database
            chunk_data_row = {
                "id": chunk_id,
                "filename": chunk_id,
                "purpose": "chunk",
                "bytes": len(chunk),
                "organization_id": self._org_id,
            }
            try:
                self._ow._supabase.table("files").insert(chunk_data_row).execute()
                logging.info(f"Chunk registered in database: {chunk_id}")
            except Exception as e:
                # Chunk might have been inserted by concurrent upload, ignore
                logging.info(
                    f"Could not insert chunk {chunk_id} (may already exist): {e}"
                )

            chunk_ids.append(chunk_id)

        return chunk_ids

    def _find_chunks_in_db(self, base_file_id: str) -> Optional[List[Dict[str, Any]]]:
        """Find all chunks for a file by querying the database.

        Searches for chunk records matching pattern: {base_file_id}.chunk.* or
        {base_file_id}.gz.chunk.* depending on whether base_file_id already has .gz.

        Args:
            base_file_id: The base file ID (without chunk suffix).

        Returns:
            List of chunk records sorted by chunk index, or None if no chunks found.
        """
        # Handle case where base_file_id already includes .gz
        if base_file_id.endswith(".gz"):
            pattern = f"{base_file_id}{CHUNK_PATTERN}%"
        else:
            # Chunks are always compressed, so IDs are {base_file_id}.gz.chunk.{n}of{k}
            pattern = f"{base_file_id}.gz{CHUNK_PATTERN}%"
        try:
            result = (
                self._ow._supabase.table("files")
                .select("*")
                .like("id", pattern)
                .execute()
            )
            if result.data:
                # Sort by chunk index
                chunks = sorted(result.data, key=lambda c: parse_chunk_id(c["id"])[1])
                return chunks
        except Exception as e:
            logging.info(f"Error querying chunks with pattern {pattern}: {e}")
        return None

    def _download_chunked(self, file_id: str) -> Optional[bytes]:
        """Download a chunked file by detecting and fetching all chunks.

        Queries the database for chunk records, which is much more efficient than
        probing storage for each potential chunk total. Chunks are always compressed,
        so decompression is always performed after reassembly.

        Args:
            file_id: The base file ID (not a chunk ID).

        Returns:
            The reassembled and decompressed bytes, or None if not a chunked file.
        """
        # If file_id is itself a chunk ID, extract the base file ID to fetch the full file
        if is_chunk_id(file_id):
            base_file_id_with_gz, _, _ = parse_chunk_id(file_id)
            logging.info(
                f"file_id {file_id} is a chunk ID, extracting base: {base_file_id_with_gz}"
            )
            # The base_file_id includes .gz (e.g., 'result:file-abc123.gz')
            # Strip it since _find_chunks_in_db adds it back
            if base_file_id_with_gz.endswith(".gz"):
                file_id = base_file_id_with_gz[:-3]
            else:
                file_id = base_file_id_with_gz

        # Query database for chunks
        chunk_records = self._find_chunks_in_db(file_id)

        if not chunk_records:
            # Not a chunked file (no chunk records found)
            return None

        # Parse first chunk to determine base ID and total count
        first_chunk_id = chunk_records[0]["id"]
        base_id_with_gz, _, total_chunks = parse_chunk_id(first_chunk_id)

        # Validate we have all chunks
        if len(chunk_records) != total_chunks:
            logging.warning(
                f"Expected {total_chunks} chunks but found {len(chunk_records)} in database"
            )

        logging.info(f"Downloading chunked file {file_id}: {total_chunks} chunks")

        # Download all chunks in order
        chunks = []
        for idx in range(total_chunks):
            chunk_id = make_chunk_id(base_id_with_gz, idx, total_chunks)
            chunk_storage_path = self._get_storage_path(chunk_id)
            logging.info(f"Downloading chunk {idx + 1}/{total_chunks}: {chunk_id}")
            chunk_data = self._download_single_blob(chunk_storage_path)
            chunks.append(chunk_data)

        # Reassemble
        reassembled = join_chunks(chunks)
        logging.info(f"Reassembled {total_chunks} chunks into {len(reassembled)} bytes")

        # Always decompress (chunks are always compressed)
        reassembled = self._decompress_data(reassembled)
        logging.info(f"Decompressed to {len(reassembled)} bytes")

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

        # If the file already exists in the database, return the existing record
        # Check if file already exists (either uncompressed or compressed version)
        for check_id in [file_id, file_id + ".gz"]:
            try:
                existing_file = (
                    self._ow._supabase.table("files")
                    .select("*")
                    .eq("id", check_id)
                    .single()
                    .execute()
                    .data
                )
                if existing_file:
                    logging.info(
                        f"File already exists: {check_id} (purpose: {purpose})"
                    )
                    return existing_file
            except Exception:
                pass  # File doesn't exist, continue checking

        original_size = len(data)
        logging.info(
            f"Uploading file: {filename} (purpose: {purpose}, size: {original_size} bytes)"
        )

        # Validate file content using a fresh buffer
        if not self.validate(io.BytesIO(data), purpose):
            self.validate(io.BytesIO(data), purpose)
            raise ValueError("File content is not valid")

        # Determine the storage path based on whether compression will be applied
        should_compress = self._should_compress(data, purpose)
        storage_filename = filename  # Track actual storage filename

        # Compression and chunking only apply to result files
        if should_compress:
            file_id = file_id + ".gz"
            logging.info(
                f"Compressing file {filename} ({original_size / 1024 / 1024:.1f} MB)..."
            )
            compressed_data = self._compress_data(data)
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size
            logging.info(
                f"Compressed {original_size / 1024 / 1024:.1f} MB -> "
                f"{compressed_size / 1024 / 1024:.1f} MB "
                f"({compression_ratio:.1f}x reduction)"
            )

            # Chunking only for result files with compressed size > CHUNK_SIZE_BYTES
            must_be_chunked = compressed_size > CHUNK_SIZE_BYTES

            if must_be_chunked:
                # Upload as chunks (already compressed)
                # Chunk existence is checked inside _upload_chunked
                self._upload_chunked(file_id, compressed_data)
                storage_filename = f"{file_id}.chunk.*"
                logging.info(
                    f"Uploaded chunked file: {file_id} "
                    f"({compressed_size / 1024 / 1024:.1f} MB in chunks)"
                )
            else:
                # Upload as single compressed file
                storage_path = self._get_storage_path(file_id)
                storage_filename = f"{filename}.gz"
                self._upload_single_blob(storage_path, compressed_data)
        else:
            # Upload as single uncompressed file
            storage_path = self._get_storage_path(file_id)
            self._upload_single_blob(storage_path, data)

        max_int32 = 2**31 - 1
        if original_size > max_int32:
            original_size = 0  # Value too large to store in int32

        # Create database entry (store original size, not compressed)
        file_record = {
            "id": file_id,
            "filename": storage_filename,
            "purpose": purpose,
            "bytes": original_size,
            "organization_id": self._org_id,
        }

        # Check if file record already exists to avoid duplicate key errors on retry.
        # For chunked files, the parent file record uses the original file_id (e.g.,
        # 'result:file-abc123.gz') with correct purpose and original size, while
        # chunks are stored separately with their own records.
        existing = self._file_exists_in_db(file_id)
        if existing:
            logging.info(f"File record already exists: {file_id}")
            file_record = existing
        else:
            self._ow._supabase.table("files").insert(file_record).execute()
            logging.info(f"File uploaded successfully: {file_id}")

        # Add fields for the return value (not stored in DB)
        file_record["object"] = "file"
        file_record["created_at"] = datetime.now().timestamp()

        return file_record

    @supabase_retry(max_time=600, max_tries=5)
    def content(self, file_id: str) -> bytes:
        """Get file content, automatically handling chunked files and decompression.

        If the user passes a chunk ID (e.g., 'file-abc.chunk.0of3'), this method
        automatically detects it and returns the full reassembled file, not just
        that single chunk.

        Args:
            file_id: The ID of the file to download (can be a chunk ID).

        Returns:
            The file content as bytes (reassembled and decompressed if needed).
        """
        logging.info(f"Downloading file: {file_id}")

        # Check if the user passed a chunk ID - if so, extract the base file ID
        # and return the full reassembled file
        if is_chunk_id(file_id):
            base_file_id, _, _ = parse_chunk_id(file_id)
            logging.info(
                f"Detected chunk ID {file_id}, fetching full file: {base_file_id}"
            )
            chunked_content = self._download_chunked(base_file_id)
            if chunked_content is not None:
                return chunked_content
            # Fall through if chunks not found (shouldn't happen normally)

        # Check if file exists in database
        file_record = self._file_exists_in_db(file_id)
        if not file_record:
            raise FileNotFoundError(f"File not found: {file_id}")

        storage_path = self._get_storage_path(file_id)
        is_compressed = file_id.endswith(".gz")

        # If file_id indicates compression (.gz suffix), check for chunks first
        # since large compressed files are stored as chunks, not as a single file
        if is_compressed:
            chunked_content = self._download_chunked(file_id)
            if chunked_content is not None:
                return chunked_content

            # Not chunked, download as single compressed file
            content = self._ow._supabase.storage.from_("files").download(storage_path)
            logging.info(
                f"Downloaded compressed file: {file_id} ({len(content)} bytes)"
            )
            content = self._decompress_data(content)
            logging.info(f"Decompressed to {len(content)} bytes")
            return content

        # For non-.gz file IDs, try legacy compressed path first (backward compatibility),
        # then fall back to uncompressed path
        try:
            content = self._ow._supabase.storage.from_("files").download(
                storage_path + ".gz"
            )
            logging.info(
                f"Downloaded compressed file (legacy): {file_id} ({len(content)} bytes)"
            )
            content = self._decompress_data(content)
            logging.info(f"Decompressed to {len(content)} bytes")
            return content
        except Exception:
            pass

        # Download as uncompressed file
        content = self._ow._supabase.storage.from_("files").download(storage_path)
        logging.info(f"Downloaded file: {file_id} ({len(content)} bytes)")
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
