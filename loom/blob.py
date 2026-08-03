"""
Blob storage with compression and space reuse.

Blobs are variable-size binary data stored in fixed-size slots.
Deleted blob slots are tracked in a freelist for reuse.
"""

import struct
from math import ceil

try:
    import brotli

    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import zlib

    HAS_ZLIB = True
except ImportError:
    HAS_ZLIB = False

# Sentinel: "no per-call codec given → use the store's default compression".
# Distinct from None, which means "explicitly store this blob uncompressed".
_DEFAULT = object()

# Default compression level per codec when none is specified.  brotli's own
# default is quality 11 — pathologically slow (tens of ms for a few KB) for a
# marginal ratio gain over ~5; zlib's is 6.  These are the write-side levels;
# decompression needs no level (it's self-describing in the stream).
_DEFAULT_LEVEL = {"brotli": 5, "zlib": 6}


class BlobStore:
    """Append-only blob storage with slot-based allocation and freelist reuse.

    Blobs are stored in fixed-size slots (default 64 bytes). Large blobs
    span multiple consecutive slots. Deleted slots go to a freelist and
    are reused for new blobs (best-fit allocation).

    Storage format per blob:
        [compressed_size: 4 bytes][original_size: 4 bytes][compressed_data: N bytes]

    Usage:
        store = BlobStore(db)

        # Write blob
        offset, n_slots = store.write(b"my data")

        # Read blob
        data = store.read(offset)

        # Delete (returns slots to freelist)
        store.delete(offset, n_slots)
    """

    SLOT_SIZE = 64  # Minimum allocation unit (bytes)
    HEADER_SIZE = 8  # compressed_size (4) + original_size (4)

    def __init__(self, db, compression=None, level=None):
        """Initialize blob store.

        Args:
            db: ByteFileDB instance
            compression: Compression algorithm ("brotli", "zlib", or None).
                Default None — no compression.
            level: Compression level. None → the codec default (brotli 5,
                zlib 6). brotli accepts 0–11, zlib 0–9. A per-write override
                is possible via write(..., level=...); a per-field one via the
                schema tag ``text[brotli:5]`` (see Dataset).
        """
        self._db = db
        self._compression = compression
        self._level = level

        # Validate compression
        if compression == "brotli" and not HAS_BROTLI:
            raise ImportError("brotli package required for brotli compression")
        if compression == "zlib" and not HAS_ZLIB:
            raise ImportError("zlib module required for zlib compression")

        # Freelist: list of (offset, n_slots) tuples
        # Sorted by n_slots ascending for best-fit allocation
        self._freelist = []
        self._freelist_dirty = False

        # Load freelist from header if exists
        self._load_freelist()

    def _compress(self, data: bytes, algo, level=None) -> bytes:
        """Compress data with the given algorithm and level.

        level=None picks the codec default (see _DEFAULT_LEVEL).
        """
        if algo == "brotli":
            if not HAS_BROTLI:
                raise ImportError("brotli package required for brotli compression")
            q = _DEFAULT_LEVEL["brotli"] if level is None else level
            return brotli.compress(data, quality=q)
        elif algo == "zlib":
            if not HAS_ZLIB:
                raise ImportError("zlib module required for zlib compression")
            lv = _DEFAULT_LEVEL["zlib"] if level is None else level
            return zlib.compress(data, lv)
        return data

    def _decompress(self, data: bytes, algo) -> bytes:
        """Decompress data using the given algorithm ("brotli"/"zlib"/None)."""
        if algo == "brotli":
            return brotli.decompress(data)
        elif algo == "zlib":
            return zlib.decompress(data)
        return data

    def _slots_needed(self, data_size: int) -> int:
        """Calculate number of slots needed for data."""
        total_size = self.HEADER_SIZE + data_size
        return int(ceil(total_size / self.SLOT_SIZE))

    def _load_freelist(self):
        """Load freelist from database header."""
        freelist_data = self._db.get_header_field("_blob_freelist")
        if freelist_data:
            self._freelist = freelist_data

    def _save_freelist(self):
        """Save freelist to database header (only if changed)."""
        if not self._freelist_dirty:
            return
        self._db.set_header_field("_blob_freelist", self._freelist)
        self._freelist_dirty = False

    def _alloc_from_freelist(self, n_slots: int) -> int | None:
        """Find best-fit slot in freelist.

        Args:
            n_slots: Number of slots needed

        Returns:
            Offset if found, None otherwise
        """
        best_idx = None
        best_slots = None

        for i, (offset, slots) in enumerate(self._freelist):
            if slots >= n_slots:
                if best_idx is None or slots < best_slots:
                    best_idx = i
                    best_slots = slots

        if best_idx is not None:
            offset, slots = self._freelist.pop(best_idx)
            self._freelist_dirty = True
            # Return excess slots to freelist
            excess = slots - n_slots
            if excess > 0:
                excess_offset = offset + n_slots * self.SLOT_SIZE
                self._add_to_freelist(excess_offset, excess)
            return offset

        return None

    def _add_to_freelist(self, offset: int, n_slots: int):
        """Add slots to freelist, merging adjacent regions."""
        # Add to list
        self._freelist.append((offset, n_slots))
        self._freelist_dirty = True

        # Sort by offset for merging
        self._freelist.sort(key=lambda x: x[0])

        # Merge adjacent regions
        merged = []
        for offset, slots in self._freelist:
            if merged:
                prev_offset, prev_slots = merged[-1]
                prev_end = prev_offset + prev_slots * self.SLOT_SIZE
                if prev_end == offset:
                    # Adjacent - merge
                    merged[-1] = (prev_offset, prev_slots + slots)
                    continue
            merged.append((offset, slots))

        self._freelist = merged

    def write(self, data: bytes, compression=_DEFAULT, level=_DEFAULT) -> tuple[int, int]:
        """Write blob to storage.

        Args:
            data: Raw bytes to store
            compression: Per-call codec override ("brotli"/"zlib"/None). When
                left at the default sentinel, the store's own compression (the
                DB-level setting) is used — preserving prior behaviour. Callers
                that know a field's declared codec (see Dataset) pass it here so
                one field can be compressed while the rest of the DB is not.
            level: Per-call compression level (int) or None for the codec
                default; the sentinel means "use the store's level". Write-only
                — reads don't need it.

        Returns:
            Tuple of (offset, n_slots) - needed for later deletion
        """
        algo = self._compression if compression is _DEFAULT else compression
        lvl = self._level if level is _DEFAULT else level
        # Compress
        compressed = self._compress(data, algo, lvl)

        # Build blob record: [compressed_size][original_size][data]
        header = struct.pack("<II", len(compressed), len(data))
        blob_bytes = header + compressed

        # Calculate slots needed
        n_slots = self._slots_needed(len(compressed))
        total_bytes = n_slots * self.SLOT_SIZE

        # Pad to slot boundary
        padded = blob_bytes.ljust(total_bytes, b"\x00")

        # Try freelist first
        offset = self._alloc_from_freelist(n_slots)

        if offset is not None:
            # Reuse freed space
            self._db.write(offset, padded)
        else:
            # Append to end of file
            offset = self._db.allocate(total_bytes)
            self._db.write(offset, padded)

        self._save_freelist()
        return offset, n_slots

    def read(self, offset: int, compression=_DEFAULT) -> bytes:
        """Read blob from storage.

        Args:
            offset: Blob offset (from write())
            compression: Per-call codec override — MUST match what the blob was
                written with (blobs are not self-describing). Defaults to the
                store's compression, as before. The Dataset passes the field's
                declared codec so a per-field-compressed blob decodes correctly.

        Returns:
            Original uncompressed data
        """
        algo = self._compression if compression is _DEFAULT else compression
        # Read header
        header = self._db.read(offset, self.HEADER_SIZE)
        compressed_size, original_size = struct.unpack("<II", header)

        # Read compressed data
        compressed = self._db.read(offset + self.HEADER_SIZE, compressed_size)

        # Decompress
        return self._decompress(compressed, algo)

    def delete(self, offset: int, n_slots: int):
        """Delete blob and return slots to freelist.

        Args:
            offset: Blob offset
            n_slots: Number of slots (from write())
        """
        self._add_to_freelist(offset, n_slots)
        self._save_freelist()

    def get_stats(self) -> dict:
        """Get storage statistics."""
        free_slots = sum(slots for _, slots in self._freelist)
        free_bytes = free_slots * self.SLOT_SIZE
        return {
            "compression": self._compression,
            "slot_size": self.SLOT_SIZE,
            "free_slots": free_slots,
            "free_bytes": free_bytes,
            "freelist_entries": len(self._freelist),
        }
