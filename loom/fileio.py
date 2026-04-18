import os
import mmap
import pickle
import numpy as np


class ByteFileDB:
    def __init__(self, filename, initial_size=1024, header_size=4096):
        self.filename = filename
        self.log_filename = self.filename + ".log"
        self.initial_size = initial_size
        self.header_size = header_size
        self.ensure_file_size(max(self.initial_size, self.header_size))
        self.mapped_file = None
        self.file_handle = None

        # Header metadata dictionary (in-memory cache)
        self._header_data = {}
        self._header_dirty = False
        self._batch_mode = False
        self._freelist = []
        self._freelist_dirty = False
        self._allocation_index_key = "_allocation_index"
        self._is_initialized_key = "_is_initialized"

    def _grow_file_size(self, min_size):
        """Grow the file to at least min_size using exponential growth."""
        current = os.path.getsize(self.filename) if os.path.exists(self.filename) else 0
        target = max(min_size, current * 2)
        self.ensure_file_size(target)

    def ensure_file_size(self, size):
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) < size:
            with open(self.filename, "ab") as f:
                f.truncate(size)

    def open(self):
        self.ensure_file_size(max(self.initial_size, self.header_size))
        self.file_handle = open(self.filename, "r+b")
        self.mapped_file = mmap.mmap(self.file_handle.fileno(), 0)
        self.recover_from_log()
        self._load_header()

    def close(self):
        if self.mapped_file:
            # Persist freelist if dirty
            if self._freelist_dirty:
                self._save_freelist()
                self._save_header()
            self.mapped_file.close()
            self.mapped_file = None
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def _remap(self, min_size):
        """Grow file and re-create mmap. Flushes dirty header first."""
        if self._header_dirty:
            self._save_header()
        self.close()
        self._grow_file_size(min_size)
        self.open()

    def write(self, address, data):
        assert self.mapped_file, "DB is not open. Call open() first."
        end = address + len(data)
        if end > len(self.mapped_file):
            self._remap(end)
        self.mapped_file[address : end] = data

    def read(self, address, size):
        assert self.mapped_file, "DB is not open. Call open() first."
        return self.mapped_file[address : address + size]

    def log_write(self, log_handle, address, data):
        log_handle.write((address).to_bytes(8, "big"))
        log_handle.write((len(data)).to_bytes(4, "big"))
        log_handle.write(data)

    def log_commit(self, log_handle):
        log_handle.write(b"COMMIT\x00\x00")
        log_handle.flush()
        os.fsync(log_handle.fileno())

    def recover_from_log(self):
        if os.path.exists(self.log_filename):
            with open(self.log_filename, "rb") as log:
                writes = []
                while True:
                    address_data = log.read(8)
                    if not address_data:
                        break
                    # Check if this is the COMMIT marker
                    if address_data == b"COMMIT\x00\x00":
                        # Transaction was committed, apply writes
                        for address, data in writes:
                            self.write(address, data)
                        break

                    address = int.from_bytes(address_data, "big")
                    length = int.from_bytes(log.read(4), "big")
                    data = log.read(length)
                    writes.append((address, data))

            self.clear_log()

    def transaction(self, writes):
        with open(self.log_filename, "ab") as log:
            for address, data in writes:
                self.log_write(log, address, data)
            self.log_commit(log)

        for address, data in writes:
            self.write(address, data)

        self.clear_log()

    def clear_log(self):
        with open(self.log_filename, "wb") as log:
            pass

    # -------------------------------------------------------------------------
    # Header management methods (Phase 1)
    # -------------------------------------------------------------------------

    def _load_header(self):
        """Load header metadata from file into memory."""
        if len(self.mapped_file) < self.header_size:
            return

        header_bytes = self.mapped_file[0 : self.header_size]

        # Check if header is initialized (first 4 bytes should be magic number)
        magic = header_bytes[0:4]
        if magic == b"LOOM":
            # Header exists, deserialize it
            try:
                # Read size of pickled data (next 4 bytes)
                data_size = np.frombuffer(header_bytes[4:8], dtype="uint32")[0]
                if data_size > 0 and data_size < self.header_size - 8:
                    pickled_data = header_bytes[8 : 8 + data_size]
                    self._header_data = pickle.loads(pickled_data)
            except Exception:
                # Corrupted header, initialize fresh
                self._initialize_header()
        else:
            # No header, initialize
            self._initialize_header()

        self._header_dirty = False
        self._load_freelist()

    def _initialize_header(self):
        """Initialize a fresh header."""
        self._header_data = {
            self._is_initialized_key: True,
            self._allocation_index_key: self.header_size,  # Start allocating after header
        }
        self._save_header()

    def _save_header(self):
        """Persist header metadata to file (direct mmap write, no WAL)."""
        assert self.mapped_file, "DB is not open. Call open() first."

        # Serialize header data
        pickled_data = pickle.dumps(self._header_data)
        data_size = len(pickled_data)

        if data_size > self.header_size - 8:
            raise ValueError(
                f"Header data too large: {data_size} bytes (max {self.header_size - 8})"
            )

        # Write magic number, size, and data — direct mmap (no WAL overhead)
        header_bytes = b"LOOM" + np.uint32(data_size).tobytes() + pickled_data
        header_bytes += b"\x00" * (self.header_size - len(header_bytes))
        self.mapped_file[0 : self.header_size] = header_bytes
        self.mapped_file.flush()
        self._header_dirty = False

    def set_header_field(self, name, value):
        """Set a metadata field in the header.

        Persists immediately unless batch mode is active, in which case
        the write is deferred until end_batch().

        Args:
            name: Field name (string)
            value: Any picklable Python object
        """
        assert self.mapped_file, "DB is not open. Call open() first."
        self._header_data[name] = value
        if self._batch_mode:
            self._header_dirty = True
        else:
            self._save_header()

    def begin_batch(self):
        """Enter batch mode: defer header writes until end_batch()."""
        self._batch_mode = True

    def end_batch(self):
        """Exit batch mode and flush any pending header writes."""
        self._batch_mode = False
        if self._header_dirty:
            self._save_header()

    def flush_header(self):
        """Flush pending header changes to disk."""
        if self._header_dirty:
            self._save_header()

    def get_header_field(self, name, default=None):
        """Get a metadata field from the header.

        Args:
            name: Field name (string)
            default: Default value if field doesn't exist

        Returns:
            Field value or default
        """
        return self._header_data.get(name, default)

    def has_header_field(self, name):
        """Check if a header field exists."""
        return name in self._header_data

    def delete_header_field(self, name):
        """Delete a header field."""
        if name in self._header_data:
            del self._header_data[name]
            self._save_header()

    # -------------------------------------------------------------------------
    # Allocation management with freelist
    # -------------------------------------------------------------------------

    _FREELIST_KEY = "_file_freelist"

    def _load_freelist(self):
        """Load file-level freelist from header."""
        self._freelist = self._header_data.get(self._FREELIST_KEY, [])
        self._freelist_dirty = False

    def _save_freelist(self):
        """Persist freelist to header (only if changed)."""
        if not self._freelist_dirty:
            return
        self._header_data[self._FREELIST_KEY] = self._freelist
        self._freelist_dirty = False
        # The header will be flushed by set_header_field or end_batch

    def free(self, address, size):
        """Return a previously allocated block to the freelist.

        The block can be reused by future allocate() calls.
        Adjacent freed regions are merged automatically.

        Args:
            address: Start address of the block
            size: Size in bytes
        """
        if size <= 0:
            return

        # Insert and merge with adjacent regions
        self._freelist.append((address, size))
        self._freelist.sort(key=lambda x: x[0])

        # Merge adjacent
        merged = []
        for offset, sz in self._freelist:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + sz)
            else:
                merged.append((offset, sz))
        self._freelist = merged
        self._freelist_dirty = True

    def _alloc_from_freelist(self, size):
        """Try to allocate from freelist (best-fit).

        Returns address if found, None otherwise.
        """
        best_idx = None
        best_size = None

        for i, (offset, sz) in enumerate(self._freelist):
            if sz >= size:
                if best_idx is None or sz < best_size:
                    best_idx = i
                    best_size = sz

        if best_idx is not None:
            offset, sz = self._freelist.pop(best_idx)
            excess = sz - size
            if excess > 0:
                self._freelist.append((offset + size, excess))
                self._freelist.sort(key=lambda x: x[0])
            self._freelist_dirty = True
            # Zero-fill to prevent stale data from being read
            self.write(offset, b"\x00" * size)
            return offset

        return None

    def allocate(self, size):
        """Allocate a block of memory and return its address.

        Tries the freelist first (best-fit), then bump-allocates.

        Args:
            size: Number of bytes to allocate

        Returns:
            Address (int) of the allocated block
        """
        assert self.mapped_file, "DB is not open. Call open() first."

        # Try freelist first
        addr = self._alloc_from_freelist(size)
        if addr is not None:
            # Persist freelist change with header
            self._save_freelist()
            if self._batch_mode:
                self._header_dirty = True
            else:
                self._save_header()
            return addr

        # Bump-allocate
        current_index = self._header_data.get(
            self._allocation_index_key, self.header_size
        )

        required_size = current_index + size
        if required_size > len(self.mapped_file):
            self._remap(required_size)

        self._header_data[self._allocation_index_key] = current_index + size
        if self._batch_mode:
            self._header_dirty = True
        else:
            self._save_header()

        return current_index

    def get_allocation_index(self):
        """Get the current allocation index (write head position)."""
        return self._header_data.get(self._allocation_index_key, self.header_size)

    def get_file_size(self):
        """Get the current file size."""
        if self.mapped_file:
            return len(self.mapped_file)
        return os.path.getsize(self.filename) if os.path.exists(self.filename) else 0

    def get_used_space(self):
        """Get the amount of space currently allocated."""
        return self.get_allocation_index()

    def get_free_space(self):
        """Get the amount of free space in the file."""
        return self.get_file_size() - self.get_used_space()
