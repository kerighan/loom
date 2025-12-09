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
        self._allocation_index_key = "_allocation_index"
        self._is_initialized_key = "_is_initialized"

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
            self.mapped_file.close()
            self.mapped_file = None
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def write(self, address, data):
        assert self.mapped_file, "DB is not open. Call open() first."
        if address + len(data) > len(self.mapped_file):
            self.close()
            self.ensure_file_size(address + len(data) + self.initial_size)
            self.open()
        self.mapped_file[address : address + len(data)] = data

    def read(self, address, size):
        assert self.mapped_file, "DB is not open. Call open() first."
        return self.mapped_file[address : address + size]

    def log_write(self, address, data):
        with open(self.log_filename, "ab") as log:
            log.write((address).to_bytes(4, "big"))
            log.write((len(data)).to_bytes(4, "big"))
            log.write(data)

    def log_commit(self):
        with open(self.log_filename, "ab") as log:
            log.write(b"COMMIT")

    def recover_from_log(self):
        if os.path.exists(self.log_filename):
            with open(self.log_filename, "rb") as log:
                writes = []
                while True:
                    address_data = log.read(4)
                    if not address_data:
                        break
                    # Check if this is the COMMIT marker
                    if address_data == b"COMM":
                        # Read the remaining 'IT' part
                        if log.read(2) == b"IT":
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
        # First, log all the writes.
        for address, data in writes:
            self.log_write(address, data)

        # Commit the transaction.
        self.log_commit()

        # Now apply the logged writes to the database.
        for address, data in writes:
            self.write(address, data)

        self.clear_log()

    def clear_log(self):
        # Ensure parent directory exists
        log_dir = os.path.dirname(self.log_filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Clear or create log file
        try:
            with open(self.log_filename, "wb") as log:
                log.truncate(0)
        except OSError:
            # If file doesn't exist, just create it
            open(self.log_filename, "wb").close()

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

    def _initialize_header(self):
        """Initialize a fresh header."""
        self._header_data = {
            self._is_initialized_key: True,
            self._allocation_index_key: self.header_size,  # Start allocating after header
        }
        self._save_header()

    def _save_header(self):
        """Persist header metadata to file."""
        assert self.mapped_file, "DB is not open. Call open() first."

        # Serialize header data
        pickled_data = pickle.dumps(self._header_data)
        data_size = len(pickled_data)

        if data_size > self.header_size - 8:
            raise ValueError(
                f"Header data too large: {data_size} bytes (max {self.header_size - 8})"
            )

        # Write magic number, size, and data
        header_bytes = b"LOOM" + np.uint32(data_size).tobytes() + pickled_data

        # Pad to header_size
        header_bytes += b"\x00" * (self.header_size - len(header_bytes))

        self.transaction([(0, header_bytes)])
        if self.mapped_file:
            self.mapped_file.flush()

    def set_header_field(self, name, value):
        """Set a metadata field in the header.

        Args:
            name: Field name (string)
            value: Any picklable Python object
        """
        assert self.mapped_file, "DB is not open. Call open() first."
        self._header_data[name] = value
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
    # Allocation management methods (Phase 1)
    # -------------------------------------------------------------------------

    def allocate(self, size):
        """Allocate a block of memory and return its address.

        Args:
            size: Number of bytes to allocate

        Returns:
            Address (int) of the allocated block
        """
        assert self.mapped_file, "DB is not open. Call open() first."

        # Get current allocation index
        current_index = self._header_data.get(
            self._allocation_index_key, self.header_size
        )

        # Ensure file is large enough
        required_size = current_index + size
        if required_size > len(self.mapped_file):
            self.close()
            self.ensure_file_size(required_size)
            self.open()

        # Update allocation index
        self._header_data[self._allocation_index_key] = current_index + size
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
