import os
import mmap
import pickle
import shutil
import struct
import zlib
import numpy as np

from loom.errors import ReadOnlyError


class ByteFileDB:
    # Double-buffer header layout:
    #   [0]            active slot indicator (0=A, 1=B) — a hint, see below
    #   [1 : slot_size+1]   slot A  — [LOM2 4B][seqno 8B][size 4B][crc32 4B][pickle NB]
    #   [slot_size+1 : header_size]  slot B  — same layout
    #
    # Crash consistency (why seqno + CRC, not just the double buffer):
    #   With lazy writeback (sync_writes=False) mmap dirty pages reach disk in
    #   arbitrary order and with no barrier, so a hard shutdown can flush the
    #   active-slot flip byte while the slot's own pages are only half written.
    #   A half-written pickle can still deserialize into a *plausible but mixed*
    #   dict (two header generations spliced page-by-page) — which is exactly
    #   how a List ended up with `length` from one generation and `p_last` from
    #   another, crashing _calculate_block_and_offset on reopen.
    #   The per-slot CRC rejects any torn slot; the monotonic seqno lets the
    #   reader pick the newest *intact* slot regardless of the (possibly
    #   half-flushed) hint byte.  A crash can still roll the header back to an
    #   older but fully-consistent generation — never to a spliced one.
    _MAGIC = b"LOM2"  # current slot format (seqno + CRC)
    _LEGACY_MAGIC = b"LOOM"  # pre-CRC format, still readable
    _SLOT_HDR = 20  # 4 magic + 8 seqno + 4 size + 4 crc

    def __init__(
        self,
        filename,
        initial_size=1024,
        header_size=32768,
        sync_writes=False,
        flag="r+",
    ):
        if flag not in ("r", "r+", "w", "rw"):
            raise ValueError("flag must be one of 'r', 'r+', 'w', or 'rw'")
        self.filename = filename
        self.log_filename = self.filename + ".log"
        # Opt-in durable transaction: a full-file pre-image snapshot. Its mere
        # presence on disk means an in-flight durable block did not commit, so
        # open() restores from it (see _recover_txn_snapshot / begin_txn).
        self.txn_filename = self.filename + ".txn"
        self._txn_depth = 0
        self.initial_size = initial_size
        self.header_size = header_size
        self._slot_size = (header_size - 1) // 2  # usable bytes per slot
        self.sync_writes = sync_writes  # if True, flush after every header save
        self.flag = flag
        self.read_only = flag == "r"
        if not self.read_only:
            self.ensure_file_size(max(self.initial_size, self.header_size))
        self.mapped_file = None
        self.file_handle = None
        self._map_size = 0

        # Header metadata dictionary (in-memory cache)
        self._header_data = {}
        self._header_seqno = 0  # monotonic; picks newest intact slot on load
        self._header_dirty = False
        self._write_count = 0  # tracks writes since last flush
        self._batch_mode = False
        self._batch_depth = 0  # begin_batch/end_batch nesting level
        self._freelist = []
        self._freelist_dirty = False
        self._allocation_index_key = "_allocation_index"
        self._is_initialized_key = "_is_initialized"
        self._header_size_key = "_header_size"  # persisted → self-describing on reopen

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
        if self.read_only:
            if not os.path.exists(self.filename):
                raise FileNotFoundError(self.filename)
            if os.path.exists(self.log_filename) and os.path.getsize(self.log_filename):
                raise ReadOnlyError(
                    "Cannot open read-only database with a pending WAL log; "
                    "reopen it writable once to recover"
                )
            if os.path.exists(self.txn_filename):
                raise ReadOnlyError(
                    "Cannot open read-only database with an interrupted durable "
                    "transaction; reopen it writable once to roll back"
                )
            self.file_handle = open(self.filename, "rb")
            self.mapped_file = mmap.mmap(
                self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
            )
        else:
            # A leftover snapshot means a durable block was interrupted (hard
            # crash / kill -9): roll the main file back to its pre-block state
            # before mapping it.  Must run before ensure_file_size touches it.
            self._recover_txn_snapshot()
            self.ensure_file_size(max(self.initial_size, self.header_size))
            self.file_handle = open(self.filename, "r+b")
            self.mapped_file = mmap.mmap(self.file_handle.fileno(), 0)
            self.recover_from_log()
        self._map_size = len(self.mapped_file)
        # Adopt the header_size the file was actually created with (stored in
        # slot A, readable at a fixed offset) so reopening with a different /
        # default header_size can't silently mis-read the header.
        stored = self._peek_stored_header_size()
        if stored is not None and stored != self.header_size:
            self.header_size = stored
            self._slot_size = (stored - 1) // 2
        self._load_header()

    def close(self):
        if self.mapped_file:
            if not self.read_only:
                # Persist freelist if dirty
                if self._freelist_dirty:
                    self._save_freelist()
                # Final header save + flush to guarantee on-disk durability
                if self._header_dirty:
                    self._save_header()
                self.mapped_file.flush()
            self.mapped_file.close()
            self.mapped_file = None
            self._map_size = 0
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    # -------------------------------------------------------------------------
    # Durable transaction: full-file pre-image snapshot (opt-in, off by default)
    #
    # begin_txn() fsyncs a consistent snapshot of the whole DB to <file>.txn
    # BEFORE the block's writes touch the main file.  Everything then runs at
    # normal (lazy, fast) speed against the main file.  commit_txn() fsyncs the
    # main file, then removes the snapshot — that removal is the atomic commit
    # point.  A hard crash anywhere in between leaves the snapshot on disk, so
    # the next open() restores the main file to its exact pre-block state:
    # all-or-nothing, no torn record/node/header can survive.  The cost is one
    # file copy per outermost block, paid only when durability is requested.
    # -------------------------------------------------------------------------

    @staticmethod
    def _fsync_path(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_dir(self):
        directory = os.path.dirname(os.path.abspath(self.filename)) or "."
        self._fsync_path(directory)

    def _recover_txn_snapshot(self):
        """Restore the main file from an interrupted durable block, if any.

        No-op while a block is active (``_txn_depth > 0``): a _remap() during
        the block reopens the file and must NOT roll back the very snapshot it
        is protecting.
        """
        if self._txn_depth > 0:
            return
        if not os.path.exists(self.txn_filename):
            return
        shutil.copyfile(self.txn_filename, self.filename)
        self._fsync_path(self.filename)
        os.remove(self.txn_filename)
        self._fsync_dir()
        # The pre-image predates any WAL entries; drop a stale log too.
        if os.path.exists(self.log_filename) and os.path.getsize(self.log_filename):
            with open(self.log_filename, "wb"):
                pass

    def begin_txn(self):
        """Enter a durable block. Re-entrant; only the outermost snapshots."""
        self._ensure_writable()
        if self._txn_depth == 0:
            # Flush a consistent pre-image, then copy it durably to <file>.txn.
            self.flush()
            if self.file_handle is not None:
                os.fsync(self.file_handle.fileno())
            tmp = self.txn_filename + ".partial"
            shutil.copyfile(self.filename, tmp)
            self._fsync_path(tmp)
            os.replace(tmp, self.txn_filename)  # atomic: .txn appears complete
            self._fsync_dir()
        self._txn_depth += 1

    def commit_txn(self):
        """Commit the durable block: make the main file durable, drop snapshot."""
        if self._txn_depth == 0:
            return
        self._txn_depth -= 1
        if self._txn_depth == 0:
            self.flush()
            if self.file_handle is not None:
                os.fsync(self.file_handle.fileno())
            if os.path.exists(self.txn_filename):
                os.remove(self.txn_filename)
                self._fsync_dir()

    def rollback_txn(self):
        """Abort: discard in-flight changes and restore the pre-image live."""
        self._txn_depth = 0
        if not os.path.exists(self.txn_filename):
            return
        if self.mapped_file is not None:
            self.mapped_file.close()
            self.mapped_file = None
            self._map_size = 0
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
        # _recover_txn_snapshot() restores the file, clears the log, and (now
        # that _txn_depth is 0) is not skipped; open() then remaps + reloads.
        self.open()

    def _remap(self, min_size):
        """Grow file and re-create mmap. Flushes before closing."""
        self._ensure_writable()
        if self._header_dirty:
            self._save_header()
        if self.mapped_file:
            self.mapped_file.flush()
        self.close()
        self._grow_file_size(min_size)
        self.open()

    def write(self, address, data):
        assert self.mapped_file, "DB is not open. Call open() first."
        self._ensure_writable()
        end = address + len(data)
        if end > len(self.mapped_file):
            self._remap(end)
        self.mapped_file[address:end] = data

    def read(self, address, size):
        assert self.mapped_file, "DB is not open. Call open() first."
        end = address + size
        if end > self._map_size:
            self._refresh_map(end)
        return self.mapped_file[address:end]

    def _refresh_map(self, min_end):
        """Remap when a read lands beyond the current mmap.

        mmap pages are shared through the page cache, so a long-lived
        (typically read-only) handle sees another handle's *in-place*
        updates immediately — including fresh pointers into regions
        appended AFTER this handle mapped the file.  Its mmap length is
        frozen at open time though, and slicing past it silently returns
        truncated bytes (surfacing as struct.error / garbage far from the
        cause).  If the file has grown, remap to the new size and carry
        on; if the requested range is still past EOF, the reference is
        dangling — fail loudly.
        """
        actual = os.path.getsize(self.filename)
        if actual > self._map_size:
            self.mapped_file.close()
            if self.read_only:
                self.mapped_file = mmap.mmap(
                    self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
                )
            else:
                self.mapped_file = mmap.mmap(self.file_handle.fileno(), 0)
            self._map_size = len(self.mapped_file)
        if min_end > self._map_size:
            raise ValueError(
                f"read past end of file ({min_end} > {self._map_size} bytes): "
                f"dangling reference — the record points at data that was "
                f"never written (or the file was truncated)"
            )

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
        self._ensure_writable()
        with open(self.log_filename, "ab") as log:
            for address, data in writes:
                self.log_write(log, address, data)
            self.log_commit(log)

        for address, data in writes:
            self.write(address, data)

        self.clear_log()

    def clear_log(self):
        self._ensure_writable()
        with open(self.log_filename, "wb") as log:
            pass

    # -------------------------------------------------------------------------
    # Header management methods (Phase 1)
    # -------------------------------------------------------------------------

    def _slot_offset(self, slot):
        """Get the mmap offset of a header slot (0=A, 1=B)."""
        return 1 + slot * self._slot_size

    def _peek_stored_header_size(self):
        """Read the stored header_size from slot A WITHOUT relying on the
        header_size we were constructed with.  Slot A always starts at byte
        offset 1.  Handles both the CRC format and the legacy one.  Returns
        int, or None for an uninitialised / torn / pre-header_size file."""
        mf = self.mapped_file
        if mf is None or len(mf) < 9:
            return None
        magic = bytes(mf[1:5])
        try:
            if magic == self._MAGIC:
                if len(mf) < 1 + self._SLOT_HDR:
                    return None
                _seqno, data_size, crc = struct.unpack("<QII", bytes(mf[5:21]))
                start = 1 + self._SLOT_HDR
                if data_size <= 0 or start + data_size > len(mf):
                    return None
                payload = bytes(mf[start : start + data_size])
                if zlib.crc32(payload) & 0xFFFFFFFF != crc:
                    return None
                data = pickle.loads(payload)
            elif magic == self._LEGACY_MAGIC:
                data_size = int(np.frombuffer(mf[5:9], dtype="uint32")[0])
                if data_size <= 0 or 9 + data_size > len(mf):
                    return None
                data = pickle.loads(bytes(mf[9 : 9 + data_size]))
            else:
                return None
        except Exception:
            return None
        v = data.get(self._header_size_key) if isinstance(data, dict) else None
        return int(v) if isinstance(v, int) and v > 0 else None

    def _read_slot(self, slot):
        """Deserialize a slot.  Returns ``(header_dict, seqno)`` or None.

        A slot whose CRC does not match its payload is a torn write and is
        rejected (returns None) so its intact sibling wins.  Legacy slots
        (``LOOM``, no CRC/seqno) are still read, tagged with seqno ``-1`` so
        any modern slot is preferred over them.
        """
        offset = self._slot_offset(slot)
        slot_bytes = bytes(self.mapped_file[offset : offset + self._slot_size])
        magic = slot_bytes[0:4]
        if magic == self._MAGIC:
            try:
                seqno, data_size, crc = struct.unpack("<QII", slot_bytes[4:20])
            except struct.error:
                return None
            if 0 < data_size <= self._slot_size - self._SLOT_HDR:
                payload = slot_bytes[self._SLOT_HDR : self._SLOT_HDR + data_size]
                if zlib.crc32(payload) & 0xFFFFFFFF == crc:
                    try:
                        return pickle.loads(payload), seqno
                    except Exception:
                        return None
            return None
        if magic == self._LEGACY_MAGIC:
            try:
                data_size = int(np.frombuffer(slot_bytes[4:8], dtype="uint32")[0])
                if 0 < data_size < self._slot_size - 8:
                    return pickle.loads(slot_bytes[8 : 8 + data_size]), -1
            except Exception:
                pass
        return None

    def _load_header(self):
        """Load header from the newest intact slot.

        Both slots are read and CRC-checked; the one with the highest seqno
        wins (a torn slot fails CRC → None → loses to its intact sibling).
        Ties, and the all-legacy case (seqno -1), fall back to the active-slot
        hint byte.  This can never surface a page-spliced header: the worst
        case is rolling back to an older but fully-consistent generation.
        """
        if len(self.mapped_file) < self.header_size:
            return

        active = self.mapped_file[0]
        if active not in (0, 1):
            active = 0  # Default to slot A

        candidates = []
        for slot in (0, 1):
            r = self._read_slot(slot)
            if r is not None:
                candidates.append((slot, r[0], r[1]))  # (slot, data, seqno)

        chosen = None
        if candidates:
            max_seq = max(seq for _, _, seq in candidates)
            best = [c for c in candidates if c[2] == max_seq]
            if len(best) == 1:
                chosen = best[0]
            else:
                # Tie (typically both legacy, seqno -1): honour the hint byte.
                chosen = next((c for c in best if c[0] == active), best[0])

        if chosen is None:
            if self.read_only:
                raise ReadOnlyError(
                    "Cannot initialize a missing header in read-only mode"
                )
            self._initialize_header()
            return

        self._header_data = chosen[1]
        self._header_seqno = max(0, chosen[2])
        # Gracefully upgrade pre-existing files: record the header_size so the
        # next save makes the file self-describing (persisted on next write).
        self._header_data.setdefault(self._header_size_key, self.header_size)
        self._header_dirty = False
        self._load_freelist()

    def _initialize_header(self):
        """Initialize a fresh header."""
        self._ensure_writable()
        self._header_data = {
            self._is_initialized_key: True,
            self._allocation_index_key: self.header_size,
            self._header_size_key: self.header_size,
        }
        # Write to both slots for a clean start (slot A gets the higher seqno
        # so the hint byte and the seqno agree).
        self.mapped_file[0] = 0  # Slot A active
        self._write_slot(1, 1)
        self._write_slot(0, 2)
        self._header_seqno = 2
        self.mapped_file.flush()
        self._header_dirty = False

    def _write_slot(self, slot, seqno):
        """Serialize current header data into a slot.

        Layout: ``[LOM2][seqno u64][size u32][crc32 u32][pickle]``.  The CRC
        covers the pickle payload so a partially-flushed slot is detectable;
        the seqno orders the two slots independently of the hint byte.
        """
        pickled_data = pickle.dumps(self._header_data)
        data_size = len(pickled_data)
        max_data = self._slot_size - self._SLOT_HDR

        if data_size > max_data:
            from loom.errors import HeaderTooLargeError

            raise HeaderTooLargeError(data_size, max_data)

        crc = zlib.crc32(pickled_data) & 0xFFFFFFFF
        # Marker + seqno + length + CRC + payload.  The reader keys off the
        # length and validates the CRC, so trailing bytes from a previous
        # (possibly larger) write are ignored — no need to zero-pad the slot.
        slot_bytes = (
            self._MAGIC + struct.pack("<QII", seqno, data_size, crc) + pickled_data
        )

        offset = self._slot_offset(slot)
        self.mapped_file[offset : offset + len(slot_bytes)] = slot_bytes

    def _save_header(self):
        """Persist header via double-buffer: write inactive slot, flip.

        flush=False (lazy): mmap pages are written by the OS dirty-page
        writeback (typically within seconds) and guaranteed on close().
        This avoids a ~10 ms msync() syscall on every allocation.

        Crash safety:
        - Both slots always contain a valid state (old or new).
        - On recovery, _load_header() reads the active slot or falls back
          to the other — no partial-write corruption possible.
        - Data durability is guaranteed by close() which calls flush().
        """
        assert self.mapped_file, "DB is not open. Call open() first."
        self._ensure_writable()

        active = self.mapped_file[0]
        if active not in (0, 1):
            active = 0
        inactive = 1 - active

        # Write to inactive slot with the next seqno, then flip the hint byte.
        # Correctness no longer rests on the flip landing after the slot: the
        # seqno + CRC let _load_header pick the newest intact slot on their own.
        self._header_seqno += 1
        self._write_slot(inactive, self._header_seqno)
        self.mapped_file[0] = inactive
        self._header_dirty = False
        self._write_count += 1

        # sync_writes=True: flush after every save (server mode, full durability)
        # sync_writes=False (default): flush only on close/remap (fast mode)
        if self.sync_writes:
            self.mapped_file.flush()

    def flush(self):
        """Force all pending mmap writes to disk immediately.

        Called automatically by close() and _remap().
        Call explicitly when you need immediate on-disk durability
        (e.g. before an expected hard shutdown).
        """
        if self.mapped_file and not self.read_only:
            if self._header_dirty:
                self._save_header()
            self.mapped_file.flush()

    def set_header_field(self, name, value):
        """Set a metadata field in the header.

        Persists immediately unless batch mode is active, in which case
        the write is deferred until end_batch().

        Args:
            name: Field name (string)
            value: Any picklable Python object
        """
        assert self.mapped_file, "DB is not open. Call open() first."
        self._ensure_writable()
        self._header_data[name] = value
        if self._batch_mode:
            self._header_dirty = True
        else:
            self._save_header()

    def begin_batch(self):
        """Enter batch mode: defer header writes until end_batch().

        Re-entrant: nested begin/end pairs are counted and the header is
        only persisted when the outermost batch exits."""
        self._ensure_writable()
        self._batch_depth += 1
        self._batch_mode = True

    def end_batch(self, save=True):
        """Exit batch mode and flush any pending header writes.

        save=False leaves a dirty header in memory instead of persisting it —
        the next unbatched set_header_field(), flush_header() or close() will
        write it.  Callers use this to amortise the header pickle over many
        small batches (crash window = auto_save_interval ops, the documented
        durability model) instead of paying it per operation."""
        if self._batch_depth > 0:
            self._batch_depth -= 1
        if self._batch_depth > 0:
            return  # still inside an outer batch — keep deferring
        self._batch_mode = False
        if save and self._header_dirty:
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
        self._ensure_writable()
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
        self._ensure_writable()
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

    def _ensure_writable(self):
        if self.read_only:
            raise ReadOnlyError()

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
