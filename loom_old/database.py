import os
import pickle

from numpy import array, dtype, frombuffer, int8, uint32, ceil

from .dataset import Array, BoolArray, Dataset, Group, blob_dt
from .exception import DatasetExistsError, HeaderExistsError
from .datastructure.array import List

STEP_SIZE = 4096


class DB:
    def __init__(
        self,
        filename,
        flag="w",
        blob_protocol="pickle",
        blob_compression=None,
    ):
        """
        (str) filename: string name of the database file
        (str) flag: 'w' for write, 'r' for read and 'n' for new
        (str) blob_protocol: protocol defining encoding and decoding functions
        """
        self.filename = filename
        self._get_encoder_and_decoder(blob_protocol, blob_compression)

        self._id2size = {}
        self._id2dataset = {}

        # references to header, datasets and datastructures
        self.header = None
        self.datasets = {}
        self.datastructures = {}
        self.commit = True

        # blob management variables
        self._block_size = 4096*4
        self._slot_size = 32
        self._n_slots_per_block = self._block_size // self._slot_size

        # internal list of header fields
        self._header_fields = {"_index": dtype("uint64")}

        # index of read/write head
        self._index = None
        self._blob_identifier = int8(1).tobytes()

        # open file
        self.flag = flag
        if flag == "n" and os.path.exists(filename):
            os.remove(filename)
        self.open()

    def begin_transaction(self):
        self.commit = False

    def end_transaction(self):
        self.commit = True
        self.f.flush()

    def open(self):
        # create new file if needed, else open in rb+ mode
        if self._is_new_database():
            self.f = open(self.filename, "wb+")
        elif self.flag != "r":
            self.f = open(self.filename, "rb+")
            self._load()
        else:
            self.f = open(self.filename, "rb")
            self._load()

    def _get_encoder_and_decoder(self, blob_protocol, blob_compression):
        # available compressions
        if blob_compression is None:
            def compress(x): return x
            def decompress(x): return x
        elif blob_compression == "zlib":
            import zlib
            compress = zlib.compress
            decompress = zlib.decompress
        elif blob_compression == "brotli":
            import brotli
            compress = brotli.compress
            decompress = brotli.decompress
        else:
            raise ValueError(f"Compression {blob_compression} unknown")

        # available protocols to transform python objects to bytes
        if blob_protocol == "pickle":
            def dumps(x): return pickle.dumps(x,
                                              protocol=pickle.HIGHEST_PROTOCOL)

            def loads(x): return pickle.loads(x)
        elif blob_protocol == "orjson":
            import orjson
            dumps = orjson.dumps
            loads = orjson.loads
        elif blob_protocol == "ujson":
            import ujson
            dumps = ujson.dumps
            loads = ujson.loads
        elif blob_protocol == "cbor":
            import cbor2
            dumps = cbor2.dumps
            loads = cbor2.loads

        self.encode = lambda x: compress(dumps(x))
        self.decode = lambda x: loads(decompress(x))

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------

    @property
    def file_size(self):
        return os.stat(self.filename).st_size

    @property
    def capacity(self):
        return self.file_size - self.table_start

    @property
    def index(self):
        if self._index is None:
            res = self.header._get_field_no_index("_index")
            self._index = res
            return res
        return self._index

    @index.setter
    def index(self, value):
        value = int(value)
        self.header._set_field_no_index("_index", value)
        self._index = value

    @property
    def _n_empty_slots(self):
        fs = os.stat(self.filename).st_size
        return fs - self.index
    
    @property
    def _blocks_list(self):
        return self.datastructures["_blocks_list"]

    # -------------------------------------------------------------------------
    # datasets and header management
    # -------------------------------------------------------------------------

    def _initialize(self):
        for datastructure in self.datastructures.values():
            datastructure._initialize()

    def _remove_database_reference(self):
        self.header._remove_database_reference()
        for name in self.datasets:
            self.datasets[name]._remove_database_reference()

    def _add_database_reference(self):
        self.header._add_database_reference(self)
        for dataset in self.datasets.values():
            dataset._add_database_reference(self)
        for datastructure in self.datastructures.values():
            datastructure._add_database_reference(self)

    def _compile_datastructures(self):
        for dstruct in self.datastructures.values():
            dstruct._compile(self)

    def _dump(self):
        # complete headers instructions from other datastructures
        self._compile_datastructures()

        # build header
        self.header = Dataset(1, self, "header",
                              list(self._header_fields.items()))

        # remove database reference in datasets for pickle
        self._remove_database_reference()

        # dump header and datasets data
        data_bytes = pickle.dumps({"header": self.header,
                                   "datasets": self.datasets,
                                   "datastructures": self.datastructures},
                                  protocol=pickle.HIGHEST_PROTOCOL)
        data_len_bytes = array(len(data_bytes), dtype=uint32).tobytes()
        pickle_bytes = data_len_bytes + data_bytes
        pickle_bytes_len = len(pickle_bytes)

        # extend file and write on file
        self._extend_file(pickle_bytes_len)
        self._write_at(0, pickle_bytes)

        # bring back database reference in datasets
        self._add_database_reference()

        # initialize heads
        self._extend_file(len(self.header))
        self.header._offset = pickle_bytes_len
        self.table_start = self.header._offset + len(self.header)
        self.index = self.file_size

    def _load(self):
        self.f.seek(0)
        data_len = frombuffer(self.f.read(4), dtype=uint32)[0]
        data = pickle.loads(self.f.read(data_len))
        # grab values if not already done
        if self.header is None:
            self.header = data["header"]
        if len(self.datasets) == 0:
            self.datasets = data["datasets"]
        if len(self.datastructures) == 0:
            self.datastructures = data["datastructures"]

        # bring back database reference in datasets
        self._add_database_reference()

        # initialize heads
        self.header._offset = data_len + 4
        self.table_start = self.header._offset + len(self.header)

        # initialize datastructures
        for dstruct in self.datastructures.values():
            dstruct._load()

        self._slots = [self._blocks_list[i]
                       for i in range(len(self._blocks_list))]

    def create_dataset(self, name, **kwargs):
        if name in self.datasets:
            raise DatasetExistsError(
                f"A dataset named '{name}' already exists")

        identifier = len(self.datasets) + 3
        dtypes = []
        for key, dt in kwargs.items():
            if not isinstance(dt, str) or dt != "blob":
                dtypes.append((key, dtype(dt)))
            else:
                dtypes.append((key, blob_dt))

        dset = Dataset(identifier, self, name, dtypes)
        self.datasets[name] = dset
        self._id2size[identifier] = len(dset)
        self._id2dataset[identifier] = dset
        return dset

    def create_group(self, name, dataset, **kwargs):
        if name in self.datasets:
            raise DatasetExistsError(
                f"A dataset named '{name}' already exists")

        identifier = len(self.datasets) + 3
        dtypes = []
        for key, dt in kwargs.items():
            if not isinstance(dt, str) or dt != "blob":
                dtypes.append((key, dtype(dt)))
            else:
                dtypes.append((key, blob_dt))

        dset = Group(identifier, dataset, name, dtypes)
        self.datasets[name] = dset
        self._id2size[identifier] = len(dset)
        self._id2dataset[identifier] = dset
        return dset

    def create_array(self, name, dt):
        if name in self.datasets:
            raise DatasetExistsError(
                f"A dataset named '{name}' already exists")

        identifier = len(self.datasets) + 3
        if dt == "bool":
            dset = BoolArray(identifier, self, name)
        else:
            dset = Array(identifier, self, name, dt)
        self.datasets[name] = dset
        self._id2size[identifier] = len(dset)
        self._id2dataset[identifier] = dset
        return dset

    def create_header(self, **fields):
        if self.file_size != 0:
            raise HeaderExistsError("Header already exists")

        if "_index" not in self._header_fields:
            self._header_fields["_index"] = dtype("uint64")

        for field, dt in fields.items():
            self._header_fields[field] = dtype(dt)

    def create_datastructure(self, name, dstruct):
        self.datastructures[name] = dstruct
        return dstruct

    # -------------------------------------------------------------------------
    # overloading methods
    # -------------------------------------------------------------------------

    def __getitem__(self, key):
        try:
            return self.datasets[key]
        except KeyError:
            return self.datastructures[key]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.compile()

    def compile(self):
        # create block list for blob management
        _blocks = self.create_dataset(
            "blocks", position="uint64", slots_taken="uint8")
        self.create_datastructure("_blocks_list", List(_blocks))

        self._dump()
        self.close()
        self.open()

    # -------------------------------------------------------------------------
    # blob manipulation methods
    # -------------------------------------------------------------------------

    def _append(self, data_bytes):
        data_size = len(data_bytes)
        n_slots = int(ceil(data_size / self._slot_size))
        assert data_size <= self._block_size

        if len(self._slots) == 0:
            # allocate a new block
            index = self._allocate(self._block_size)
            self._write_at(index, data_bytes)

            slot = {"position": index, "slots_taken": n_slots}
            self._slots = [slot]
            self._blocks_list.append(slot)
        else:
            _slot_found = False
            for i, slot in enumerate(self._slots):
                slots_in_block = self._n_slots_per_block + i
                if slots_in_block - slot["slots_taken"] < n_slots:
                    continue
                _slot_found = True
                index = slot["position"] + self._slot_size * slot["slots_taken"]

                self._write_at(index, data_bytes)
                slot["slots_taken"] += n_slots
                self._blocks_list[i] = slot
            if not _slot_found:
                # allocate a new block
                slots_in_block = self._n_slots_per_block + len(self._slots)
                index = self._allocate(self._slot_size * slots_in_block)
                self._write_at(index, data_bytes)

                slot = {"position": index, "slots_taken": n_slots}
                self._slots.append(slot)
                self._blocks_list.append(slot)
        return index

    def append_blob(self, blob):
        blob_bytes = self.encode(blob)
        blob_size = len(blob_bytes)

        data_bytes = b''.join((
            self._blob_identifier,  # 1 is blob identifier
            uint32(blob_size).tobytes(),  # blob size in uint32
            blob_bytes  # actual content of the blob
        ))
        return self._append(data_bytes)

    def get_blob(self, index):
        byte_index = index + 1
        size = frombuffer(self._read_at(byte_index, 4), dtype=uint32)[0]
        blob_bytes = self.f.read(size)
        return self.decode(blob_bytes)

    # -------------------------------------------------------------------------
    # file IO management methods
    # -------------------------------------------------------------------------

    def _allocate(self, bytes_size):
        # returns start and end indices of allocated data
        self._extend_file(bytes_size)
        start = self.index
        self.index += bytes_size
        return start

    def _is_new_database(self):
        return not os.path.exists(self.filename) or self.file_size == 0

    def _extend_file(self, bytes_size):
        self.f.truncate(self.file_size + bytes_size)

    def _write_at(self, index, data):
        self.f.seek(index)
        self.f.write(data)
        if self.commit:
            self.f.flush()

    def _read_at(self, start, size):
        self.f.seek(start)
        return self.f.read(size)

    def close(self):
        self.f.close()

    def __del__(self):
        self.close()


class Dict:
    def __init__(
        self, filename,
        flag="w",
        dtype="blob",
        use_hash=True,
        max_key_len=30,
        blob_compression=None,
        blob_protocol="pickle",
        cache_len=100000,
        lru_cache=10000,
        p_init=16,
        probe_factor=.4
    ):
        from .datastructure import Hashmap
        if not os.path.exists(filename) or flag == "n":
            self.db = DB(filename, flag=flag,
                         blob_compression=blob_compression,
                         blob_protocol=blob_protocol)
            self.db.create_header(max_key_len="uint8", use_hash="bool")
            if use_hash:
                self.data = self.db.create_dataset(
                    "data", key="uint64", value=dtype)
            else:
                self.data = self.db.create_dataset(
                    "data", key=f"U{max_key_len}", value=dtype)
            self.table = self.db.create_datastructure(
                "table", Hashmap(
                    self.data, "key",
                    cache_len=cache_len,
                    p_init=p_init,
                    probe_factor=probe_factor))
            self.db.compile()

            # initialize values
            self.db.header["max_key_len"] = max_key_len
            self.db.header["use_hash"] = use_hash
        else:
            self.db = DB(filename, flag=flag,
                         blob_compression=blob_compression,
                         blob_protocol=blob_protocol)
            self.data = self.db["data"]
            self.table = self.db["table"]
            max_key_len = self.db.header["max_key_len"]
            use_hash = self.db.header["use_hash"]

        self.max_key_len = max_key_len
        self.use_hash = use_hash
        self.lru_cache = lru_cache
        if lru_cache > 0:
            from lru import LRU
            self.lru = LRU(lru_cache)

    def _hash(self, key, seed=0):
        import mmh3
        if not isinstance(key, str):
            key = str(key)
        return mmh3.hash(key, seed=seed, signed=False)

    def __setitem__(self, key, value):
        if self.lru_cache > 0:
            self.lru[key] = value

        if self.use_hash:
            key_hash = self._hash(key)
            self.table[key_hash] = {"value": value}
        else:
            self.table[key] = {"value": value}

    def __getitem__(self, key):
        if self.lru_cache > 0:
            val = self.lru.get(key)
            if val is not None:
                return val

        if self.use_hash:
            key_hash = self._hash(key)
            return self.table[key_hash]["value"]
        else:
            return self.table[key]["value"]

    def __delitem__(self, key):
        if self.use_hash:
            key = self._hash(key)
        del self.table[key]
