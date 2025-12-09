from numpy import (array, dtype, frombuffer, int8, int32, int64, str_, uint32,
                   uint64)

from .exception import NotCompiledError

PREFIX_DTYPE = int8
blob_dt = dtype([("blob", uint32)])
integer = (int, uint64, int64, uint32, int32)


class Dataset:
    def __init__(self, identifier, db, name, dtypes, offset=0):
        self.name = name
        self._identifier = identifier
        self._dtypes = dtypes
        self._len = 0
        self._prefix = PREFIX_DTYPE(identifier).tobytes()
        self._prefix_size = len(self._prefix)
        self._offset = 0

        self._compile()

    def _remove_database_reference(self):
        if hasattr(self, "_read_at"):
            del self._read_at
        if hasattr(self, "_write_at"):
            del self._write_at
        if hasattr(self, "_db_appendNotCompiledError"):
            del self._db_append
        if hasattr(self, "_db_allocate"):
            del self._db_allocate
        if hasattr(self, "_db_append_blob"):
            del self._db_append_blob
        if hasattr(self, "_db_get_blob"):
            del self._db_get_blob

    def _add_database_reference(self, db):
        self._read_at = db._read_at
        self._write_at = db._write_at
        self._db_append = db._append
        self._db_allocate = db._allocate
        self._db_get_blob = db.get_blob
        self._db_append_blob = db.append_blob

    def _compile(self):
        self._blob_fields = set()
        self._string_fields = set()
        self._field = {}
        self._field_list = ["prefix"]

        position = self._prefix_size
        for i, (key, dt) in enumerate(self._dtypes, 1):
            dt_size = dt.itemsize
            self._field[key] = (i, dt.itemsize, position, dt)
            if dt != blob_dt:
                if dt.type is str_:  # faster that way
                    self._string_fields.add(key)
            else:
                self._blob_fields.add(key)
            position += dt_size
            self._field_list.append(key)
        self._len = position
        self._has_blob = len(self._blob_fields) != 0

    # -------------------------------------------------------------------------
    # encoding and decoding functions
    # -------------------------------------------------------------------------

    def _to_numpy(self, data):
        for f in self._string_fields:
            if f not in data:
                data[f] = ""
        if self._has_blob:
            for f in self._blob_fields:
                if f not in data:
                    data[f] = 0
                else:
                    data[f] = self._db_append_blob(data[f])
        res = tuple(data.get(key, 0) for key in self._field)
        return array(res, dtype=self._dtypes)

    def _to_bytes(self, data):
        if self._has_blob:
            # self.parse_blob(data)
            pass
        arr = self._to_numpy(data)
        return self._prefix + arr.tobytes()

    def _parse(self, res):
        res = frombuffer(res, dtype=self._dtypes)[0]
        if not self._has_blob:
            return dict(zip(self._field, res))

        res = dict(zip(self._field, res))
        for field in self._blob_fields:
            blob_id = res[field][0]
            if blob_id == 0:
                del res[field]
                continue
            res[field] = self._db_get_blob(blob_id)
        return res

    def _parse_with_prefix(self, res):
        _dtypes = [("prefix", PREFIX_DTYPE)] + self._dtypes
        res = frombuffer(res, dtype=_dtypes)

        f = self._field_list
        id_ = self._identifier
        data = []
        for r in res:
            tmp = {}
            for i, field in enumerate(f):
                if i != 0:
                    tmp[field] = r[i]
                else:
                    if r[0] == id_:
                        continue
                    else:
                        tmp = None
                        break
            data.append(tmp)
        return data

    def _parse_values(self, res, key):
        _dtypes = [("prefix", PREFIX_DTYPE)] + self._dtypes
        res = frombuffer(res, dtype=_dtypes)

        index, _, _, _ = self._field[key]
        f = self._field_list
        id_ = self._identifier
        data = []
        for r in res:
            if r[0] == id_:
                data.append(r[index])
            else:
                data.append(None)
        return data

    # -------------------------------------------------------------------------
    # overloading functions
    # -------------------------------------------------------------------------

    def __repr__(self):
        txt = []
        for key, dt in self._dtypes:
            txt.append(f"{key}={dt}")
        txt = ", ".join(txt)
        return self.name.capitalize() + "(" + txt + ")"

    def __len__(self):
        return self._len

    # -------------------------------------------------------------------------
    # setters and getters
    # -------------------------------------------------------------------------

    def _set_field_no_index(self, key, value):
        _, _, align, dt = self._field[key]
        data = array(value, dtype=dt)
        self._write_at(self._offset + align, data)

    def _get_field_no_index(self, key):
        _, dt_size, align, dt = self._field[key]
        data_bytes = self._read_at(self._offset + align, dt_size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    def _get_index_from(self, block_index, row_index):
        return int(block_index + row_index * self._len)

    def new_block(self, size):
        try:
            return self._db_allocate(self._len * size)
        except AttributeError:
            raise NotCompiledError(
                "Please compile the database first")

    def append(self, **data):
        return self._db_append(self._to_bytes(data))

    def get(self, block_index, row_index=0):
        index = self._get_index_from(block_index, row_index)
        data_bytes = self._read_at(index, 1)

        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self._identifier:
            raise KeyError
        data_bytes = self._read_at(index + self._prefix_size,
                                   self._len - self._prefix_size)
        return self._parse(data_bytes)

    def get_slice(self, block_index, s):
        start = s.start or 0
        stop = s.stop
        length = stop - start
        index = self._get_index_from(block_index, start)
        data_bytes = self._read_at(index,
                                   self._len * length)
        return self._parse_with_prefix(data_bytes)

    def get_slice_as_bytes(self, block_index, s):
        start = s.start or 0
        stop = s.stop
        length = stop - start
        index = self._get_index_from(block_index, start)
        data_bytes = self._read_at(index,
                                   self._len * length)
        return data_bytes

    def get_slice_values(self, block_index, s, field):
        start = s.start or 0
        stop = s.stop
        length = stop - start
        index = self._get_index_from(block_index, start)
        data_bytes = self._read_at(index,
                                   self._len * length)
        return self._parse_values(data_bytes, field)

    def get_value(self, block_index, row_index, key):
        _, dt_size, align, dt = self._field[key]
        index = self._get_index_from(block_index, row_index) + align
        data_bytes = self._read_at(index, dt_size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    def set_at_address(self, address, data):
        # only when appending data
        data_bytes = self._to_bytes(data)
        self._write_at(address, data_bytes)

    def set(self, block_index, row_index, data):
        try:
            data_bytes = self._to_bytes(data)
            index = self._get_index_from(block_index, row_index)
            self._write_at(index, data_bytes)
        except AttributeError:
            raise NotCompiledError(
                "Please compile the database first")

    def set_value(self, block_index, row_index, key, value):
        _, _, align, dt = self._field[key]
        index = self._get_index_from(block_index, row_index) + align

        data = array(value, dtype=dt).tobytes()
        self._write_at(index, data)

    def delete(self, block_index, row_index):
        index = self._get_index_from(block_index, row_index)
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self._identifier:
            raise KeyError
        new_identifier = array(-identifier, dtype="int8").tobytes()
        self._write_at(index, new_identifier)

    def exists(self, block_index, row_index):
        index = self._get_index_from(block_index, row_index)
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self._identifier:
            return False
        return True

    def status(self, block_index, row_index):
        index = self._get_index_from(block_index, row_index)
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier == self._identifier:
            return 1
        elif identifier == -self._identifier:
            return -1
        return 0

    def __getitem__(self, args):
        if isinstance(args, tuple):
            arg_len = len(args)
            if arg_len == 2:
                if isinstance(args[1], (int, uint64, uint32)):
                    return self.get(*args)
                return self.get_slice(*args)
            elif arg_len == 3:
                if isinstance(args[1], (int, uint64, uint32)):
                    return self.get_value(*args)
                return self.get_slice_values(*args)
        return self._get_field_no_index(args)

    def __setitem__(self, args, value):
        if isinstance(args, tuple):
            arg_len = len(args)
            if arg_len == 2:
                return self.set(*args, value)
            elif arg_len == 3:
                return self.set_value(*args, value)
        return self._set_field_no_index(args, value)


class Group(Dataset):
    def __init__(self, identifier, dataset, name, dtypes):
        self.name = name
        self._identifier = identifier
        self._dtypes = dtypes
        self._len = 0
        self._prefix = PREFIX_DTYPE(identifier).tobytes()
        self._prefix_size = len(self._prefix)
        self._compile()

        # dataset methods
        self._dataset = dataset
        self._dataset_len = len(dataset)
        self._dataset_to_bytes = dataset._to_bytes
        self._dataset_parse = dataset._parse
        self._dataset_identifier = dataset._identifier
        self._dataset_field = dataset._field
        self._dataset_get_index_from = dataset._get_index_from

    def new_block(self, size):
        block_id = self._db_allocate(self._len + self._dataset_len * size)
        self._write_at(block_id, self._prefix)
        return block_id

    def set_value(self, block_index, key, value):
        _, _, align, dt = self._field[key]
        index = int(block_index + align)

        data = array(value, dtype=dt).tobytes()
        self._write_at(index, data)

    def get_value(self, block_index, key):
        _, dt_size, align, dt = self._field[key]
        index = int(block_index + align)
        data_bytes = self._read_at(index, dt_size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    def set_data(self, block_index, row_index, data):
        data_bytes = self._dataset_to_bytes(data)
        index = int(self._dataset_get_index_from(
            block_index, row_index) + self._len)
        self._write_at(index, data_bytes)

    def set_data_value(self, block_index, row_index, key, value):
        _, _, align, dt = self._dataset_field[key]
        index = int(self._dataset_get_index_from(
            block_index, row_index) + align + self._len)
        data = array(value, dtype=dt).tobytes()
        self._write_at(index, data)

    def get_data(self, block_index, row_index):
        index = int(self._dataset_get_index_from(
            block_index, row_index) + self._len)
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self._dataset_identifier:
            raise KeyError
        data_bytes = self._read_at(index + self._prefix_size,
                                   self._dataset_len - self._prefix_size)
        return self._dataset_parse(data_bytes)

    def get_data_value(self, block_index, row_index, key):
        _, dt_size, align, dt = self._dataset_field[key]
        index = int(self._dataset_get_index_from(
            block_index, row_index) + self._len + align)
        data_bytes = self._read_at(index, dt_size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    def get(self, index):
        index = int(index)
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self._identifier:
            raise KeyError
        data_bytes = self._read_at(index + self._prefix_size,
                                   self._len - self._prefix_size)
        return self._parse(data_bytes)

    def status(self, block_index, row_index):
        index = self._dataset_get_index_from(
            block_index, row_index) + self._len
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier == self._dataset_identifier:
            return 1
        elif identifier == -self._dataset_identifier:
            return -1
        return 0

    def __getitem__(self, args):
        if isinstance(args, tuple):
            arg_len = len(args)
            if arg_len == 2:
                block_index, item_1 = args
                if isinstance(item_1, integer):
                    return self.get_data(block_index, item_1)
                elif isinstance(item_1, str):
                    return self.get_value(block_index, item_1)
                elif isinstance(item_1, slice):
                    return self._dataset.get_slice(block_index + self._len, item_1)
            elif arg_len == 3:

                block_index, item_1, field = args
                if isinstance(item_1, integer):
                    return self.get_data_value(block_index, item_1, field)
                elif isinstance(item_1, slice):
                    return self._dataset.get_slice_values(
                        block_index + self._len, item_1, field)
        else:
            return self.get(args)

    def __setitem__(self, args, value):
        if isinstance(args, tuple):
            arg_len = len(args)
            if arg_len == 2:
                block_index, item_1 = args
                if isinstance(item_1, int):
                    return self.set_data(block_index, item_1, value)
                elif isinstance(item_1, str):
                    return self.set_value(block_index, item_1, value)
            elif arg_len == 3:
                return self.set_data_value(*args, value)


class Array(Dataset):
    def __init__(self, identifier, db, name, dt):
        self.name = name
        self._identifier = identifier
        self._dtype = dtype(dt)
        self._dtypes = self._dtype
        self._dt_size = self._dtype.itemsize
        self._prefix = PREFIX_DTYPE(identifier).tobytes()
        self._prefix_size = len(self._prefix)
        self._len = self._dt_size

        # db methods
        self._db_allocate = db._allocate
        self._write_at = db._write_at
        self._read_at = db._read_at

    def new_block(self, size):
        return self._db_allocate(self._dt_size * size + self._prefix_size)

    def set_value(self, block_index, index, value):
        index = int(block_index + self._dt_size * index + self._prefix_size)
        data = array(value, dtype=self._dtype).tobytes()
        self._write_at(index, data)

    def get_value(self, block_index, index):
        index = int(block_index + self._dt_size * index + self._prefix_size)
        data_bytes = self._read_at(index, self._dt_size)
        res = frombuffer(data_bytes, dtype=self._dtype)[0]
        return res

    def get_values(self, block_index, start, end):
        start = int(block_index + self._dt_size * start + self._prefix_size)
        end = int(block_index + self._dt_size * end + self._prefix_size)
        data_bytes = self._read_at(start, end-start)
        res = frombuffer(data_bytes, dtype=self._dtype)
        return res

    def __getitem__(self, args):
        block_index, position = args
        if isinstance(position, int):
            return self.get_value(block_index, position)
        elif isinstance(position, slice):
            start = position.start or 0
            end = position.stop
            assert end is not None
            return self.get_values(block_index, start, end)

    def __setitem__(self, args, value):
        return self.set_value(*args, value)

    def __repr__(self):
        txt = str(self._dtypes)
        return self.name.capitalize() + "(" + txt + ")"


class BoolArray(Dataset):
    def __init__(self, identifier, db, name):
        self.name = name
        self._identifier = identifier
        self._byte_to_bool = {
            b'\x01': True,
            b'\x00': False
        }
        self._bool_to_byte = {
            True: b'\x01',
            False: b'\x00'
        }
        self._prefix = PREFIX_DTYPE(identifier).tobytes()
        self._prefix_size = len(self._prefix)
        self._len = 1

        # db methods
        self._db_allocate = db._allocate
        self._write_at = db._write_at
        self._read_at = db._read_at

    def new_block(self, size):
        return self._db_allocate(size + self._prefix_size)

    def set_value(self, block_index, index, value):
        index = int(block_index + index + self._prefix_size)
        self._write_at(index, self._bool_to_byte[value])

    def get_value(self, block_index, index):
        index = int(block_index + index + self._prefix_size)
        data_bytes = self._read_at(index, 1)
        return self._byte_to_bool[data_bytes]

    def get_values(self, block_index, start, end):
        index = int(block_index + start + self._prefix_size)
        data_bytes = self._read_at(index, end-start)
        return list(data_bytes)

    def __getitem__(self, args):
        block_index, position = args
        if isinstance(position, int):
            return self.get_value(block_index, position)
        elif isinstance(position, slice):
            start = position.start or 0
            end = position.stop
            assert end is not None
            return self.get_values(block_index, start, end)

    def __setitem__(self, args, value):
        return self.set_value(*args, value)

    def __repr__(self):
        txt = str(self._dtypes)
        return self.name.capitalize() + "(" + txt + ")"
