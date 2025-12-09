from numpy import dtype
import numpy as np
from .base import DataStructure


class List(DataStructure):
    def __init__(
        self, dataset, start_size=512, growth_factor=1.33
    ):
        # hashtable parametrization
        self.growth_factor = growth_factor
        self.start_size = start_size

        self.dataset = dataset
        self.dstruct_name = f"{dataset.name}_list"
        self._addr_name = f"{self.dstruct_name}_addr"
        self._index_name = f"{self.dstruct_name}_index"
        self._tables_id_name = f"{self.dstruct_name}_tables_id"

        self._table_sizes = [0 for _ in range(64)]
        for i in range(64):
            if i == 0:
                self._table_sizes[0] = start_size - 1
            else:
                self._table_sizes[i] = int(
                    self._table_sizes[i - 1] * growth_factor) - 1

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------

    def _compile(self, db):
        db._header_fields[self._addr_name] = dtype("uint64")
        db._header_fields[self._index_name] = dtype("uint64")
        self._tables_pos = db.create_array(
            self._tables_id_name, "uint64")

    def _load(self):
        self._tables_addr = self._db.header[self._addr_name]
        self._index = self._db.header[self._index_name]
        self._tables_pos = self._db[self._tables_id_name]

        # if not yet initialized
        if self._tables_addr == 0:
            self._tables_addr = self._tables_pos.new_block(64)
            self._db.header[self._addr_name] = self._tables_addr
            self._db.header[self._index_name] = 0

            table_id = self.dataset.new_block(self.start_size)
            self._tables_pos.set_value(self._tables_addr, 0, table_id)

        self._load_tables_id()

    # -------------------------------------------------------------------------
    # refresh functions
    # -------------------------------------------------------------------------

    def _save_tables_id(self, index, table_id):
        self._tables_pos.set_value(self._tables_addr, index, table_id)

    def _load_tables_id(self):
        self.tables_id = list(
            self._tables_pos.get_values(self._tables_addr, 0, 64))

    # -------------------------------------------------------------------------
    # public functions
    # -------------------------------------------------------------------------

    def __setitem__(self, position, value):
        self.insert(position, value)

    def insert(self, position, value):
        index = self._db.header[self._index_name]
        if position >= index:
            raise IndexError("list index out of range")

        table_number = np.searchsorted(self._table_sizes, position)
        if table_number != 0:
            table_index = int(
                position - self._table_sizes[table_number - 1] - 1)
        else:
            table_index = position
        table_id = self.tables_id[table_number]
        return self.dataset.set(table_id, table_index, value)

    def append(self, value):
        index = self._db.header[self._index_name]
        table_number = np.searchsorted(self._table_sizes, index)
        table_size = self._table_sizes[table_number]
        if table_size == index:
            new_table_size = self._table_sizes[table_number + 1] - table_size
            table_id = self.dataset.new_block(new_table_size)
            self._save_tables_id(table_number + 1, table_id)
            self._load_tables_id()

        if table_number != 0:
            table_index = int(index - self._table_sizes[table_number - 1] - 1)
        else:
            table_index = index
        table_id = self.tables_id[table_number]
        self.dataset.set(table_id, table_index, value)
        self._db.header[self._index_name] = index + 1

    def lookup(self, position):
        if isinstance(position, int):
            return self.lookup_single_value(position)
        elif isinstance(position, slice):
            start = position.start
            stop = position.stop
            step = position.step
            if start is None:
                start = 0
            if stop is None:
                stop = self._db.header[self._index_name]
            if step is None:
                step = 1
            return [self.lookup_single_value(i)
                    for i in range(start, stop, step)]
        elif isinstance(position, (list, np.ndarray)):
            return [self.lookup_single_value(i) for i in position]

    def lookup_single_value(self, position):
        index = self._db.header[self._index_name]
        if position >= index:
            raise IndexError("list index out of range")

        table_number = np.searchsorted(self._table_sizes, position)
        if table_number != 0:
            table_index = int(
                position - self._table_sizes[table_number - 1] - 1)
        else:
            table_index = position
        table_id = self.tables_id[table_number]
        return self.dataset.get(table_id, table_index)

    def __len__(self):
        return self._db.header[self._index_name]
