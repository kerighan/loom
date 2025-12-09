import mmh3


class DataStructure:
    def _get_header_fields(self):
        return {}

    def _remove_database_reference(self):
        if hasattr(self, "_db"):
            del self._db

    def _add_database_reference(self, db):
        self._db = db

    def _hash(self, key, seed=0):
        if not isinstance(key, str):
            key = str(key)
        return mmh3.hash(key, seed=seed, signed=False)

    def get(self, key, res=None):
        try:
            return self.lookup(key)
        except KeyError:
            return res

    def __contains__(self, key):
        return self.contains(key)

    def __getitem__(self, key):
        return self.lookup(key)

    def __setitem__(self, key, data):
        if hasattr(self, "key"):
            data[self.key] = key
            self.insert(data)
        else:
            self.insert(key, data)

    def __delitem__(self, key):
        self.delete(key)
