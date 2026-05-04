import json
from urllib.parse import quote


class LoomClientError(Exception):
    pass


class LoomHTTPError(LoomClientError):
    def __init__(self, status_code, message, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LoomNotFoundError(LoomHTTPError):
    pass


class LoomValidationError(LoomHTTPError):
    pass


class LoomConflictError(LoomHTTPError):
    pass


class _ResourceGroup:
    def __init__(self, client, resource_cls):
        self._client = client
        self._resource_cls = resource_cls

    def __getitem__(self, name):
        return self._resource_cls(self._client, name)


class _BaseRemoteResource:
    resource_path = ""

    def __init__(self, client, name):
        self._client = client
        self.name = name

    @property
    def _name_quoted(self):
        return quote(str(self.name), safe="")

    def _path(self, suffix=""):
        base = f"/{self.resource_path}/{self._name_quoted}"
        if suffix:
            return f"{base}/{suffix.lstrip('/')}"
        return base

    def info(self):
        return self._client._request("GET", self._path())


class RemoteDict(_BaseRemoteResource):
    resource_path = "dicts"

    def keys(self, limit=1000):
        return self._client._request("GET", self._path("keys"), params={"limit": limit})

    def get(self, key):
        return self._client._request(
            "GET", self._path(f"items/{quote(str(key), safe='')}")
        )

    def set(self, key, value):
        return self._client._request(
            "PUT", self._path(f"items/{quote(str(key), safe='')}"), json=value
        )

    def delete(self, key):
        return self._client._request(
            "DELETE", self._path(f"items/{quote(str(key), safe='')}")
        )

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __delitem__(self, key):
        self.delete(key)


class RemoteLRUDict(RemoteDict):
    resource_path = "lru_dicts"


class RemoteList(_BaseRemoteResource):
    resource_path = "lists"

    def slice(self, start=0, end=None):
        params = {"start": start}
        if end is not None:
            params["end"] = end
        return self._client._request("GET", self._path("items"), params=params)

    def append(self, item):
        return self._client._request("POST", self._path("items"), json=item)

    def get(self, index):
        return self._client._request("GET", self._path(f"items/{index}"))

    def set(self, index, item):
        return self._client._request("PUT", self._path(f"items/{index}"), json=item)

    def delete(self, index):
        return self._client._request("DELETE", self._path(f"items/{index}"))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(0 if index.start is None else index.start, index.stop)
        return self.get(index)

    def __setitem__(self, index, item):
        self.set(index, item)

    def __delitem__(self, index):
        self.delete(index)


class RemoteSet(_BaseRemoteResource):
    resource_path = "sets"

    def members(self, limit=1000):
        return self._client._request(
            "GET", self._path("members"), params={"limit": limit}
        )

    def add(self, item):
        return self._client._request("POST", self._path("members"), json={"item": item})

    def contains(self, item):
        body = self._client._request(
            "GET", self._path(f"contains/{quote(str(item), safe='')}")
        )
        return body["contains"]

    def remove(self, item):
        return self._client._request(
            "DELETE", self._path(f"members/{quote(str(item), safe='')}")
        )

    def __contains__(self, item):
        return self.contains(item)


class RemoteBTree(_BaseRemoteResource):
    resource_path = "btrees"

    def get(self, key):
        return self._client._request(
            "GET", self._path(f"items/{quote(str(key), safe='')}")
        )

    def set(self, key, value):
        return self._client._request(
            "PUT", self._path(f"items/{quote(str(key), safe='')}"), json=value
        )

    def delete(self, key):
        return self._client._request(
            "DELETE", self._path(f"items/{quote(str(key), safe='')}")
        )

    def range(self, start=None, end=None, limit=1000):
        params = {"limit": limit}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        return self._client._request("GET", self._path("range"), params=params)

    def prefix(self, prefix, limit=1000):
        return self._client._request(
            "GET",
            self._path(f"prefix/{quote(str(prefix), safe='')}"),
            params={"limit": limit},
        )

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __delitem__(self, key):
        self.delete(key)


class RemoteQueue(_BaseRemoteResource):
    resource_path = "queues"

    def peek(self):
        return self._client._request("GET", self._path("peek"))

    def push(self, item):
        return self._client._request("POST", self._path("push"), json=item)

    def pop(self):
        return self._client._request("POST", self._path("pop"))


class RemoteBloomFilter(_BaseRemoteResource):
    resource_path = "bloomfilters"

    def add(self, item):
        return self._client._request("POST", self._path("items"), json={"item": item})

    def contains(self, item):
        body = self._client._request(
            "GET", self._path(f"contains/{quote(str(item), safe='')}")
        )
        return body["contains"]

    def __contains__(self, item):
        return self.contains(item)


class RemoteCountingBloomFilter(RemoteBloomFilter):
    resource_path = "counting_bloomfilters"

    def remove(self, item):
        return self._client._request(
            "DELETE", self._path(f"items/{quote(str(item), safe='')}")
        )


class LoomClient:
    def __init__(self, base_url, session=None, timeout=30.0, headers=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or self._build_default_session(headers)
        self.dicts = _ResourceGroup(self, RemoteDict)
        self.lists = _ResourceGroup(self, RemoteList)
        self.sets = _ResourceGroup(self, RemoteSet)
        self.btrees = _ResourceGroup(self, RemoteBTree)
        self.queues = _ResourceGroup(self, RemoteQueue)
        self.bloomfilters = _ResourceGroup(self, RemoteBloomFilter)
        self.counting_bloomfilters = _ResourceGroup(self, RemoteCountingBloomFilter)

    @staticmethod
    def _build_default_session(headers=None):
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "The lightweight HTTP client requires `requests`. Install it with: pip install requests"
            ) from e

        session = requests.Session()
        if headers:
            session.headers.update(headers)
        return session

    def info(self):
        return self._request("GET", "/")

    @property
    def structures(self):
        body = self.info()
        return body.get("structures", {})

    def close(self):
        close = getattr(self._session, "close", None)
        if close is not None:
            close()

    def _request(self, method, path, params=None, json=None):
        url = f"{self.base_url}{path}"
        kwargs = {"params": params}
        if json is not None:
            kwargs["json"] = json
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        response = self._session.request(method, url, **kwargs)
        if 200 <= response.status_code < 300:
            if response.status_code == 204:
                return None
            text = response.text.strip()
            if not text:
                return None
            return response.json()
        self._raise_http_error(response)

    def _raise_http_error(self, response):
        message = f"HTTP {response.status_code}"
        body = None
        try:
            body = response.json()
        except Exception:
            text = getattr(response, "text", "")
            if text:
                body = text
        if isinstance(body, dict) and "detail" in body:
            detail = body["detail"]
            if isinstance(detail, (dict, list)):
                message = json.dumps(detail)
            else:
                message = str(detail)
        elif body is not None:
            message = str(body)

        error_cls = LoomHTTPError
        if response.status_code == 404:
            error_cls = LoomNotFoundError
        elif response.status_code == 409:
            error_cls = LoomConflictError
        elif response.status_code == 422:
            error_cls = LoomValidationError
        raise error_cls(response.status_code, message, body=body)
