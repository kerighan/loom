"""
Vector similarity search indexes for loom.

FlatIndex — Exact brute-force search, no training required.
  Best for < 100K vectors.
  O(n·d) per query (one sequential mmap read + numpy matmul).

IVFIndex  — Approximate search using Inverted File Index.
  Optional Product Quantization (pq=True) for 10–100× storage compression.
  Requires a training step on a representative sample.
  Best for > 100K vectors.
  O(nprobe · cell_size · [d or M]) per query.

Usage:
    with DB("embeddings.db") as db:
        # Exact search
        idx = db.create_flat_index("passages", dim=1536)
        idx.add("doc_1", embedding_array)
        results = idx.search(query, k=10)
        # → [("doc_1", 0.95), ("doc_2", 0.87), ...]

        # Approximate (IVF + optional PQ)
        ivf = db.create_ivf_index("passages", dim=1536,
                                   n_clusters=256, pq=True, n_sub=16)
        ivf.train(matrix_of_sample_vectors)
        ivf.add_batch([("doc_1", v1), ("doc_2", v2)])
        results = ivf.search(query, k=10, nprobe=32)

Metrics: "cosine" (default), "l2", "dot"
  cosine: vectors are L2-normalised at insertion; search = dot product.
  l2:     Euclidean distance (negated for consistent top-k logic).
  dot:    Raw inner product.
"""

from __future__ import annotations

import numpy as np

from loom.datastructures.base import DataStructure


# ── helpers ───────────────────────────────────────────────────────────────


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def _kmeans(
    vecs: np.ndarray, k: int, n_iter: int = 25, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Simple numpy Lloyd's k-means.  k is capped to len(vecs)."""
    k   = min(k, len(vecs))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(vecs), k, replace=False)
    centroids = vecs[idx].copy().astype(np.float32)

    for _ in range(n_iter):
        diff   = vecs[:, None, :] - centroids[None, :, :]    # (n, k, d)
        dists  = np.sum(diff * diff, axis=2)                  # (n, k)
        assign = np.argmin(dists, axis=1)                     # (n,)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = vecs[assign == j]
            new_centroids[j] = members.mean(axis=0) if len(members) else centroids[j]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids, assign


def _train_pq(
    vecs: np.ndarray, n_sub: int, n_bits: int = 8, seed: int = 42
) -> np.ndarray:
    """Train PQ codebooks.  Returns array of shape (n_sub, k_actual, sub_dim).

    k_actual = min(2**n_bits, len(vecs)).
    """
    dim     = vecs.shape[1]
    sub_dim = dim // n_sub
    k_sub   = min(2 ** n_bits, len(vecs))   # cap to available samples
    codebooks_list = []
    for m in range(n_sub):
        sub   = vecs[:, m * sub_dim : (m + 1) * sub_dim]
        cents, _ = _kmeans(sub, k_sub, seed=seed + m)
        codebooks_list.append(cents)
    # All sub-kmeans return the same k (capped equally)
    return np.array(codebooks_list, dtype=np.float32)  # (n_sub, k, sub_dim)


def _encode_pq(vecs: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Encode vectors to PQ codes (uint8).  Returns (n, n_sub)."""
    n_sub, k_sub, sub_dim = codebooks.shape
    codes = np.empty((len(vecs), n_sub), dtype=np.uint8)
    for m in range(n_sub):
        sub = vecs[:, m * sub_dim : (m + 1) * sub_dim]
        diff = sub[:, None, :] - codebooks[m][None, :, :]    # (n, k, sub_d)
        dists = np.sum(diff * diff, axis=2)                   # (n, k)
        codes[:, m] = np.argmin(dists, axis=1)
    return codes


def _adc_table(query: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Precompute ADC distance table.  Returns (n_sub, k_sub) L2² distances."""
    n_sub, k_sub, sub_dim = codebooks.shape
    table = np.empty((n_sub, k_sub), dtype=np.float32)
    for m in range(n_sub):
        q_sub = query[m * sub_dim : (m + 1) * sub_dim]
        diff = q_sub[None, :] - codebooks[m]                 # (k_sub, sub_d)
        table[m] = np.sum(diff * diff, axis=1)
    return table


def _adc_distances(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Resolve PQ codes to approximate L2² distances via the ADC table."""
    # codes: (n, n_sub) uint8;  table: (n_sub, k_sub)
    n_sub = codes.shape[1]
    dists = np.zeros(len(codes), dtype=np.float32)
    for m in range(n_sub):
        dists += table[m][codes[:, m]]
    return dists


def _topk(scores: np.ndarray, k: int, higher_is_better: bool = True):
    """Return (indices, scores) of top-k entries."""
    k = min(k, len(scores))
    if higher_is_better:
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
    else:
        idx = np.argpartition(scores, k)[:k]
        idx = idx[np.argsort(scores[idx])]
    return idx, scores[idx]


# ── base ─────────────────────────────────────────────────────────────────


class VectorIndex(DataStructure):
    """Shared plumbing for FlatIndex and IVFIndex."""

    _outer_types_supported = ()
    _inner_types_supported = ()

    def __init__(
        self,
        name: str,
        db,
        dim: int = 0,
        metric: str = "cosine",
        auto_save_interval=None,
        _parent=None,
    ):
        self.dim    = dim
        self.metric = metric

        super().__init__(name, db, auto_save_interval, _parent=_parent)

        meta = self._load_metadata()
        if meta:
            self._load()
        elif dim > 0:
            self._initialize()

    # ── Distance ────────────────────────────────────────────────────────

    def _prep_query(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32).ravel()
        if self.metric == "cosine":
            q = _normalize(q[None])[0]
        return q

    def _prep_vec(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).ravel()
        if self.metric == "cosine":
            v = _normalize(v[None])[0]
        return v

    def _scores(self, query: np.ndarray, vecs: np.ndarray) -> np.ndarray:
        """(n,) scores — higher is better for cosine/dot, lower (negated) for l2."""
        if self.metric in ("cosine", "dot"):
            return vecs @ query                            # (n,)
        # l2: return negative squared distances → highest = nearest
        diff = vecs - query[None, :]
        return -np.sum(diff * diff, axis=1)

    # ── ID store helpers ─────────────────────────────────────────────────

    def _ensure_id_structures(self, n_vecs: int = 0):
        """Create the ID ↔ slot mapping datasets (if not already done)."""
        pass  # subclasses set up in _initialize

    # ── ABC stubs — concrete subclasses override ─────────────────────────

    def add(self, vector_id: str, vector: np.ndarray) -> None:
        raise NotImplementedError

    def add_batch(self, items, show_progress: bool = False) -> None:
        for vid, vec in items:
            self.add(vid, vec)

    def remove(self, vector_id: str) -> None:
        raise NotImplementedError

    def search(
        self, query: np.ndarray, k: int = 10, **kwargs
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    # ── Registry / ABC ───────────────────────────────────────────────────

    def _get_registry_params(self):
        return {
            "dim":    self.dim,
            "metric": self.metric,
            "kind":   type(self).__name__,
        }

    def save(self, force=False):
        pass  # subclasses call _save_state()


# ── FlatIndex ─────────────────────────────────────────────────────────────


class FlatIndex(VectorIndex):
    """Exact brute-force nearest-neighbour search.

    All vectors are stored in a contiguous mmap block.  Search is a single
    sequential read followed by a vectorised dot-product / distance.

    Best for up to ~100K vectors; after that consider IVFIndex.
    """

    def _initialize(self):
        # Vectors: one float32[dim] per slot + prefix byte
        self._vecs_ds = self._db.create_dataset(
            f"_vidx_{self.name}_vecs", vec=f"float32[{self.dim}]"
        )
        # String IDs: stored alongside as text
        self._ids_ds = self._db.create_dataset(
            f"_vidx_{self.name}_ids", ext_id="text"
        )
        # Reverse mapping: ext_id → slot address
        addr_ds = self._db.create_dataset(
            f"_vidx_{self.name}_addrs", addr="uint64"
        )
        from loom.datastructures.dict import Dict
        self._rev = Dict(
            f"_vidx_{self.name}_rev", self._db, addr_ds,
            cache_size=2000, use_bloom=False, store_key=False,
        )

        # Pre-allocate a block for the vectors + ids
        cap = 4096
        self._block_addr     = int(self._vecs_ds.allocate_block(cap))
        self._ids_block_addr = int(self._ids_ds.allocate_block(cap))
        self._capacity       = cap
        self._n_vecs         = 0
        self._n_deleted      = 0
        self._save_state()

    def _load(self):
        meta = self._load_metadata()
        self.dim    = meta["dim"]
        self.metric = meta["metric"]
        self._vecs_ds        = self._db.get_dataset(meta["vecs_ds"])
        self._ids_ds         = self._db.get_dataset(meta["ids_ds"])
        self._block_addr     = meta["block_addr"]
        self._ids_block_addr = meta["ids_block_addr"]
        self._capacity       = meta["capacity"]
        self._n_vecs         = meta["n_vecs"]
        self._n_deleted      = meta["n_deleted"]
        # rev Dict loaded lazily
        self._rev_name = f"_vidx_{self.name}_rev"
        self._rev      = None

    def _ensure_rev(self):
        if self._rev is None:
            self._rev = self._db._datastructures.get(self._rev_name)

    def _save_state(self):
        self._save_metadata({
            "dim":           self.dim,
            "metric":        self.metric,
            "vecs_ds":       self._vecs_ds.name,
            "ids_ds":        self._ids_ds.name,
            "block_addr":    self._block_addr,
            "ids_block_addr":self._ids_block_addr,
            "capacity":      self._capacity,
            "n_vecs":        self._n_vecs,
            "n_deleted":     self._n_deleted,
        })

    def save(self, force=False):
        self._save_state()

    def _grow(self):
        """Double capacity when the pre-allocated block is full."""
        new_cap  = self._capacity * 2
        new_vecs = int(self._vecs_ds.allocate_block(new_cap))
        new_ids  = int(self._ids_ds.allocate_block(new_cap))
        # Copy existing data
        rs_v = self._vecs_ds.record_size
        rs_i = self._ids_ds.record_size
        old_data_v = self._vecs_ds.db.read(self._block_addr,     self._n_vecs * rs_v)
        old_data_i = self._ids_ds.db.read(self._ids_block_addr,  self._n_vecs * rs_i)
        self._vecs_ds.db.write(new_vecs, old_data_v)
        self._ids_ds.db.write(new_ids,   old_data_i)
        # Update all rev entries to new addresses
        # (addresses = block_addr + slot * rs; delta = new_block - old_block)
        # Easiest: re-build rev from scratch (n_vecs iterations)
        delta_v = new_vecs - self._block_addr
        self._ensure_rev()
        for slot in range(self._n_vecs):
            old_addr = self._block_addr + slot * rs_v
            entry = self._rev.get(self._ids_ds.read(
                self._ids_block_addr + slot * rs_i)["ext_id"])
            # simpler: just store slot index, not address
        # Restart simpler: store slot INDEX not address
        # → handled below by using slot-based storage
        self._block_addr     = new_vecs
        self._ids_block_addr = new_ids
        self._capacity       = new_cap

    # Store slot index in rev (not raw address) for simplicity
    def _vec_addr(self, slot: int) -> int:
        return self._block_addr + slot * self._vecs_ds.record_size

    def _id_addr(self, slot: int) -> int:
        return self._ids_block_addr + slot * self._ids_ds.record_size

    # ── Public API ───────────────────────────────────────────────────────

    def add(self, vector_id: str, vector: np.ndarray) -> None:
        self._ensure_rev()
        existing = self._rev.get(vector_id)
        if existing is not None:
            # Update in place
            slot = int(existing["addr"])
            vec  = self._prep_vec(np.asarray(vector, dtype=np.float32))
            self._vecs_ds.write(self._vec_addr(slot), vec=vec)
            return

        # Grow if needed
        if self._n_vecs >= self._capacity:
            new_cap  = self._capacity * 2
            new_vecs = int(self._vecs_ds.allocate_block(new_cap))
            new_ids  = int(self._ids_ds.allocate_block(new_cap))
            rs_v = self._vecs_ds.record_size
            rs_i = self._ids_ds.record_size
            old_v = self._vecs_ds.db.read(self._block_addr,      self._n_vecs * rs_v)
            old_i = self._ids_ds.db.read(self._ids_block_addr,   self._n_vecs * rs_i)
            self._vecs_ds.db.write(new_vecs, old_v)
            self._ids_ds.db.write(new_ids,   old_i)
            self._block_addr     = new_vecs
            self._ids_block_addr = new_ids
            self._capacity       = new_cap

        slot = self._n_vecs
        vec  = self._prep_vec(np.asarray(vector, dtype=np.float32))
        self._vecs_ds.write(self._vec_addr(slot), vec=vec)
        self._ids_ds.write(self._id_addr(slot), ext_id=vector_id)
        self._rev[vector_id] = {"addr": slot}
        self._n_vecs += 1
        self._auto_save_check()

    def add_batch(self, items, show_progress: bool = False):
        items = list(items)
        for vid, vec in items:
            self.add(vid, vec)

    def remove(self, vector_id: str) -> None:
        self._ensure_rev()
        ptr = self._rev.get(vector_id)
        if ptr is None:
            raise KeyError(vector_id)
        slot = int(ptr["addr"])
        self._vecs_ds.delete(self._vec_addr(slot))
        self._ids_ds.delete(self._id_addr(slot))
        del self._rev[vector_id]
        self._n_deleted += 1
        self._auto_save_check()

    def search(
        self, query: np.ndarray, k: int = 10, **kwargs
    ) -> list[tuple[str, float]]:
        if self._n_vecs == 0:
            return []
        q = self._prep_query(np.asarray(query, dtype=np.float32))

        # Bulk read all vector records
        rs_v = self._vecs_ds.record_size
        rs_i = self._ids_ds.record_size
        raw_v = self._vecs_ds.db.read(self._block_addr, self._n_vecs * rs_v)
        raw_i = self._ids_ds.db.read(self._ids_block_addr, self._n_vecs * rs_i)

        arr_v = np.frombuffer(raw_v, dtype=self._vecs_ds.schema)
        arr_i = np.frombuffer(raw_i, dtype=self._ids_ds.schema)

        # Valid mask (non-deleted)
        ident   = self._vecs_ds.identifier
        valid   = arr_v["_prefix"] == ident
        vecs    = np.array(arr_v["vec"][valid], dtype=np.float32)   # (m, dim)
        ids_raw = arr_i["ext_id"][valid]

        if len(vecs) == 0:
            return []

        higher = self.metric in ("cosine", "dot")
        scores = self._scores(q, vecs)
        top_idx, top_scores = _topk(scores, k, higher_is_better=higher)

        results = []
        for i, s in zip(top_idx, top_scores):
            ext_id = str(ids_raw[i]).rstrip("\x00")
            # text field stored via BlobStore — read properly
            id_addr = self._ids_block_addr + int(np.where(valid)[0][i]) * rs_i
            try:
                rec = self._ids_ds.read(id_addr)
                ext_id = rec["ext_id"]
            except Exception:
                pass
            results.append((ext_id, float(s)))
        return results

    def __len__(self) -> int:
        return self._n_vecs - self._n_deleted

    def __repr__(self) -> str:
        return (f"FlatIndex('{self.name}', dim={self.dim}, "
                f"n={len(self)}, metric='{self.metric}')")

    # ── Registry ────────────────────────────────────────────────────────

    def _get_registry_params(self):
        return {"dim": self.dim, "metric": self.metric, "kind": "FlatIndex"}

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(name, db, dim=params.get("dim", 0),
                   metric=params.get("metric", "cosine"))


# ── IVFIndex ──────────────────────────────────────────────────────────────


class IVFIndex(VectorIndex):
    """Approximate nearest-neighbour using an Inverted File Index.

    Storage layout (uses loom Dict[List]):
      cell_vecs: Dict[cluster_id → List[{vec: float32[dim]}]]
        Per-cluster sequential blocks → single mmap read per probed cell.
      cell_ids:  Dict[cluster_id → List[{ext_id: U{max_id_len}}]]
        IDs resolved *after* distance computation (late resolution for top-k).

    Search:
      1. Find nprobe nearest centroids.
      2. For each probed cell: slice_array() → numpy matrix → vectorised scores.
      3. Merge top-k across cells.
      4. Resolve IDs only for the top-k winners.

    pq=True — Product Quantization:
      cell_vecs stores uint8[n_sub] codes instead of float32 vectors.
      Codebook training runs alongside k-means on residuals.
      ~32–384× storage compression depending on n_sub and dim.
    """

    def __init__(
        self,
        name: str,
        db,
        dim: int = 0,
        metric: str = "cosine",
        n_clusters: int = 256,
        pq: bool = False,
        n_sub: int = 16,
        n_bits: int = 8,
        max_id_len: int = 50,
        auto_save_interval=None,
        _parent=None,
    ):
        self.n_clusters = n_clusters
        self.pq         = pq
        self.n_sub      = n_sub
        self.n_bits     = n_bits
        self.max_id_len = max_id_len
        self._trained   = False
        self._centroids = None
        self._codebooks = None

        super().__init__(name, db, dim=dim, metric=metric,
                         auto_save_interval=auto_save_interval, _parent=_parent)

    # ── Internal structures ───────────────────────────────────────────────

    def _make_cell_dicts(self):
        """Create the two Dict[List] structures for per-cell storage."""
        from loom.datastructures.dict import Dict
        from loom.datastructures.list import List

        # vec schema: float32[dim] or uint8[n_sub] for PQ
        vec_field = f"pq_codes" if self.pq else "vec"
        vec_dtype = (f"uint8[{self.n_sub}]" if self.pq
                     else f"float32[{self.dim}]")

        vec_ds = self._db.create_dataset(
            f"_vidx_{self.name}_vec_ds", **{vec_field: vec_dtype}
        )
        id_ds  = self._db.create_dataset(
            f"_vidx_{self.name}_id_ds",
            ext_id=f"U{self.max_id_len}",    # fixed-length → inline, fast bulk read
        )
        VecList = List.template(vec_ds)
        IdList  = List.template(id_ds)

        self._cell_vecs = Dict(
            f"_vidx_{self.name}_cell_vecs", self._db, VecList,
            cache_size=self.n_clusters + 8, use_bloom=False, store_key=True,
        )
        self._cell_ids = Dict(
            f"_vidx_{self.name}_cell_ids", self._db, IdList,
            cache_size=self.n_clusters + 8, use_bloom=False, store_key=True,
        )
        self._vec_field = vec_field

    def _initialize(self):
        # Centroids and codebooks datasets
        self._centroid_ds = self._db.create_dataset(
            f"_vidx_{self.name}_centroids", vec=f"float32[{self.dim}]"
        )
        sub_dim = max(1, self.dim // self.n_sub)
        self._codebook_ds = self._db.create_dataset(
            f"_vidx_{self.name}_codebooks", vec=f"float32[{sub_dim}]"
        )
        # Reverse: ext_id → (cell_id, list_idx)
        rev_ds = self._db.create_dataset(
            f"_vidx_{self.name}_rev_ds", cell_id="uint32", list_idx="uint64"
        )
        from loom.datastructures.dict import Dict
        self._rev = Dict(
            f"_vidx_{self.name}_rev", self._db, rev_ds,
            cache_size=4000, use_bloom=False, store_key=False,
        )
        self._make_cell_dicts()

        self._n_vecs         = 0
        self._cent_block_addr = 0
        self._cb_block_addr   = 0
        self._k_actual        = 0
        self._k_sub_actual    = 0
        self._save_state()

    def _load(self):
        meta = self._load_metadata()
        self.dim          = meta["dim"]
        self.metric       = meta["metric"]
        self.n_clusters   = meta["n_clusters"]
        self.pq           = meta["pq"]
        self.n_sub        = meta["n_sub"]
        self.n_bits       = meta["n_bits"]
        self.max_id_len   = meta.get("max_id_len", 50)
        self._trained     = meta["trained"]
        self._n_vecs      = meta["n_vecs"]
        self._cent_block_addr = meta.get("cent_block_addr", 0)
        self._cb_block_addr   = meta.get("cb_block_addr", 0)
        self._k_actual        = meta.get("k_actual", 0)
        self._k_sub_actual    = meta.get("k_sub_actual", 0)
        self._centroid_ds = self._db.get_dataset(meta["centroid_ds"])
        self._codebook_ds = self._db.get_dataset(meta["codebook_ds"])
        # Lazy-load cell Dicts and rev
        self._cell_vecs   = None
        self._cell_ids    = None
        self._rev         = None
        self._centroids   = None
        self._codebooks   = None
        self._vec_field   = "pq_codes" if self.pq else "vec"
        if self._trained:
            self._load_centroids()

    def _ensure_cells(self):
        """Lazily resolve cell Dicts and rev after DB reload."""
        if self._cell_vecs is None:
            self._cell_vecs = self._db._datastructures.get(
                f"_vidx_{self.name}_cell_vecs"
            )
            self._cell_ids = self._db._datastructures.get(
                f"_vidx_{self.name}_cell_ids"
            )
            self._rev = self._db._datastructures.get(
                f"_vidx_{self.name}_rev"
            )

    def _save_state(self):
        self._save_metadata({
            "dim":             self.dim,
            "metric":          self.metric,
            "n_clusters":      self.n_clusters,
            "pq":              self.pq,
            "n_sub":           self.n_sub,
            "n_bits":          self.n_bits,
            "max_id_len":      self.max_id_len,
            "trained":         self._trained,
            "n_vecs":          self._n_vecs,
            "centroid_ds":     self._centroid_ds.name,
            "codebook_ds":     self._codebook_ds.name,
            "cent_block_addr": self._cent_block_addr,
            "cb_block_addr":   self._cb_block_addr,
            "k_actual":        self._k_actual,
            "k_sub_actual":    self._k_sub_actual,
        })

    def save(self, force=False):
        self._save_state()

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, vectors: np.ndarray) -> None:
        """Train IVF centroids (and PQ codebooks if pq=True).

        Args:
            vectors: representative sample, shape (n_sample, dim)
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            vecs = _normalize(vecs)

        # IVF centroids
        self._centroids, _ = _kmeans(vecs, self.n_clusters)

        # PQ codebooks trained on residuals (query - nearest_centroid)
        if self.pq:
            assigns = self._assign_batch(vecs)
            residuals = vecs - self._centroids[assigns]
            self._codebooks = _train_pq(residuals, self.n_sub, self.n_bits)
            self._save_centroids()
        else:
            self._save_centroids()

        self._trained = True
        self._save_state()

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _assign_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Return centroid index for each vector in vecs."""
        diff  = vecs[:, None, :] - self._centroids[None, :, :]  # (n, K, d)
        dists = np.sum(diff * diff, axis=2)                       # (n, K)
        return np.argmin(dists, axis=1)                           # (n,)

    def _assign_one(self, vec: np.ndarray) -> int:
        diff  = vec[None, :] - self._centroids                    # (K, d)
        dists = np.sum(diff * diff, axis=1)                       # (K,)
        return int(np.argmin(dists))

    def _save_centroids(self):
        """Write centroids and codebooks to their dedicated Datasets."""
        K    = len(self._centroids)
        rs_c = self._centroid_ds.record_size
        block = int(self._centroid_ds.allocate_block(K))
        for i, c in enumerate(self._centroids):
            self._centroid_ds.write(block + i * rs_c, vec=c)
        self._cent_block_addr = block
        self._k_actual        = K

        if self._codebooks is not None:
            # codebooks shape: (n_sub, k_sub, sub_dim) — store as (n_sub * k_sub) records
            n_sub, k_sub, sub_dim = self._codebooks.shape
            n_cb = n_sub * k_sub
            rs_b = self._codebook_ds.record_size
            cb_block = int(self._codebook_ds.allocate_block(n_cb))
            flat = self._codebooks.reshape(n_cb, sub_dim)
            for i, v in enumerate(flat):
                self._codebook_ds.write(cb_block + i * rs_b, vec=v)
            self._cb_block_addr = cb_block
            self._k_sub_actual  = k_sub

    def _load_centroids(self):
        """Read centroids and codebooks from their Datasets back to numpy."""
        K    = self._k_actual
        rs_c = self._centroid_ds.record_size
        raw  = self._centroid_ds.db.read(self._cent_block_addr, K * rs_c)
        arr  = np.frombuffer(raw, dtype=self._centroid_ds.schema)
        self._centroids = np.array(arr["vec"], dtype=np.float32)

        if self.pq and self._k_sub_actual > 0:
            k_sub   = self._k_sub_actual
            sub_dim = self.dim // self.n_sub
            n_cb    = self.n_sub * k_sub
            rs_b    = self._codebook_ds.record_size
            raw_b   = self._codebook_ds.db.read(self._cb_block_addr, n_cb * rs_b)
            arr_b   = np.frombuffer(raw_b, dtype=self._codebook_ds.schema)
            self._codebooks = (
                np.array(arr_b["vec"], dtype=np.float32)
                  .reshape(self.n_sub, k_sub, sub_dim)
            )

    # ── Insert ────────────────────────────────────────────────────────────

    def add(self, vector_id: str, vector: np.ndarray) -> None:
        if not self._trained:
            raise RuntimeError("Call train() before adding vectors.")
        self._ensure_cells()
        vec     = self._prep_vec(np.asarray(vector, dtype=np.float32))
        cell_id = self._assign_one(vec)
        cid_str = str(cell_id)

        # Get current length of this cell's list (= insertion index)
        cell_vec_list = self._cell_vecs[cid_str]
        list_idx      = len(cell_vec_list)

        if self.pq:
            residual = vec - self._centroids[cell_id]
            codes    = _encode_pq(residual[None], self._codebooks)[0]
            cell_vec_list.append({"pq_codes": codes})
        else:
            cell_vec_list.append({"vec": vec})

        self._cell_ids[cid_str].append({"ext_id": vector_id})
        self._rev[vector_id] = {"cell_id": cell_id, "list_idx": list_idx}
        self._n_vecs += 1
        self._auto_save_check()

    def add_batch(self, items, show_progress: bool = False):
        for vid, vec in items:
            self.add(vid, vec)

    def remove(self, vector_id: str) -> None:
        """Soft-delete a vector (marks its slot invalid in both cell lists)."""
        self._ensure_cells()
        ptr = self._rev.get(vector_id)
        if ptr is None:
            raise KeyError(vector_id)
        cid_str  = str(int(ptr["cell_id"]))
        list_idx = int(ptr["list_idx"])
        del self._cell_vecs[cid_str][list_idx]
        del self._cell_ids[cid_str][list_idx]
        del self._rev[vector_id]
        self._n_vecs -= 1
        self._auto_save_check()

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self, query: np.ndarray, k: int = 10, nprobe: int | None = None
    ) -> list[tuple[str, float]]:
        if not self._trained:
            raise RuntimeError("Call train() before searching.")
        if self._n_vecs == 0:
            return []

        self._ensure_cells()
        K      = len(self._centroids)
        nprobe = min(nprobe or max(1, int(K ** 0.5)), K)
        q      = self._prep_query(np.asarray(query, dtype=np.float32))

        # Step 1: find nprobe nearest centroids
        diff        = q[None, :] - self._centroids      # (K, d)
        cent_dists  = np.sum(diff * diff, axis=1)        # (K,)
        if nprobe >= K:
            probe_cells = np.arange(K)
        else:
            probe_cells = np.argpartition(cent_dists, nprobe)[:nprobe]

        # Step 2: load only the probed cells — one mmap read per cell
        # Collect (local_score, global_cid, local_idx) tuples
        higher      = self.metric in ("cosine", "dot")
        heap_scores : list[float] = []
        heap_meta   : list[tuple[int,int]] = []  # (cell_id, local_idx)

        for cell_id in probe_cells:
            cid_str   = str(int(cell_id))
            vec_list  = self._cell_vecs[cid_str]
            n_cell    = len(vec_list)
            if n_cell == 0:
                continue

            # Single sequential mmap read for the whole cell
            cell_arr = vec_list.slice_array(0, n_cell)
            if cell_arr is None or len(cell_arr) == 0:
                continue

            if self.pq:
                codes = np.array(cell_arr["pq_codes"], dtype=np.uint8)
                # Per-cell residual for correct ADC
                residual_q = q - self._centroids[cell_id]
                table  = _adc_table(residual_q, self._codebooks)
                scores = -_adc_distances(codes, table)    # higher=better
            else:
                vecs   = np.array(cell_arr["vec"], dtype=np.float32)
                scores = self._scores(q, vecs)

            top_idx, top_sc = _topk(scores, min(k, len(scores)),
                                     higher_is_better=higher)
            for li, sc in zip(top_idx, top_sc):
                heap_scores.append(float(sc))
                heap_meta.append((int(cell_id), int(li)))

        if not heap_scores:
            return []

        # Step 3: global top-k across all probed cells
        arr_sc    = np.array(heap_scores)
        global_ki, _ = _topk(arr_sc, min(k, len(arr_sc)),
                               higher_is_better=higher)

        # Step 4: late ID resolution — only for winners
        results = []
        for gi in global_ki:
            cell_id, local_idx = heap_meta[gi]
            sc     = heap_scores[gi]
            cid_str = str(cell_id)
            try:
                ext_id = str(self._cell_ids[cid_str][local_idx]["ext_id"])
            except Exception:
                ext_id = f"cell{cell_id}_idx{local_idx}"
            results.append((ext_id, float(sc)))
        return results

    def __len__(self) -> int:
        return self._n_vecs

    def __repr__(self) -> str:
        trained = "trained" if self._trained else "untrained"
        pq_info = f", pq(M={self.n_sub})" if self.pq else ""
        return (f"IVFIndex('{self.name}', dim={self.dim}, "
                f"K={self.n_clusters}{pq_info}, n={len(self)}, {trained})")

    # ── Registry ────────────────────────────────────────────────────────

    def _get_registry_params(self):
        return {
            "dim": self.dim, "metric": self.metric, "kind": "IVFIndex",
            "n_clusters": self.n_clusters, "pq": self.pq,
            "n_sub": self.n_sub, "n_bits": self.n_bits,
        }

    @classmethod
    def _from_registry_params(cls, name, db, params):
        return cls(
            name, db,
            dim=params.get("dim", 0),
            metric=params.get("metric", "cosine"),
            n_clusters=params.get("n_clusters", 256),
            pq=params.get("pq", False),
            n_sub=params.get("n_sub", 16),
            n_bits=params.get("n_bits", 8),
        )
