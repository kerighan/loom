API reference
=============

Database
--------

.. autoclass:: loom.DB
   :members: create_dataset, get_dataset, create_dict, create_list, create_set,
             create_btree, create_queue, create_priority_queue, create_graph,
             create_search_index, create_flat_index, create_ivf_index,
             create_bloomfilter, create_counting_bloomfilter, create_lru_dict,
             collection, migrate_collection, drop_collection, vacuum, stats,
             verify, write_lock, batch, flush, close, fastapi_app, serve

Collection
----------

.. autoclass:: loom.Collection
   :members: insert, insert_many, update, delete, increment, get, find, range,
             search, sample, get_primary, reindex, keys, values, items,
             index_names

Index specs
~~~~~~~~~~~

.. autoclass:: loom.Primary
.. autoclass:: loom.Unique
.. autoclass:: loom.Range
.. autoclass:: loom.Many
.. autoclass:: loom.Search

Data structures
---------------

.. autoclass:: loom.datastructures.Dict
   :members: get, get_many, keys, values, items, set_batch, get_ref

.. autoclass:: loom.datastructures.BTree
   :members: range, prefix, keys, items, min, max, bulk_load, get

.. autoclass:: loom.datastructures.List
   :members: append, append_many, compact, get_ref

.. autoclass:: loom.datastructures.Set
   :members: add, discard, remove

.. autoclass:: loom.datastructures.Queue
   :members: push, push_many, pop, peek

.. autoclass:: loom.PriorityQueue
   :members: push, push_many, pop, peek

.. autoclass:: loom.SearchIndex
   :members: add, add_many, search, get_document, delete

Graph
~~~~~

.. autoclass:: loom.datastructures.Graph
   :members: add_node, add_nodes, add_edge, add_edges, has_edge, get_edge,
             neighbors, degree, out_degree, in_degree, query, query_iter

Vector search
~~~~~~~~~~~~~

.. autoclass:: loom.datastructures.vector_index.FlatIndex
   :members: add, search

.. autoclass:: loom.datastructures.vector_index.IVFIndex
   :members: train, add, search

Schema helpers
--------------

.. autofunction:: loom.Utf8
.. autofunction:: loom.Datetime
.. autofunction:: loom.Json
.. autofunction:: loom.Vec
.. autofunction:: loom.schema_from_model
.. autofunction:: loom.dt_key
.. autofunction:: loom.key_dt
