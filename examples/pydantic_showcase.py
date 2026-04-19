"""
loom — exemples complets avec Pydantic

Pydantic est la façon recommandée de définir des schemas dans loom :
  - int    → int64
  - float  → float64
  - bool   → bool
  - str    → text (variable-length, compressé via BlobStore)
  - Annotated[str, StringConstraints(max_length=N)] → U{N} (fixe, rapide)

Les types fixe-longueur (U{N}) sont plus rapides à lire mais gaspillent de
l'espace si les valeurs sont courtes. Utilisez 'str' (→ text) par défaut
sauf si vous avez besoin de vitesse maximale sur ce champ.
"""

import os
import tempfile
from pydantic import BaseModel, StringConstraints
from typing import Annotated

from loom.database import DB
from loom.datastructures import List, Dict

# ─────────────────────────────────────────────────────────────
# Schemas Pydantic
# ─────────────────────────────────────────────────────────────

class User(BaseModel):
    id: int
    username: Annotated[str, StringConstraints(max_length=50)]  # U50 — clé fréquente, fixe
    email: str                                                    # text — variable
    age: int
    active: bool


class Post(BaseModel):
    id: int
    title: Annotated[str, StringConstraints(max_length=100)]
    body: str          # contenu long → text
    author_id: int
    views: int


class ChatMessage(BaseModel):
    role: Annotated[str, StringConstraints(max_length=20)]  # "user" / "assistant"
    content: str                                              # texte libre → text


class Tag(BaseModel):
    name: Annotated[str, StringConstraints(max_length=50)]
    weight: float


# Pour le graph
class Person(BaseModel):
    name: str
    age: int
    city: str


class Follows(BaseModel):
    since: int
    weight: float


# ─────────────────────────────────────────────────────────────
# 1. Dataset — CRUD de base
# ─────────────────────────────────────────────────────────────

def example_dataset(path):
    print("\n── 1. Dataset (CRUD) ──────────────────────────────────")

    with DB(path) as db:
        users = db.create_dataset("users", User)

        # Insert
        alice = users.insert({"id": 1, "username": "alice", "email": "alice@example.com",
                               "age": 30, "active": True})
        bob   = users.insert({"id": 2, "username": "bob",   "email": "bob@example.com",
                               "age": 25, "active": False})

        print(f"Alice: {users.read(alice.addr)}")
        print(f"Bob:   {users.read(bob.addr)}")

        # Update via Ref
        alice.update(age=31, email="alice-new@example.com")
        print(f"Alice (updated): {users.read(alice.addr)}")

        # Soft delete
        users.delete(bob.addr)
        print(f"Bob exists: {users.exists(bob.addr)}")

    # Round-trip
    with DB(path) as db:
        users = db["users"]
        print(f"Round-trip Alice: {users.read(alice.addr)}")


# ─────────────────────────────────────────────────────────────
# 2. Dict — clé → enregistrement Pydantic
# ─────────────────────────────────────────────────────────────

def example_dict(path):
    print("\n── 2. Dict ─────────────────────────────────────────────")

    with DB(path) as db:
        user_ds = db.create_dataset("d_users", User)
        users   = db.create_dict("users_by_name", user_ds)

        with db.batch():
            for rec in [
                {"id": 1, "username": "alice", "email": "a@x.com", "age": 30, "active": True},
                {"id": 2, "username": "bob",   "email": "b@x.com", "age": 25, "active": True},
                {"id": 3, "username": "carol", "email": "c@x.com", "age": 35, "active": False},
            ]:
                users[rec["username"]] = rec

        print(f"alice: {users['alice']}")
        print(f"'bob' in users: {'bob' in users}")
        print(f"all keys: {list(users.keys())}")

        # bulk export en une lecture
        snapshot = users.to_dict()
        print(f"to_dict() → {len(snapshot)} entrées, ages: {[v['age'] for v in snapshot.values()]}")

    with DB(path) as db:
        users = db["users_by_name"]
        print(f"Round-trip bob: {users['bob']}")


# ─────────────────────────────────────────────────────────────
# 3. List — tableau persistant
# ─────────────────────────────────────────────────────────────

def example_list(path):
    print("\n── 3. List ─────────────────────────────────────────────")

    with DB(path) as db:
        post_ds = db.create_dataset("l_posts", Post)
        posts   = db.create_list("all_posts", post_ds)

        with db.batch():
            for i in range(5):
                posts.append({"id": i, "title": f"Post {i}", "body": f"Content of post {i}",
                               "author_id": i % 2, "views": i * 100})

        print(f"len: {len(posts)}")
        print(f"posts[0]: {posts[0]}")
        print(f"posts[-1]: {posts[-1]}")
        print(f"posts[1:3]: {posts[1:3]}")

        # Soft delete + auto-compact
        del posts[2]
        print(f"After del[2]: len={len(posts)}, posts[2].title={posts[2]['title']}")

    with DB(path) as db:
        posts = db["all_posts"]
        print(f"Round-trip len: {len(posts)}, posts[0]: {posts[0]}")


# ─────────────────────────────────────────────────────────────
# 4. BTree — accès ordonné + range queries
# ─────────────────────────────────────────────────────────────

def example_btree(path):
    print("\n── 4. BTree (ordered, range queries) ───────────────────")

    with DB(path) as db:
        post_ds = db.create_dataset("bt_posts", Post)
        index   = db.create_btree("posts_by_title", post_ds, key_size=100)

        with db.batch():
            for title, body, views in [
                ("Alpha post",   "...", 50),
                ("Beta article", "...", 200),
                ("Gamma note",   "...", 10),
                ("Delta review", "...", 500),
            ]:
                index[title] = {"id": 0, "title": title, "body": body,
                                 "author_id": 1, "views": views}

        print("Tous les titres (ordre alpha):")
        for k in index.keys():
            print(f"  {k}")

        print("Range ['B'..'D']:")
        for k, v in index.range("B", "D"):
            print(f"  {k} → {v['views']} views")

        print("Prefix 'G':")
        for k, v in index.prefix("G"):
            print(f"  {k}")

    with DB(path) as db:
        index = db["posts_by_title"]
        print(f"Round-trip 'Alpha post': {index['Alpha post']['views']} views")


# ─────────────────────────────────────────────────────────────
# 5. Text (variable-length) — messages chat
# ─────────────────────────────────────────────────────────────

def example_text(path):
    print("\n── 5. Text dtype (variable-length) ─────────────────────")

    with DB(path) as db:
        msg_ds = db.create_dataset("messages", ChatMessage)

        long_msg = "Voici une réponse très détaillée. " * 50

        refs = []
        with db.batch():
            for role, content in [
                ("user",      "Explique-moi les réseaux de neurones."),
                ("assistant", long_msg),
                ("user",      "Merci !"),
            ]:
                refs.append(msg_ds.insert({"role": role, "content": content}))

        for ref in refs:
            rec = msg_ds.read(ref.addr)
            print(f"[{rec['role']}] {rec['content'][:60]}{'...' if len(rec['content'])>60 else ''}")

        # Update d'un champ text (libère l'ancien blob)
        msg_ds.write_field(refs[2].addr, "content", "Super, encore merci !")
        print(f"Updated: {msg_ds.read(refs[2].addr)['content']}")

    with DB(path) as db:
        msg_ds = db["messages"]
        rec = msg_ds.read(refs[1].addr)
        print(f"Round-trip long message: {len(rec['content'])} chars")


# ─────────────────────────────────────────────────────────────
# 6. Nested — Dict[Dict] (user → posts par slug)
# ─────────────────────────────────────────────────────────────

def example_nested_dict_dict(path):
    print("\n── 6. Nested Dict[Dict] ─────────────────────────────────")

    with DB(path) as db:
        post_ds  = db.create_dataset("nd_posts", Post)
        PostDict = Dict.template(post_ds)
        user_posts = db.create_dict("user_posts", PostDict)

        # Accès auto-crée les sous-dicts
        user_posts["alice"]["intro"]  = {"id": 1, "title": "Hello world",
                                          "body": "My first post", "author_id": 1, "views": 0}
        user_posts["alice"]["python"] = {"id": 2, "title": "Python tips",
                                          "body": "Always use type hints", "author_id": 1, "views": 42}
        user_posts["bob"]["graphs"]   = {"id": 3, "title": "Graph theory",
                                          "body": "Nodes and edges", "author_id": 2, "views": 7}

        print(f"alice/intro: {user_posts['alice']['intro']['title']}")
        print(f"alice keys: {list(user_posts['alice'].keys())}")
        print(f"bob/graphs views: {user_posts['bob']['graphs']['views']}")

    with DB(path) as db:
        user_posts = db["user_posts"]
        print(f"Round-trip alice/python: {user_posts['alice']['python']['title']}")


# ─────────────────────────────────────────────────────────────
# 7. Nested — List[Dict] (timeline de posts par user)
# ─────────────────────────────────────────────────────────────

def example_nested_list_dict(path):
    print("\n── 7. Nested List[Dict] ─────────────────────────────────")

    with DB(path) as db:
        tag_ds   = db.create_dataset("ld_tags", Tag)
        TagList  = List.template(tag_ds)
        tag_lists = db.create_dict("user_tags", TagList)

        alice_tags = tag_lists["alice"]
        alice_tags.append({"name": "python", "weight": 0.9})
        alice_tags.append({"name": "ml",     "weight": 0.7})
        alice_tags.append({"name": "graphs",  "weight": 0.5})

        bob_tags = tag_lists["bob"]
        bob_tags.append({"name": "rust",  "weight": 0.8})
        bob_tags.append({"name": "infra", "weight": 0.6})

        print(f"alice tags: {[t['name'] for t in tag_lists['alice']]}")
        print(f"bob tags:   {[t['name'] for t in tag_lists['bob']]}")
        print(f"alice len:  {len(tag_lists['alice'])}")

    with DB(path) as db:
        tag_lists = db["user_tags"]
        print(f"Round-trip alice tags: {[t['name'] for t in tag_lists['alice']]}")


# ─────────────────────────────────────────────────────────────
# 8. Graph + Pydantic + Cypher
# ─────────────────────────────────────────────────────────────

def example_graph(path):
    print("\n── 8. Graph + Cypher ────────────────────────────────────")

    people = [
        ("alice",   "Alice",   28, "Paris"),
        ("bob",     "Bob",     32, "Lyon"),
        ("charlie", "Charlie", 25, "Paris"),
        ("diana",   "Diana",   35, "Bordeaux"),
        ("eve",     "Eve",     22, "Paris"),
        ("frank",   "Frank",   40, "Lyon"),
    ]
    follows = [
        ("alice",   "bob",     2020, 0.9),
        ("alice",   "charlie", 2021, 0.6),
        ("bob",     "diana",   2019, 0.8),
        ("charlie", "alice",   2022, 0.7),  # cycle !
        ("charlie", "eve",     2023, 0.5),
        ("diana",   "alice",   2021, 0.4),
        ("frank",   "bob",     2022, 0.3),
        ("frank",   "diana",   2020, 0.9),
    ]

    with DB(path) as db:
        g = db.create_graph(
            "social",
            Person, Follows,
            directed=True,
            node_id_max_len=10,
        )

        for pid, name, age, city in people:
            g.add_node(pid, name=name, age=age, city=city)

        for src, dst, since, w in follows:
            g.add_edge(src, dst, since=since, weight=w)

        # ── Queries basiques ──
        print("Qui Alice suit:")
        for r in g.query("MATCH (a)->(b) WHERE id(a) == \"alice\" RETURN b.name"):
            print(f"  {r['b.name']}")

        print("Qui suit Alice:")
        for r in g.query("MATCH (a)<-(b) WHERE id(a) == \"alice\" RETURN id(b), b.name"):
            print(f"  {r['id(b)']} ({r['b.name']})")

        # ── Filtre attribut ──
        print("Liens forts (weight > 0.7):")
        for r in g.query(
            "MATCH (a)-[r]->(b) WHERE r.weight > 0.7 RETURN a.name, b.name, r.weight"
        ):
            print(f"  {r['a.name']} → {r['b.name']} (w={r['r.weight']:.1f})")

        # ── Filtre multi-attribut ──
        print("Parisiens qui suivent des gens > 30 ans:")
        for r in g.query(
            "MATCH (a)->(b) WHERE a.city == \"Paris\" AND b.age > 30 RETURN a.name, b.name"
        ):
            print(f"  {r['a.name']} → {r['b.name']}")

        # ── Inline props ──
        print("Followers de Charlie (inline props):")
        for r in g.query("MATCH (a {name:\"Charlie\"})<-(b) RETURN b.name"):
            print(f"  {r['b.name']}")

        # ── Quantifier + (1 à ∞ hops) ──
        print("Tout ce qu'alice peut atteindre (chemin direct ou indirect):")
        for r in g.query(
            "MATCH (a)-[+]->(b) WHERE id(a) == \"alice\" RETURN id(b)"
        ):
            print(f"  {r['id(b)']}")

        # ── Quantifier *2 ──
        print("À 2 hops depuis frank:")
        for r in g.query(
            "MATCH (a)-[*2]->(b) WHERE id(a) == \"frank\" RETURN id(b), b.name"
        ):
            print(f"  {r['id(b)']} ({r['b.name']})")

        # ── id() dans RETURN ──
        print("id() dans RETURN:")
        for r in g.query(
            "MATCH (a)->(b) WHERE id(a) == \"bob\" RETURN id(a), id(b)"
        ):
            print(f"  {r['id(a)']} → {r['id(b)']}")

        # ── LIMIT ──
        print("3 premiers liens depuis 2020:")
        for r in g.query(
            "MATCH (a)-[r]->(b) WHERE r.since >= 2020 RETURN a.name, b.name LIMIT 3"
        ):
            print(f"  {r['a.name']} → {r['b.name']}")

        # ── Degrees ──
        print(f"\nDegrés:")
        for pid, name, _, _ in people:
            print(f"  {name}: in={g.in_degree(pid)}, out={g.out_degree(pid)}")

    # ── Round-trip ──
    with DB(path) as db:
        g = db["social"]
        print(f"\nRound-trip graph: {g}")
        results = g.query(
            "MATCH (a)-[r]->(b) WHERE r.weight > 0.7 RETURN a.name, b.name"
        )
        print(f"Liens forts après reload: {[(r['a.name'], r['b.name']) for r in results]}")


# ─────────────────────────────────────────────────────────────
# 9. Batch mode — inserts en masse
# ─────────────────────────────────────────────────────────────

def example_batch(path):
    print("\n── 9. Batch mode ────────────────────────────────────────")
    import time

    with DB(path) as db:
        post_ds  = db.create_dataset("batch_posts", Post)
        post_idx = db.create_dict("batch_posts_idx", post_ds)

        N = 2000

        t0 = time.perf_counter()
        with db.batch():
            for i in range(N):
                post_idx[f"post_{i:05d}"] = {
                    "id": i, "title": f"Post {i}",
                    "body": f"Content {i}", "author_id": i % 10, "views": i,
                }
        elapsed = time.perf_counter() - t0

        print(f"{N} inserts en batch: {elapsed:.3f}s ({elapsed/N*1e6:.0f} µs/op)")
        print(f"Taille dict: {len(post_idx)}")
        print(f"post_01000: {post_idx['post_01000']['title']}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        def p(name):
            return os.path.join(tmpdir, f"{name}.db")

        example_dataset(p("dataset"))
        example_dict(p("dict"))
        example_list(p("list"))
        example_btree(p("btree"))
        example_text(p("text"))
        example_nested_dict_dict(p("nested_dd"))
        example_nested_list_dict(p("nested_ld"))
        example_graph(p("graph"))
        example_batch(p("batch"))

        print("\n✓ Tous les exemples ont tourné sans erreur.")
