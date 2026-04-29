"""
loom showcase — realistic use cases for every major structure and nesting.

Each section is self-contained and runnable independently.
Run the full file: python examples/showcase.py
"""

import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

import loom
from loom.database import DB
from loom.datastructures import BTree, Dict, List, Queue, Set

TMPDIR = tempfile.mkdtemp()


def path(name):
    return os.path.join(TMPDIR, f"{name}.db")


def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════
# 1.  Dataset + Dict  —  user directory
# ══════════════════════════════════════════════════════════════

section("1. Dataset + Dict — user directory")


class User(BaseModel):
    id: int
    username: str = Field(max_length=50)
    email: str
    age: int
    premium: bool


with DB(path("users")) as db:
    users_ds = db.create_dataset("users", User)
    users = db.create_dict("by_username", users_ds)

    # bulk insert
    with db.batch():
        for i, (uname, email, age, premium) in enumerate(
            [
                ("alice", "alice@example.com", 30, True),
                ("bob", "bob@example.com", 25, False),
                ("carol", "carol@example.com", 35, True),
                ("dave", "dave@example.com", 28, False),
                ("eve", "eve@example.com", 22, True),
            ]
        ):
            users[uname] = {
                "id": i + 1,
                "username": uname,
                "email": email,
                "age": age,
                "premium": premium,
            }

    print(f"users['alice']: {users['alice']}")
    print(f"'bob' in users: {'bob' in users}")
    print(f"premium users:  {[k for k, v in users.items() if v['premium']]}")

    # Ref-based update
    ref = users_ds.insert(
        {
            "id": 6,
            "username": "frank",
            "email": "frank@example.com",
            "age": 40,
            "premium": False,
        }
    )
    ref.update(premium=True, email="frank.new@example.com")
    print(f"frank after update: {users_ds.read(ref.addr)}")

with DB(path("users")) as db:
    users = db["by_username"]
    print(f"Round-trip: alice.email = {users['alice']['email']}")


# ══════════════════════════════════════════════════════════════
# 2.  List  —  activity feed
# ══════════════════════════════════════════════════════════════

section("2. List — activity feed")


class Event(BaseModel):
    ts: int  # unix timestamp
    kind: str = Field(max_length=30)
    payload: str  # variable-length JSON / text


with DB(path("feed")) as db:
    event_ds = db.create_dataset("events", Event)
    feed = db.create_list("feed", event_ds)

    now = int(time.time())
    actions = [
        ("login", "ip=1.2.3.4"),
        ("purchase", '{"item":"loom-pro","price":49.0}'),
        ("view", "page=/dashboard"),
        ("logout", ""),
        ("login", "ip=5.6.7.8"),
    ]
    for i, (kind, payload) in enumerate(actions):
        feed.append({"ts": now + i, "kind": kind, "payload": payload})

    print(f"feed length: {len(feed)}")
    print(f"feed[0]:  {feed[0]}")
    print(f"feed[-1]: {feed[-1]}")
    print(f"feed[1:3]: {feed[1:3]}")

    del feed[2]  # soft-delete the 'view' event
    print(f"after delete: length={len(feed)}, [2].kind={feed[2]['kind']}")

with DB(path("feed")) as db:
    feed = db["feed"]
    print(f"Round-trip: length={len(feed)}")


# ══════════════════════════════════════════════════════════════
# 3.  Queue  —  job processing pipeline
# ══════════════════════════════════════════════════════════════

section("3. Queue — job processing pipeline")


class Job(BaseModel):
    id: int
    task: str = Field(max_length=80)
    priority: float
    retries: int


with DB(path("jobs")) as db:
    q = db.create_queue("pending", Job, block_size=32)

    # Producer: enqueue jobs
    q.push_many(
        [
            {
                "id": i,
                "task": f"process_batch_{i}",
                "priority": random.random(),
                "retries": 0,
            }
            for i in range(20)
        ]
    )
    print(f"queued: {len(q)} jobs")
    print(f"peek:   {q.peek()}")

    # Consumer: process 5 jobs
    done = []
    for _ in range(5):
        job = q.pop()
        done.append(job["id"])
    print(f"processed ids: {done}")
    print(f"remaining:     {len(q)}")

with DB(path("jobs")) as db:
    q = db["pending"]
    print(f"Round-trip: {len(q)} jobs still pending, next={q.peek()['id']}")


# ══════════════════════════════════════════════════════════════
# 4.  BTree  —  product catalog with range queries
# ══════════════════════════════════════════════════════════════

section("4. BTree — product catalog (sorted by name)")


class Product(BaseModel):
    sku: str = Field(max_length=20)
    name: str
    price: float
    stock: int
    category: str = Field(max_length=30)


with DB(path("catalog")) as db:
    prod_ds = db.create_dataset("products", Product)
    catalog = db.create_btree("by_name", prod_ds, key_size=80)

    products = [
        ("P001", "Apple MacBook Pro", 2499.0, 15, "laptops"),
        ("P002", "Apple iPhone 15", 999.0, 50, "phones"),
        ("P003", "Dell XPS 15", 1799.0, 8, "laptops"),
        ("P004", "Google Pixel 8", 699.0, 30, "phones"),
        ("P005", "Samsung Galaxy S24", 799.0, 25, "phones"),
        ("P006", "Sony WH-1000XM5", 349.0, 40, "audio"),
        ("P007", "iPad Pro 12.9", 1099.0, 20, "tablets"),
    ]
    for sku, name, price, stock, cat in products:
        catalog[name] = {
            "sku": sku,
            "name": name,
            "price": price,
            "stock": stock,
            "category": cat,
        }

    # All products in alphabetical order
    print("All products (alpha):")
    for k in catalog.keys():
        print(f"  {k}")

    # Range query: names between 'D' and 'T'
    print("\nNames [D → S]:")
    for k, v in catalog.range("D", "S"):
        print(f"  {k}: ${v['price']:.0f}")

    # Prefix: all 'Apple' products
    print("\nApple prefix:")
    for k, v in catalog.prefix("Apple"):
        print(f"  {k}: ${v['price']:.0f}  stock={v['stock']}")

with DB(path("catalog")) as db:
    catalog = db["by_name"]
    print(f"Round-trip: {catalog['Sony WH-1000XM5']['price']}")


# ══════════════════════════════════════════════════════════════
# 5.  BTree + datetime  —  ticker timeseries
# ══════════════════════════════════════════════════════════════

section("5. Dict[BTree] — ticker timeseries (OHLCV)")


class OHLCV(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int


with DB(path("market")) as db:
    ohlcv_ds = db.create_dataset("ohlcv", OHLCV)
    OHLCVTree = BTree.template(ohlcv_ds, key_size=loom.dt_key_size("second"))
    tickers = db.create_dict("tickers", OHLCVTree)

    # Simulate 30 daily bars for AAPL and TSLA
    random.seed(1)
    market_open = datetime(2024, 1, 2, 9, 30)

    for ticker, seed_price in [("AAPL", 185.0), ("TSLA", 250.0)]:
        price = seed_price
        for day in range(30):
            dt = market_open + timedelta(days=day)
            price *= 1 + random.gauss(0, 0.015)
            tickers[ticker].set_dt(
                dt,
                {
                    "open": round(price, 2),
                    "high": round(price * 1.008, 2),
                    "low": round(price * 0.992, 2),
                    "close": round(price * 1.003, 2),
                    "volume": random.randint(15_000_000, 60_000_000),
                },
            )

    # Range: AAPL first week
    week_end = market_open + timedelta(days=4)
    print("AAPL first week:")
    for k, row in tickers["AAPL"].range_dt(market_open, week_end):
        print(
            f"  {loom.key_dt(k).date()}  "
            f"C={row['close']:>8.2f}  V={int(row['volume']):>12,}"
        )

    # Point lookup
    bar = tickers["TSLA"].get_dt(market_open + timedelta(days=10))
    print(f"TSLA day 11 close: {bar['close']}")

    # All-time high (iterate keys in order, keep max)
    max_close = max(v["close"] for _, v in tickers["AAPL"].items())
    print(f"AAPL all-time high close over period: {max_close:.2f}")

with DB(path("market")) as db:
    tickers = db["tickers"]
    print(
        f"Round-trip: AAPL bars={len(tickers['AAPL'])}, TSLA bars={len(tickers['TSLA'])}"
    )


# ══════════════════════════════════════════════════════════════
# 6.  Dict[List]  —  user post feed (social platform)
# ══════════════════════════════════════════════════════════════

section("6. Dict[List] — per-user post feed")


class Post(BaseModel):
    id: int
    title: str = Field(max_length=100)
    body: str
    likes: int


with DB(path("posts")) as db:
    post_ds = db.create_dataset("posts", Post)
    PostFeed = List.template(post_ds)
    user_feed = db.create_dict("feed", PostFeed)

    # Each user has their own list of posts
    posts_data = {
        "alice": [
            ("Intro to loom", "loom is a persistent Python DB...", 42),
            ("BTree range queries", "Fast ordered lookups with range()...", 18),
        ],
        "bob": [
            ("My TSLA strategy", "Buy when RSI < 30...", 7),
        ],
        "carol": [
            ("Python tips", "Use type hints everywhere...", 93),
            ("Async patterns", "asyncio.gather is your friend...", 55),
            ("loom timeseries", "Dict[BTree] for OHLCV data...", 31),
        ],
    }
    post_id = 1
    for username, posts in posts_data.items():
        for title, body, likes in posts:
            user_feed[username].append(
                {"id": post_id, "title": title, "body": body, "likes": likes}
            )
            post_id += 1

    # Access
    print("Alice's posts:")
    for p in user_feed["alice"]:
        print(f"  [{p['likes']:>3} likes] {p['title']}")

    print("Carol's most liked post:")
    best = max(user_feed["carol"], key=lambda p: p["likes"])
    print(f"  {best['title']} ({best['likes']} likes)")

with DB(path("posts")) as db:
    user_feed = db["feed"]
    print(f"Round-trip: carol posts={len(user_feed['carol'])}")


# ══════════════════════════════════════════════════════════════
# 7.  Dict[Set]  —  user tag system
# ══════════════════════════════════════════════════════════════

section("7. Dict[Set] — user interests / tag system")

with DB(path("tags")) as db:
    TagSet = Set.template(key_size=50)
    user_tags = db.create_dict("tags", TagSet)

    interests = {
        "alice": ["python", "machine-learning", "databases", "finance"],
        "bob": ["trading", "finance", "rust"],
        "carol": ["python", "databases", "open-source", "devops"],
    }
    for user, tags in interests.items():
        for tag in tags:
            user_tags[user].add(tag)

    # Membership test
    print(f"alice has 'finance': {'finance' in user_tags['alice']}")
    print(f"bob has 'python':    {'python'  in user_tags['bob']}")

    # Shared interests (alice ∩ carol)
    alice_tags = set(user_tags["alice"])
    carol_tags = set(user_tags["carol"])
    print(f"alice ∩ carol: {alice_tags & carol_tags}")

    # All users interested in 'finance'
    finance_users = [u for u in ["alice", "bob", "carol"] if "finance" in user_tags[u]]
    print(f"users into finance: {finance_users}")

with DB(path("tags")) as db:
    user_tags = db["tags"]
    print(f"Round-trip: alice tags={sorted(user_tags['alice'])}")


# ══════════════════════════════════════════════════════════════
# 8.  Dict[Queue]  —  per-user notification inbox
# ══════════════════════════════════════════════════════════════

section("8. Dict[Queue] — per-user notification inbox")


class Notification(BaseModel):
    id: int
    kind: str = Field(max_length=30)
    message: str


with DB(path("inbox")) as db:
    notif_ds = db.create_dataset("notifs", Notification)
    NotifQ = Queue.template(notif_ds, block_size=16)
    inboxes = db.create_dict("inboxes", NotifQ)

    # Push notifications to various users
    nid = 1
    events = [
        ("alice", "like", "Bob liked your post"),
        ("alice", "follow", "Carol started following you"),
        ("bob", "mention", "Alice mentioned you in a post"),
        ("alice", "comment", "Dave commented: 'Great article!'"),
        ("carol", "like", "Eve liked your post"),
    ]
    for user, kind, msg in events:
        inboxes[user].push({"id": nid, "kind": kind, "message": msg})
        nid += 1

    print("Inbox sizes:", {u: len(inboxes[u]) for u in ["alice", "bob", "carol"]})

    # Alice reads her notifications one by one
    print("Alice reading inbox:")
    while inboxes["alice"]:
        notif = inboxes["alice"].pop()
        print(f"  [{notif['kind']}] {notif['message']}")
    print(f"  inbox empty: {len(inboxes['alice']) == 0}")

with DB(path("inbox")) as db:
    inboxes = db["inboxes"]
    print(f"Round-trip: bob has {len(inboxes['bob'])} unread")


# ══════════════════════════════════════════════════════════════
# 9.  Dict[Dict]  —  multi-env config store
# ══════════════════════════════════════════════════════════════

section("9. Dict[Dict] — application config (app → env → values)")


class ConfigEntry(BaseModel):
    value: str
    secret: bool
    updated: int  # unix timestamp


with DB(path("config")) as db:
    cfg_ds = db.create_dataset("cfg", ConfigEntry)
    CfgDict = Dict.template(cfg_ds)
    app_cfg = db.create_dict("config", CfgDict)

    now = int(time.time())
    configs = {
        "api-service": {
            "production": {
                "DATABASE_URL": ("postgres://prod-host/db", True, now),
                "LOG_LEVEL": ("WARNING", False, now),
                "WORKERS": ("8", False, now),
            },
            "staging": {
                "DATABASE_URL": ("postgres://staging-host/db", True, now),
                "LOG_LEVEL": ("DEBUG", False, now),
                "WORKERS": ("2", False, now),
            },
        },
        "worker": {
            "production": {
                "QUEUE_URL": ("redis://prod-redis:6379", True, now),
                "CONCURRENCY": ("16", False, now),
            },
        },
    }
    for app, envs in configs.items():
        for env, keys in envs.items():
            ns_key = f"{app}/{env}"
            for k, (val, secret, ts) in keys.items():
                app_cfg[ns_key][k] = {"value": val, "secret": secret, "updated": ts}

    # Read config for api-service/production
    prod = app_cfg["api-service/production"]
    print("api-service/production:")
    for k in prod.keys():
        entry = prod[k]
        display = "***" if entry["secret"] else entry["value"]
        print(f"  {k} = {display}")

    # Update a value
    app_cfg["api-service/staging"]["WORKERS"] = {
        "value": "4",
        "secret": False,
        "updated": now + 1,
    }
    print(
        f"staging workers updated: {app_cfg['api-service/staging']['WORKERS']['value']}"
    )

with DB(path("config")) as db:
    app_cfg = db["config"]
    log = app_cfg["api-service/production"]["LOG_LEVEL"]
    print(f"Round-trip: LOG_LEVEL = {log['value']}")


# ══════════════════════════════════════════════════════════════
# 10.  Graph + Cypher  —  social follow network
# ══════════════════════════════════════════════════════════════

section("10. Graph + Cypher — social follow network")


class Person(BaseModel):
    name: str
    age: int
    location: str = Field(max_length=50)


class Follows(BaseModel):
    since: int  # unix timestamp
    weight: float  # engagement score 0–1


with DB(path("social")) as db:
    g = db.create_graph("social", Person, Follows, directed=True, node_id_max_len=20)

    people = [
        ("alice", "Alice", 30, "Paris"),
        ("bob", "Bob", 25, "London"),
        ("carol", "Carol", 35, "Paris"),
        ("dave", "Dave", 28, "NYC"),
        ("eve", "Eve", 22, "London"),
        ("frank", "Frank", 40, "Berlin"),
    ]
    g.add_nodes(
        [(pid, {"name": n, "age": a, "location": loc}) for pid, n, a, loc in people]
    )

    follows = [
        ("alice", "bob", 2021, 0.9),
        ("alice", "carol", 2020, 0.7),
        ("alice", "dave", 2022, 0.5),
        ("bob", "alice", 2021, 0.8),
        ("bob", "eve", 2023, 0.6),
        ("carol", "alice", 2020, 0.9),
        ("carol", "frank", 2019, 0.4),
        ("dave", "alice", 2022, 0.7),
        ("eve", "bob", 2023, 0.5),
        ("frank", "carol", 2021, 0.3),
    ]
    g.add_edges([(s, d, {"since": ts, "weight": w}) for s, d, ts, w in follows])

    # Followers / following
    print(f"Alice follows:   {list(g.neighbors('alice'))}")
    print(f"Alice followers: {list(g.predecessors('alice'))}")
    print(f"Degrees: in={g.in_degree('alice')}, out={g.out_degree('alice')}")

    # Cypher: mutual follows (a ↔ b)
    print("\nMutual follows:")
    seen = set()
    for r in g.query("MATCH (a)->(b) RETURN id(a), id(b)"):
        a, b = r["id(a)"], r["id(b)"]
        if g.has_edge(b, a) and (b, a) not in seen:
            print(f"  {a} ↔ {b}")
            seen.add((a, b))

    # Cypher: strong connections (weight > 0.7)
    print("\nStrong connections (weight > 0.7):")
    for r in g.query(
        "MATCH (a)-[e]->(b) WHERE e.weight > 0.7 RETURN a.name, b.name, e.weight"
    ):
        print(f"  {r['a.name']} → {r['b.name']}  (w={r['e.weight']})")

    # Cypher: Parisians who follow Londoners
    print("\nParis → London:")
    for r in g.query(
        "MATCH (a)->(b) "
        "WHERE a.location == 'Paris' AND b.location == 'London' "
        "RETURN a.name, b.name"
    ):
        print(f"  {r['a.name']} → {r['b.name']}")

    # Cypher: 2-hop reach from alice
    print("\nAlice can reach in ≤ 2 hops:")
    for r in g.query("MATCH (a)-[*1..2]->(b) WHERE id(a) == 'alice' RETURN id(b)"):
        print(f"  {r['id(b)']}")

    # Inline props
    for r in g.query("MATCH (a {location:'London'})->(b) RETURN a.name, b.name"):
        print(f"  London user {r['a.name']} follows {r['b.name']}")

with DB(path("social")) as db:
    g = db["social"]
    print(f"Round-trip: {g}")
    print(f"  alice still follows bob: {g.has_edge('alice','bob')}")


# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("  All examples completed successfully.")
print(f"{'═'*60}\n")

import shutil

shutil.rmtree(TMPDIR)
