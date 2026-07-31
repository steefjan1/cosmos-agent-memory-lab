"""A tiny in-memory fake of the Cosmos DB container API.

Implements just enough of `azure.cosmos.ContainerProxy`'s surface
(`upsert_item`, `read_item`, `query_items`, `delete_item`,
`query_items_change_feed`) to exercise checkpointer.py and
change_feed_demo.py without any network access, Docker, or a live Cosmos DB
account or emulator. It is not a SQL engine -- it recognizes the small,
fixed set of query shapes this repo actually issues.
"""

from __future__ import annotations

import pytest


class CosmosNotFound(Exception):
    pass


class FakeContainer:
    def __init__(self, partition_key_paths: list[str]):
        self._pk_paths = partition_key_paths  # e.g. ["tenantId", "threadId"]
        self._items: dict[tuple, dict] = {}

    def _key(self, item_or_pk) -> tuple:
        if isinstance(item_or_pk, list):
            return tuple(item_or_pk)
        return tuple(item_or_pk[p] for p in self._pk_paths)

    def upsert_item(self, item: dict) -> dict:
        pk = self._key(item)
        self._items[(pk, item["id"])] = dict(item)
        return item

    def read_item(self, item: str, partition_key) -> dict:
        pk = self._key(partition_key)
        key = (pk, item)
        if key not in self._items:
            raise CosmosNotFound(f"item {item!r} not found in partition {pk!r}")
        return dict(self._items[key])

    def delete_item(self, item: str, partition_key) -> None:
        pk = self._key(partition_key)
        self._items.pop((pk, item), None)

    def query_items(self, query: str, parameters: list[dict] | None = None, partition_key=None):
        params = {p["name"]: p["value"] for p in (parameters or [])}
        pk = self._key(partition_key) if partition_key is not None else None

        results = []
        for (item_pk, _item_id), item in self._items.items():
            if pk is not None and item_pk != pk:
                continue
            if "@tenantId" in params and item.get("tenantId") != params["@tenantId"]:
                continue
            if "@threadId" in params and item.get("threadId") != params["@threadId"]:
                continue
            if "@ns" in params and item.get("checkpointNs") != params["@ns"]:
                continue
            results.append(item)

        if "ORDER BY c.checkpointId DESC" in query:
            results.sort(key=lambda i: i.get("checkpointId", ""), reverse=True)

        if "OFFSET 0 LIMIT 1" in query:
            results = results[:1]

        return iter(dict(r) for r in results)

    def query_items_change_feed(self, partition_key=None, start_time="Now", **kwargs):
        pk = self._key(partition_key) if partition_key is not None else None
        for (item_pk, _item_id), item in self._items.items():
            if pk is not None and item_pk != pk:
                continue
            yield dict(item)


@pytest.fixture
def fake_turns_container() -> FakeContainer:
    return FakeContainer(partition_key_paths=["tenantId", "threadId"])


@pytest.fixture
def fake_checkpoints_container() -> FakeContainer:
    return FakeContainer(partition_key_paths=["tenantId", "threadId"])


@pytest.fixture
def fake_writes_container() -> FakeContainer:
    return FakeContainer(partition_key_paths=["tenantId", "threadId"])
