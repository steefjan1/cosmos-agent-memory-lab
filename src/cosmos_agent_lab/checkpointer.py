"""A Cosmos DB-backed LangGraph checkpointer (post 4).

Implements `BaseCheckpointSaver` directly against the Cosmos DB Python SDK,
rather than depending on a third-party PyPI package -- so the whole
persistence path is visible in one file and testable with a plain mock, no
live Cosmos DB or emulator required.

Storage shape, deliberately simple (one full checkpoint per item, no
blob/version splitting):

  checkpoints container   -- one item per (tenantId, threadId, checkpointNs,
                              checkpointId), holding the serialized
                              Checkpoint + CheckpointMetadata + parent id.
  checkpoint_writes container -- one item per (tenantId, threadId,
                              checkpointNs, checkpointId), holding an array
                              of pending writes for that checkpoint.

Both containers share the [/tenantId, /threadId] hierarchical partition key
from post 2's schema, via `thread_id` encoded as "tenantId:threadId" -- the
same convention used in the LangGraph example in post 4. A bare thread_id
with no ":" falls back to a "default" tenant so the saver still works
outside a multitenant setup.
"""

from __future__ import annotations

import asyncio
import base64
import functools
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

try:  # pragma: no cover - only used for type checking / editor support
    from langchain_core.runnables import RunnableConfig
except ImportError:  # pragma: no cover
    RunnableConfig = dict  # type: ignore[assignment, misc]

DEFAULT_TENANT = "default"


def split_thread_id(thread_id: str) -> tuple[str, str]:
    """Split a "tenantId:threadId" string; falls back to DEFAULT_TENANT."""
    if ":" in thread_id:
        tenant_id, _, rest = thread_id.partition(":")
        return tenant_id, rest
    return DEFAULT_TENANT, thread_id


def _encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _item_id(checkpoint_ns: str, checkpoint_id: str) -> str:
    return f"{checkpoint_ns or '_'}::{checkpoint_id}"


class CosmosDBSaver(BaseCheckpointSaver[str]):
    """LangGraph checkpoint saver backed by two Cosmos DB containers.

    Args:
        checkpoints_container: a `ContainerProxy` (or compatible fake/mock)
            for the checkpoints container, as provisioned by
            `schema.ensure_checkpoint_containers`.
        writes_container: the matching container for pending writes.
        serde: optional custom serializer; defaults to LangGraph's
            JsonPlusSerializer via the base class.
    """

    def __init__(
        self,
        checkpoints_container: Any,
        writes_container: Any,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._checkpoints = checkpoints_container
        self._writes = writes_container

    # -- sync API -----------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        tenant_id, bare_thread_id = split_thread_id(thread_id)
        partition_key = [tenant_id, bare_thread_id]

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            try:
                item = self._checkpoints.read_item(
                    item=_item_id(checkpoint_ns, checkpoint_id),
                    partition_key=partition_key,
                )
            except Exception:
                return None
        else:
            candidates = list(
                self._checkpoints.query_items(
                    query=(
                        "SELECT * FROM c WHERE c.tenantId = @tenantId "
                        "AND c.threadId = @threadId AND c.checkpointNs = @ns "
                        "ORDER BY c.checkpointId DESC OFFSET 0 LIMIT 1"
                    ),
                    parameters=[
                        {"name": "@tenantId", "value": tenant_id},
                        {"name": "@threadId", "value": bare_thread_id},
                        {"name": "@ns", "value": checkpoint_ns},
                    ],
                    partition_key=partition_key,
                )
            )
            if not candidates:
                return None
            item = candidates[0]
            checkpoint_id = item["checkpointId"]

        checkpoint: Checkpoint = self.serde.loads_typed(
            (item["checkpointType"], _decode(item["checkpointData"]))
        )
        metadata: CheckpointMetadata = self.serde.loads_typed(
            (item["metadataType"], _decode(item["metadataData"]))
        )
        pending_writes = self._load_writes(
            tenant_id, bare_thread_id, checkpoint_ns, checkpoint_id, partition_key
        )
        parent_checkpoint_id = item.get("parentCheckpointId")

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            raise ValueError(
                "CosmosDBSaver.list() requires a config with a thread_id -- "
                "listing across every thread would mean a cross-partition "
                "fan-out query, which post 3's pitfalls section flags as an "
                "avoidable RU and isolation cost."
            )
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns")
        tenant_id, bare_thread_id = split_thread_id(thread_id)
        partition_key = [tenant_id, bare_thread_id]

        query = "SELECT * FROM c WHERE c.tenantId = @tenantId AND c.threadId = @threadId"
        params = [
            {"name": "@tenantId", "value": tenant_id},
            {"name": "@threadId", "value": bare_thread_id},
        ]
        if checkpoint_ns is not None:
            query += " AND c.checkpointNs = @ns"
            params.append({"name": "@ns", "value": checkpoint_ns})
        query += " ORDER BY c.checkpointId DESC"

        before_id = get_checkpoint_id(before) if before else None
        yielded = 0
        for item in self._checkpoints.query_items(
            query=query, parameters=params, partition_key=partition_key
        ):
            if before_id and item["checkpointId"] >= before_id:
                continue
            metadata: CheckpointMetadata = self.serde.loads_typed(
                (item["metadataType"], _decode(item["metadataData"]))
            )
            if filter and not all(
                metadata.get(k) == v for k, v in filter.items()
            ):
                continue
            if limit is not None and yielded >= limit:
                break
            yielded += 1

            checkpoint: Checkpoint = self.serde.loads_typed(
                (item["checkpointType"], _decode(item["checkpointData"]))
            )
            parent_checkpoint_id = item.get("parentCheckpointId")
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": item["checkpointNs"],
                        "checkpoint_id": item["checkpointId"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                pending_writes=self._load_writes(
                    tenant_id,
                    bare_thread_id,
                    item["checkpointNs"],
                    item["checkpointId"],
                    partition_key,
                ),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": item["checkpointNs"],
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        tenant_id, bare_thread_id = split_thread_id(thread_id)

        checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
        full_metadata = get_checkpoint_metadata(config, metadata)
        metadata_type, metadata_bytes = self.serde.dumps_typed(full_metadata)

        item = {
            "id": _item_id(checkpoint_ns, checkpoint["id"]),
            "tenantId": tenant_id,
            "threadId": bare_thread_id,
            "checkpointNs": checkpoint_ns,
            "checkpointId": checkpoint["id"],
            "parentCheckpointId": config["configurable"].get("checkpoint_id"),
            "checkpointType": checkpoint_type,
            "checkpointData": _encode(checkpoint_bytes),
            "metadataType": metadata_type,
            "metadataData": _encode(metadata_bytes),
        }
        self._checkpoints.upsert_item(item)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]
        tenant_id, bare_thread_id = split_thread_id(thread_id)
        partition_key = [tenant_id, bare_thread_id]
        item_id = _item_id(checkpoint_ns, checkpoint_id)

        try:
            existing = self._writes.read_item(item=item_id, partition_key=partition_key)
            entries: list[dict[str, Any]] = existing.get("writes", [])
        except Exception:
            existing = None
            entries = []

        existing_keys = {(e["taskId"], e["idx"]) for e in entries}
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            key = (task_id, write_idx)
            if write_idx >= 0 and key in existing_keys:
                continue  # regular writes are write-once per (task, idx)
            value_type, value_bytes = self.serde.dumps_typed(value)
            entries.append(
                {
                    "taskId": task_id,
                    "idx": write_idx,
                    "channel": channel,
                    "valueType": value_type,
                    "valueData": _encode(value_bytes),
                    "taskPath": task_path,
                }
            )
            existing_keys.add(key)

        item = {
            "id": item_id,
            "tenantId": tenant_id,
            "threadId": bare_thread_id,
            "checkpointNs": checkpoint_ns,
            "checkpointId": checkpoint_id,
            "writes": entries,
        }
        self._writes.upsert_item(item)

    def delete_thread(self, thread_id: str) -> None:
        tenant_id, bare_thread_id = split_thread_id(thread_id)
        partition_key = [tenant_id, bare_thread_id]
        for container in (self._checkpoints, self._writes):
            items = list(
                container.query_items(
                    query="SELECT c.id FROM c WHERE c.tenantId = @tenantId AND c.threadId = @threadId",
                    parameters=[
                        {"name": "@tenantId", "value": tenant_id},
                        {"name": "@threadId", "value": bare_thread_id},
                    ],
                    partition_key=partition_key,
                )
            )
            for item in items:
                container.delete_item(item=item["id"], partition_key=partition_key)

    def _load_writes(
        self,
        tenant_id: str,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        partition_key: list[str],
    ) -> list[tuple[str, str, Any]]:
        try:
            item = self._writes.read_item(
                item=_item_id(checkpoint_ns, checkpoint_id), partition_key=partition_key
            )
        except Exception:
            return []
        return [
            (
                e["taskId"],
                e["channel"],
                self.serde.loads_typed((e["valueType"], _decode(e["valueData"]))),
            )
            for e in item.get("writes", [])
        ]

    # -- async API: thin executor wrappers around the sync methods ----
    # The Cosmos SDK calls above are blocking network I/O; offloading to a
    # thread keeps an async graph run (`.ainvoke()`) from stalling the
    # event loop, unlike InMemorySaver's in-process shortcuts.

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(self.get_tuple, config))

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        loop = asyncio.get_running_loop()
        items = await loop.run_in_executor(
            None,
            functools.partial(
                lambda: list(self.list(config, filter=filter, before=before, limit=limit))
            ),
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.put, config, checkpoint, metadata, new_versions)
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, functools.partial(self.put_writes, config, writes, task_id, task_path)
        )

    async def adelete_thread(self, thread_id: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, functools.partial(self.delete_thread, thread_id))
