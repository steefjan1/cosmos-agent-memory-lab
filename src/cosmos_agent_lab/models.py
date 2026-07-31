"""Turn-based agent memory item shape.

Post 2 (https://sjwiggers.com — "Designing a Cosmos DB Agent Memory Schema")
settles on one document per turn as the recommended default: each item is a
complete exchange, tagged with tenantId/threadId/turnIndex, carrying an
embedding for vector search, and an optional ttl for lifecycle management.

This module is deliberately dependency-light (stdlib dataclasses only) so it
can be imported by schema.py, seed.py, search.py, and the tests without
pulling in the Cosmos SDK.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "agent" | "tool"
    content: str
    entity_id: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.entity_id:
            d["entityId"] = self.entity_id
        d["timestamp"] = self.timestamp or _now_iso()
        return d


@dataclass
class MemoryTurn:
    """One document per turn, per post 2.

    `tenant_id` and `thread_id` together form the hierarchical partition key
    `[/tenantId, /threadId]` used by schema.py, mirroring the pattern from
    "Azure Cosmos DB's Latest Performance Features" (2023) and reused for
    the checkpointer's thread_id encoding in post 4 (`tenantId:threadId`).
    """

    tenant_id: str
    thread_id: str
    turn_index: int
    messages: list[Message]
    embedding: list[float] | None = None
    ttl: int | None = None  # seconds; -1 = never expire; None = container default
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": self.id,
            "tenantId": self.tenant_id,
            "threadId": self.thread_id,
            "turnIndex": self.turn_index,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.embedding is not None:
            # Top-level field, per post 2's "embeddings must be top-level to
            # be indexable" pitfall.
            item["embedding"] = self.embedding
        if self.ttl is not None:
            item["ttl"] = self.ttl
        return item

    @staticmethod
    def from_item(item: dict[str, Any]) -> "MemoryTurn":
        return MemoryTurn(
            tenant_id=item["tenantId"],
            thread_id=item["threadId"],
            turn_index=item["turnIndex"],
            messages=[
                Message(
                    role=m["role"],
                    content=m["content"],
                    entity_id=m.get("entityId"),
                    timestamp=m.get("timestamp"),
                )
                for m in item.get("messages", [])
            ],
            embedding=item.get("embedding"),
            ttl=item.get("ttl"),
            id=item["id"],
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
