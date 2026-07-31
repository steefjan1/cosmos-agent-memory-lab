"""Turn-based agent memory item shape.

Post 2 (https://sjwiggers.com — "Designing a Cosmos DB Agent Memory Schema")
settles on one document per turn as the recommended default: each item is a
complete exchange, tagged with tenantId/threadId/turnIndex, carrying an
embedding for vector search, and an optional ttl for lifecycle management.

This module is deliberately dependency-light (stdlib dataclasses only) so it
can be imported by schema.py, seed.py, search.py, and the tests without
pulling in the Cosmos SDK.

Full-text search (post 3) requires a flat, top-level string path: Cosmos DB
does not support wildcard array paths (e.g. "/messages/*/content") in a
full-text policy or index (confirmed against
https://learn.microsoft.com/azure/cosmos-db/gen-ai/full-text-search-faq --
"Wildcard paths (*, []) for arrays aren't supported in full text policies or
indexes"). So each item also carries a denormalized top-level `content`
field -- the concatenation of its messages' text -- purely so it has
something full-text-indexable to point the policy at. `messages` stays the
structured source of truth; `content` is derived from it in `to_item()`.
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
            # Flat, top-level, full-text-indexable -- see the module
            # docstring for why "messages" itself can't be the full-text path.
            "content": " ".join(m.content for m in self.messages),
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
