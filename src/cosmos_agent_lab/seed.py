"""Seed a handful of turn-based memory items for the demo scripts and tests.

Uses tiny 4-dimensional-looking (but 8-d, matching schema.EMBEDDING_DIMENSIONS)
fake embeddings -- deterministic, not from a real model -- so the sample
runs with zero external API calls or keys beyond Cosmos DB itself.
"""

from __future__ import annotations

import hashlib

from .models import MemoryTurn, Message
from .schema import EMBEDDING_DIMENSIONS

SAMPLE_TENANT = "contoso"
SAMPLE_THREAD = "thread-1234"


def fake_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic pseudo-embedding derived from a text hash.

    Good enough to exercise VectorDistance() ordering in the demo without
    requiring an embeddings API key. Do not use this for anything real.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(b / 255.0) * 2 - 1 for b in digest[:dimensions]]


def sample_turns(tenant_id: str = SAMPLE_TENANT, thread_id: str = SAMPLE_THREAD) -> list[MemoryTurn]:
    exchanges = [
        ("What's your refund policy for accessories?", "Refund policy is 30 days for unopened items."),
        ("Can I return an opened accessory?", "Opened accessories are final sale, unfortunately."),
        ("What's the weather like today?", "I don't have weather data, but I can help with orders."),
        ("Do refunds cover shipping costs?", "Original shipping is non-refundable; return shipping is on us."),
        ("Is there a restocking fee?", "No restocking fee on unopened returns within 30 days."),
    ]
    turns = []
    for i, (user_text, agent_text) in enumerate(exchanges):
        turns.append(
            MemoryTurn(
                tenant_id=tenant_id,
                thread_id=thread_id,
                turn_index=i,
                messages=[
                    Message(role="user", content=user_text),
                    Message(role="agent", content=agent_text),
                ],
                embedding=fake_embedding(user_text),
                ttl=3600,
            )
        )
    return turns


def seed(container) -> int:
    turns = sample_turns()
    for turn in turns:
        container.upsert_item(turn.to_item())
    return len(turns)
