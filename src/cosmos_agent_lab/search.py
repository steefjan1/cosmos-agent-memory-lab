"""Four ways to ask the same question (post 3).

Mirrors the SQL patterns from "Finding the Right Memory: Vector, Full-Text,
and Hybrid Search in Cosmos DB" exactly, adapted to the turn-based schema
from post 2 (tenantId, threadId, turnIndex, messages, embedding, content).

Each function returns (query_text, parameters) so callers can either run it
directly against a live container.query_items(...) call, or -- as the test
suite does -- assert on the generated SQL/parameters without a live Cosmos
DB connection. Vector and hybrid queries require a real account; see
config.CosmosSettings.supports_vector_search.
"""

from __future__ import annotations

from typing import Any


def most_recent(tenant_id: str, thread_id: str, top: int = 5) -> tuple[str, list[dict[str, Any]]]:
    """Recency query: 'what happened most recently in this thread'."""
    query = (
        "SELECT TOP @top c.messages, c.turnIndex "
        "FROM c "
        "WHERE c.tenantId = @tenantId AND c.threadId = @threadId "
        "ORDER BY c.turnIndex DESC"
    )
    params = [
        {"name": "@top", "value": top},
        {"name": "@tenantId", "value": tenant_id},
        {"name": "@threadId", "value": thread_id},
    ]
    return query, params


def semantic(
    tenant_id: str, thread_id: str, query_vector: list[float], top: int = 5
) -> tuple[str, list[dict[str, Any]]]:
    """Semantic query: nearest neighbors by embedding distance."""
    query = (
        "SELECT TOP @top c.messages, VectorDistance(c.embedding, @queryVector) AS score "
        "FROM c "
        "WHERE c.tenantId = @tenantId AND c.threadId = @threadId "
        "ORDER BY VectorDistance(c.embedding, @queryVector)"
    )
    params = [
        {"name": "@top", "value": top},
        {"name": "@tenantId", "value": tenant_id},
        {"name": "@threadId", "value": thread_id},
        {"name": "@queryVector", "value": query_vector},
    ]
    return query, params


def hybrid(
    tenant_id: str,
    thread_id: str,
    query_vector: list[float],
    phrase: str,
    top: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """Hybrid query: RRF blend of vector similarity and BM25 keyword score."""
    query = (
        "SELECT TOP @top c.messages, c.turnIndex "
        "FROM c "
        "WHERE c.tenantId = @tenantId AND c.threadId = @threadId "
        "ORDER BY RANK RRF(VectorDistance(c.embedding, @queryVector), "
        "FullTextScore(c.content, @phrase))"
    )
    params = [
        {"name": "@top", "value": top},
        {"name": "@tenantId", "value": tenant_id},
        {"name": "@threadId", "value": thread_id},
        {"name": "@queryVector", "value": query_vector},
        {"name": "@phrase", "value": phrase},
    ]
    return query, params


def keyword(
    tenant_id: str, thread_id: str, phrase: str, top: int = 5
) -> tuple[str, list[dict[str, Any]]]:
    """Keyword query: exact-phrase match via FULLTEXTCONTAINS, ranked by recency."""
    query = (
        "SELECT TOP @top c.messages, c.turnIndex "
        "FROM c "
        "WHERE c.tenantId = @tenantId AND c.threadId = @threadId "
        "  AND FULLTEXTCONTAINS(c.content, @phrase) "
        "ORDER BY c.turnIndex DESC"
    )
    params = [
        {"name": "@top", "value": top},
        {"name": "@tenantId", "value": tenant_id},
        {"name": "@threadId", "value": thread_id},
        {"name": "@phrase", "value": phrase},
    ]
    return query, params


def run(container, query_and_params: tuple[str, list[dict[str, Any]]], tenant_id: str, thread_id: str):
    """Execute one of the query builders above against a live container.

    Scoped to a single logical partition via partition_key, per post 3's
    "forgetting WHERE filters still apply" pitfall -- this also keeps RU
    cost down by avoiding a cross-partition fan-out.
    """
    query, params = query_and_params
    return list(
        container.query_items(
            query=query,
            parameters=params,
            partition_key=[tenant_id, thread_id],
        )
    )
