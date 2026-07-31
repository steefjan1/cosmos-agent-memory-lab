"""Change feed as the multi-agent handoff mechanism, runnable locally (post 4).

The blog post's demonstration is an Azure Function with a Cosmos DB trigger
(see functions/change_feed_handoff/), which is the right shape for a real
deployment but needs the Azure Functions Core Tools to run. This module
does the same thing with the Cosmos Python SDK's change feed API directly,
so the handoff pattern is runnable with nothing but `python`.
"""

from __future__ import annotations

from typing import Any, Callable


def default_notify(thread_id: str, turn_index: int) -> None:
    print(f"[change-feed] waking specialist agent for thread={thread_id} turn={turn_index}")


def process_new_turns(
    container: Any,
    tenant_id: str,
    thread_id: str,
    *,
    notify: Callable[[str, int], None] = default_notify,
    start_time: str = "Beginning",
) -> int:
    """Read the change feed for one thread's partition and react to handoffs.

    Mirrors the Azure Function from the blog post: any turn item written
    with `targetAgent == "specialist"` triggers a notification. Scoped to a
    single [tenantId, threadId] partition, consistent with post 4's pitfall
    about change feed only guaranteeing order within a partition -- this
    demo never assumes ordering across threads or tenants.
    """
    handoffs = 0
    for item in container.query_items_change_feed(
        partition_key=[tenant_id, thread_id], start_time=start_time
    ):
        if item.get("targetAgent") == "specialist":
            notify(item["threadId"], item["turnIndex"])
            handoffs += 1
    return handoffs
