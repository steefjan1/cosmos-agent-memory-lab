"""Deployable version of change_feed_demo.process_new_turns (post 4).

Bound to the `turns` container's change feed via function.json. Any turn
written with `targetAgent == "specialist"` wakes the specialist agent --
this is the production-shaped counterpart to
cosmos_agent_lab.change_feed_demo, which does the same thing locally
without the Azure Functions runtime.
"""

from __future__ import annotations

import logging

import azure.functions as func


def main(documents: func.DocumentList) -> None:
    if not documents:
        return
    for doc in documents:
        if doc.get("targetAgent") == "specialist":
            logging.info(
                "change-feed handoff: waking specialist agent for thread=%s turn=%s",
                doc.get("threadId"),
                doc.get("turnIndex"),
            )
            notify_specialist_agent(doc["threadId"], doc["turnIndex"])


def notify_specialist_agent(thread_id: str, turn_index: int) -> None:
    # Replace with a real signal: a queue message, a Service Bus topic
    # publish, or a direct call into the specialist agent's entry point.
    logging.info("notify_specialist_agent(thread_id=%s, turn_index=%s)", thread_id, turn_index)
