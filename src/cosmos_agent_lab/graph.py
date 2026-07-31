"""A minimal two-agent LangGraph graph: triage -> specialist (post 4).

Mirrors the shared-but-separable pattern from "Multi-Agent State and
Checkpointing with Cosmos DB": a triage node decides whether a specialist
needs to get involved, and `checkpointer.CosmosDBSaver` persists state after
every step so the conversation can resume across process restarts.

This intentionally stays close to Microsoft's own multi-agent-langgraph
sample's shape (triage agent routes, specialist/product agent answers) but
trimmed down to the minimum needed to demonstrate checkpointing -- see
https://github.com/AzureCosmosDB/multi-agent-langgraph for the full
production-shaped version.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    target_agent: str | None


def triage_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]["content"].lower()
    target = "specialist" if "refund" in last_message else None
    reply = {
        "role": "agent",
        "entityId": "triage-agent",
        "content": (
            f"Routing to the specialist agent." if target else "I can help with that directly."
        ),
    }
    return {"messages": [*state["messages"], reply], "target_agent": target}


def specialist_node(state: AgentState) -> AgentState:
    reply = {
        "role": "agent",
        "entityId": "specialist-agent",
        "content": "Refund policy is 30 days for unopened items.",
    }
    return {"messages": [*state["messages"], reply], "target_agent": None}


def _route_after_triage(state: AgentState) -> Literal["specialist", "__end__"]:
    return "specialist" if state.get("target_agent") == "specialist" else END


def build_graph(checkpointer: Any):
    """Compile the triage -> specialist graph with the given checkpointer.

    `checkpointer` is typically a `checkpointer.CosmosDBSaver`, but any
    object implementing LangGraph's `BaseCheckpointSaver` interface works --
    including `langgraph.checkpoint.memory.InMemorySaver`, which the test
    suite uses to check the graph's routing logic in isolation from Cosmos DB.
    """
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("specialist", specialist_node)
    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage", _route_after_triage, {"specialist": "specialist", END: END}
    )
    graph.add_edge("specialist", END)
    return graph.compile(checkpointer=checkpointer)


def run_turn(app: Any, thread_id: str, user_text: str) -> AgentState:
    """Run one user turn through the graph, resuming from any prior state."""
    config = {"configurable": {"thread_id": thread_id}}
    existing = app.get_state(config)
    prior_messages = existing.values.get("messages", []) if existing.values else []
    result = app.invoke(
        {
            "messages": [*prior_messages, {"role": "user", "content": user_text}],
            "target_agent": None,
        },
        config=config,
    )
    return result
