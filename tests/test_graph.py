from langgraph.checkpoint.memory import InMemorySaver

from cosmos_agent_lab.checkpointer import CosmosDBSaver
from cosmos_agent_lab.graph import build_graph, run_turn, triage_node


def test_triage_routes_refund_questions_to_specialist():
    state = {
        "messages": [{"role": "user", "content": "Can I get a refund?"}],
        "target_agent": None,
    }
    result = triage_node(state)
    assert result["target_agent"] == "specialist"


def test_triage_handles_non_refund_questions_directly():
    state = {
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "target_agent": None,
    }
    result = triage_node(state)
    assert result["target_agent"] is None


def test_end_to_end_graph_routes_and_answers_with_in_memory_saver():
    app = build_graph(InMemorySaver())
    result = run_turn(app, "contoso:thread-1234", "Can I return an unopened item for a refund?")
    assert result["messages"][-1]["entityId"] == "specialist-agent"
    assert "30 days" in result["messages"][-1]["content"]


def test_non_refund_question_never_reaches_specialist():
    app = build_graph(InMemorySaver())
    result = run_turn(app, "contoso:thread-1234", "What's the weather today?")
    assert result["messages"][-1]["entityId"] == "triage-agent"


def test_graph_persists_across_turns_via_cosmos_checkpointer(
    fake_checkpoints_container, fake_writes_container
):
    checkpointer = CosmosDBSaver(fake_checkpoints_container, fake_writes_container)
    app = build_graph(checkpointer)
    thread_id = "contoso:thread-1234"

    run_turn(app, thread_id, "What's the weather today?")
    result = run_turn(app, thread_id, "Actually, can I get a refund?")

    user_messages = [m["content"] for m in result["messages"] if m["role"] == "user"]
    assert len(user_messages) == 2, "second turn should see the first turn's history via Cosmos"
    assert result["messages"][-1]["entityId"] == "specialist-agent"
