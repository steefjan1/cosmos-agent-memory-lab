from langgraph.checkpoint.memory import InMemorySaver

from cosmos_agent_lab.checkpointer import CosmosDBSaver
from cosmos_agent_lab.change_feed_demo import process_new_turns
from cosmos_agent_lab.graph import build_graph, record_turn_for_handoff, run_turn, triage_node


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



def test_record_turn_for_handoff_marks_specialist_routed_turns(fake_turns_container):
    app = build_graph(InMemorySaver())
    result = run_turn(app, "contoso:thread-1234", "Can I get a refund on an unopened item?")

    item = record_turn_for_handoff(
        fake_turns_container, "contoso", "thread-1234", turn_index=0, result=result
    )

    assert item["targetAgent"] == "specialist"
    stored = fake_turns_container.read_item(item=item["id"], partition_key=["contoso", "thread-1234"])
    assert stored["targetAgent"] == "specialist"


def test_record_turn_for_handoff_leaves_non_specialist_turns_unmarked(fake_turns_container):
    app = build_graph(InMemorySaver())
    result = run_turn(app, "contoso:thread-1234", "What's the weather today?")

    item = record_turn_for_handoff(
        fake_turns_container, "contoso", "thread-1234", turn_index=0, result=result
    )

    assert item["targetAgent"] is None


def test_change_feed_actually_detects_the_graphs_own_handoff(fake_turns_container):
    """Closes the loop: a real graph run -> record_turn_for_handoff -> change
    feed all the way through, the same path scripts/run_demo.py exercises
    against a real Cosmos DB account. This is the regression test for the
    bug where the demo's change feed step always reported 0 handoffs
    because nothing had ever written a targetAgent-tagged item into
    `turns` in the first place.
    """
    app = build_graph(InMemorySaver())
    result = run_turn(app, "contoso:thread-1234", "Can I get a refund on an unopened item?")
    record_turn_for_handoff(fake_turns_container, "contoso", "thread-1234", turn_index=0, result=result)

    notified = []
    handoffs = process_new_turns(
        fake_turns_container,
        "contoso",
        "thread-1234",
        notify=lambda thread_id, turn_index: notified.append((thread_id, turn_index)),
    )

    assert handoffs == 1
    assert notified == [("thread-1234", 0)]
