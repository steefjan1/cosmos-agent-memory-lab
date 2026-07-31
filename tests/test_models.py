from cosmos_agent_lab.models import MemoryTurn, Message


def test_round_trip_to_item_and_back():
    turn = MemoryTurn(
        tenant_id="contoso",
        thread_id="thread-1234",
        turn_index=7,
        messages=[Message(role="user", content="hi"), Message(role="agent", content="hello")],
        embedding=[0.1, 0.2, 0.3],
        ttl=3600,
    )
    item = turn.to_item()

    assert item["tenantId"] == "contoso"
    assert item["threadId"] == "thread-1234"
    assert item["turnIndex"] == 7
    assert item["embedding"] == [0.1, 0.2, 0.3]  # top-level, per post 2's pitfall
    assert item["ttl"] == 3600
    assert len(item["messages"]) == 2

    restored = MemoryTurn.from_item(item)
    assert restored.tenant_id == turn.tenant_id
    assert restored.thread_id == turn.thread_id
    assert restored.turn_index == turn.turn_index
    assert restored.embedding == turn.embedding
    assert restored.ttl == turn.ttl
    assert [m.content for m in restored.messages] == ["hi", "hello"]


def test_embedding_and_ttl_are_optional():
    turn = MemoryTurn(
        tenant_id="contoso",
        thread_id="thread-1234",
        turn_index=0,
        messages=[Message(role="user", content="hi")],
    )
    item = turn.to_item()
    assert "embedding" not in item
    assert "ttl" not in item
