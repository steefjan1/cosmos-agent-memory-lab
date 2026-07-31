from cosmos_agent_lab.change_feed_demo import process_new_turns


def test_only_targetagent_specialist_turns_trigger_a_handoff(fake_turns_container):
    fake_turns_container.upsert_item(
        {
            "id": "1",
            "tenantId": "contoso",
            "threadId": "thread-1234",
            "turnIndex": 0,
            "targetAgent": "specialist",
        }
    )
    fake_turns_container.upsert_item(
        {
            "id": "2",
            "tenantId": "contoso",
            "threadId": "thread-1234",
            "turnIndex": 1,
            "targetAgent": None,
        }
    )

    notified = []
    handoffs = process_new_turns(
        fake_turns_container,
        "contoso",
        "thread-1234",
        notify=lambda thread_id, turn_index: notified.append((thread_id, turn_index)),
    )

    assert handoffs == 1
    assert notified == [("thread-1234", 0)]


def test_scoped_to_a_single_thread_partition(fake_turns_container):
    """Post 4's pitfall: change feed is partition-scoped, not globally ordered."""
    fake_turns_container.upsert_item(
        {
            "id": "1",
            "tenantId": "contoso",
            "threadId": "thread-A",
            "turnIndex": 0,
            "targetAgent": "specialist",
        }
    )
    fake_turns_container.upsert_item(
        {
            "id": "2",
            "tenantId": "contoso",
            "threadId": "thread-B",
            "turnIndex": 0,
            "targetAgent": "specialist",
        }
    )

    handoffs = process_new_turns(fake_turns_container, "contoso", "thread-A")
    assert handoffs == 1  # thread-B's handoff is invisible from thread-A's partition
