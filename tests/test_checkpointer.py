from langgraph.graph import StateGraph

from cosmos_agent_lab.checkpointer import CosmosDBSaver, split_thread_id


def test_split_thread_id_parses_tenant_and_thread():
    assert split_thread_id("contoso:thread-1234") == ("contoso", "thread-1234")


def test_split_thread_id_falls_back_to_default_tenant():
    assert split_thread_id("thread-1234") == ("default", "thread-1234")


def _build_counter_graph(checkpointer):
    builder = StateGraph(int)
    builder.add_node("add_one", lambda x: x + 1)
    builder.set_entry_point("add_one")
    builder.set_finish_point("add_one")
    return builder.compile(checkpointer=checkpointer)


def test_checkpoint_persists_and_resumes_across_saver_instances(
    fake_checkpoints_container, fake_writes_container
):
    config = {"configurable": {"thread_id": "contoso:thread-1234"}}

    saver_1 = CosmosDBSaver(fake_checkpoints_container, fake_writes_container)
    app_1 = _build_counter_graph(saver_1)
    result_1 = app_1.invoke(1, config)
    assert result_1 == 2

    # Simulate a brand-new process: a fresh saver instance pointed at the
    # same (fake) Cosmos containers.
    saver_2 = CosmosDBSaver(fake_checkpoints_container, fake_writes_container)
    app_2 = _build_counter_graph(saver_2)
    state = app_2.get_state(config)
    assert state.values == 2

    tuple_ = saver_2.get_tuple(config)
    assert tuple_ is not None
    assert tuple_.checkpoint["id"] == state.config["configurable"]["checkpoint_id"]


def test_list_returns_checkpoints_newest_first(fake_checkpoints_container, fake_writes_container):
    config = {"configurable": {"thread_id": "contoso:thread-1234"}}
    saver = CosmosDBSaver(fake_checkpoints_container, fake_writes_container)
    app = _build_counter_graph(saver)
    app.invoke(1, config)
    app.invoke(app.get_state(config).values, config)

    checkpoints = list(saver.list(config))
    assert len(checkpoints) >= 2
    ids = [c.checkpoint["id"] for c in checkpoints]
    assert ids == sorted(ids, reverse=True)


def test_delete_thread_removes_all_checkpoints(fake_checkpoints_container, fake_writes_container):
    config = {"configurable": {"thread_id": "contoso:thread-1234"}}
    saver = CosmosDBSaver(fake_checkpoints_container, fake_writes_container)
    app = _build_counter_graph(saver)
    app.invoke(1, config)

    saver.delete_thread("contoso:thread-1234")
    assert saver.get_tuple(config) is None


def test_isolates_tenants_sharing_the_same_bare_thread_id(
    fake_checkpoints_container, fake_writes_container
):
    """Post 4's pitfall: shared containers without tenant isolation leak state."""
    saver = CosmosDBSaver(fake_checkpoints_container, fake_writes_container)
    app = _build_counter_graph(saver)

    app.invoke(1, {"configurable": {"thread_id": "tenant-a:thread-1234"}})
    app.invoke(100, {"configurable": {"thread_id": "tenant-b:thread-1234"}})

    state_a = app.get_state({"configurable": {"thread_id": "tenant-a:thread-1234"}})
    state_b = app.get_state({"configurable": {"thread_id": "tenant-b:thread-1234"}})
    assert state_a.values == 2
    assert state_b.values == 101
