from cosmos_agent_lab import search


def test_most_recent_orders_by_turn_index_desc():
    query, params = search.most_recent("contoso", "thread-1234", top=3)
    assert "ORDER BY c.turnIndex DESC" in query
    assert "FullTextScore" not in query
    assert {"name": "@tenantId", "value": "contoso"} in params
    assert {"name": "@threadId", "value": "thread-1234"} in params
    assert {"name": "@top", "value": 3} in params


def test_semantic_orders_by_vector_distance():
    query, params = search.semantic("contoso", "thread-1234", [0.1, 0.2])
    assert "VectorDistance(c.embedding, @queryVector)" in query
    assert "ORDER BY VectorDistance" in query
    assert {"name": "@queryVector", "value": [0.1, 0.2]} in params


def test_hybrid_uses_rrf_of_vector_and_fulltext():
    query, params = search.hybrid("contoso", "thread-1234", [0.1, 0.2], "refund")
    assert "RANK RRF(" in query
    assert "VectorDistance(c.embedding, @queryVector)" in query
    assert "FullTextScore(c.messages, @phrase)" in query
    assert {"name": "@phrase", "value": "refund"} in params


def test_keyword_uses_fulltextcontains_and_recency_tiebreak():
    query, params = search.keyword("contoso", "thread-1234", "refund")
    assert "FULLTEXTCONTAINS(c.messages, @phrase)" in query
    assert "ORDER BY c.turnIndex DESC" in query


def test_every_query_is_scoped_to_tenant_and_thread():
    """Post 3's pitfall: forgetting WHERE filters still apply on vector/hybrid queries."""
    for query, _params in [
        search.most_recent("contoso", "thread-1234"),
        search.semantic("contoso", "thread-1234", [0.1]),
        search.hybrid("contoso", "thread-1234", [0.1], "refund"),
        search.keyword("contoso", "thread-1234", "refund"),
    ]:
        assert "c.tenantId = @tenantId" in query
        assert "c.threadId = @threadId" in query


def test_run_scopes_query_to_partition_key(fake_turns_container):
    fake_turns_container.upsert_item(
        {"id": "1", "tenantId": "contoso", "threadId": "thread-1234", "turnIndex": 0}
    )
    fake_turns_container.upsert_item(
        {"id": "2", "tenantId": "other-tenant", "threadId": "thread-9999", "turnIndex": 0}
    )
    results = search.run(
        fake_turns_container,
        search.most_recent("contoso", "thread-1234"),
        "contoso",
        "thread-1234",
    )
    assert len(results) == 1
    assert results[0]["tenantId"] == "contoso"
