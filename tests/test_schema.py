from cosmos_agent_lab.config import CosmosSettings
from cosmos_agent_lab.schema import _turns_indexing_policy, _turns_vector_embedding_policy


def _settings(endpoint: str) -> CosmosSettings:
    return CosmosSettings(endpoint=endpoint, key="fake-key")


def test_emulator_endpoint_disables_vector_search():
    settings = _settings("https://localhost:8081")
    assert settings.is_emulator is True
    assert settings.supports_vector_search is False


def test_real_account_endpoint_enables_vector_search():
    settings = _settings("https://contoso.documents.azure.com:443/")
    assert settings.is_emulator is False
    assert settings.supports_vector_search is True


def test_indexing_policy_includes_vector_index_only_for_real_accounts():
    emulator_policy = _turns_indexing_policy(_settings("https://localhost:8081"))
    cloud_policy = _turns_indexing_policy(_settings("https://contoso.documents.azure.com:443/"))

    assert "vectorIndexes" not in emulator_policy
    assert cloud_policy["vectorIndexes"] == [{"path": "/embedding", "type": "diskANN"}]

    # Full-text indexing on message content is expected either way -- it
    # doesn't depend on vector support.
    assert emulator_policy["fullTextIndexes"] == [{"path": "/messages/*/content"}]
    assert cloud_policy["fullTextIndexes"] == [{"path": "/messages/*/content"}]


def test_vector_embedding_policy_matches_schema_dimensions():
    from cosmos_agent_lab.schema import EMBEDDING_DIMENSIONS

    policy = _turns_vector_embedding_policy()
    assert policy["vectorEmbeddings"][0]["dimensions"] == EMBEDDING_DIMENSIONS
    assert policy["vectorEmbeddings"][0]["path"] == "/embedding"
