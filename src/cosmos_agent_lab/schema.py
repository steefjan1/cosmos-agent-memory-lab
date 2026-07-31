"""Container provisioning for Cosmos DB agent memory (post 2).

Creates the `turns` container with:
  - a hierarchical partition key [/tenantId, /threadId]
  - TTL enabled at the container level (default -1: no expiry unless an
    item sets its own positive ttl)
  - a vector indexing policy on /embedding (DiskANN), used by search.py in
    post 3 -- skipped automatically against the Emulator, which does not
    yet support vector policies (see config.supports_vector_search)
  - a full-text indexing policy on /messages/*/content, used by the
    keyword and hybrid queries in post 3

Run via `python scripts/provision.py`.
"""

from __future__ import annotations

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy

from .config import CosmosSettings, get_client, load_settings

TURNS_CONTAINER = "turns"
CHECKPOINTS_CONTAINER = "checkpoints"
CHECKPOINT_WRITES_CONTAINER = "checkpoint_writes"

EMBEDDING_DIMENSIONS = 8  # kept small on purpose -- this is a teaching sample,
# not a production embedding size. Swap in your real model's dimension count.


def ensure_database(client: CosmosClient, settings: CosmosSettings) -> DatabaseProxy:
    return client.create_database_if_not_exists(id=settings.database_name)


def _turns_vector_embedding_policy() -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": EMBEDDING_DIMENSIONS,
            }
        ]
    }


def _turns_indexing_policy(settings: CosmosSettings) -> dict:
    policy: dict = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": "/embedding/*"}, {"path": '/"_etag"/?'}],
        "fullTextIndexes": [{"path": "/messages/*/content"}],
    }
    if settings.supports_vector_search:
        policy["vectorIndexes"] = [{"path": "/embedding", "type": "diskANN"}]
    return policy


def ensure_turns_container(
    client: CosmosClient, settings: CosmosSettings
) -> ContainerProxy:
    """Post 2's worked schema: hierarchical partition key + TTL.

    Vector and full-text policies are attached when the target supports
    them (a real account); against the Emulator, the container still gets
    created with the hierarchical partition key and TTL so posts 2 and 4
    remain fully runnable locally, and only post 3's vector/hybrid queries
    require pointing COSMOS_ENDPOINT at a real account.
    """
    database = ensure_database(client, settings)
    kwargs: dict = dict(
        id=TURNS_CONTAINER,
        partition_key=PartitionKey(path=["/tenantId", "/threadId"], kind="MultiHash"),
        default_ttl=-1,
        indexing_policy=_turns_indexing_policy(settings),
    )
    if settings.supports_vector_search:
        kwargs["vector_embedding_policy"] = _turns_vector_embedding_policy()
        kwargs["full_text_policy"] = {
            "defaultLanguage": "en-US",
            "fullTextPaths": [{"path": "/messages/*/content", "language": "en-US"}],
        }
    return database.create_container_if_not_exists(**kwargs)


def ensure_checkpoint_containers(
    client: CosmosClient, settings: CosmosSettings
) -> tuple[ContainerProxy, ContainerProxy]:
    """Containers backing checkpointer.py (post 4).

    Same [/tenantId, /threadId] hierarchical partition key as `turns`, so
    a single tenant's checkpoints and memory turns colocate the same way
    post 4's pitfalls section recommends.
    """
    database = ensure_database(client, settings)
    checkpoints = database.create_container_if_not_exists(
        id=CHECKPOINTS_CONTAINER,
        partition_key=PartitionKey(path=["/tenantId", "/threadId"], kind="MultiHash"),
    )
    writes = database.create_container_if_not_exists(
        id=CHECKPOINT_WRITES_CONTAINER,
        partition_key=PartitionKey(path=["/tenantId", "/threadId"], kind="MultiHash"),
    )
    return checkpoints, writes


def provision_all() -> None:
    settings = load_settings()
    client = get_client()
    ensure_turns_container(client, settings)
    ensure_checkpoint_containers(client, settings)
    print(f"Provisioned database '{settings.database_name}' at {settings.endpoint}")
    if not settings.supports_vector_search:
        print(
            "Note: running against the Emulator -- 'turns' was created without a "
            "vector/full-text policy. Point COSMOS_ENDPOINT at a real Azure Cosmos "
            "DB account to exercise post 3's vector and hybrid queries."
        )
