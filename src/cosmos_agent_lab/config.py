"""Cosmos DB client configuration.

Supports two modes, selected by environment variables so the same code runs
against either target:

  - Cosmos DB Emulator (local, free, no Azure account) via COSMOS_ENDPOINT
    pointing at https://localhost:8081 with the emulator's well-known key.
  - A real Azure Cosmos DB account, via COSMOS_ENDPOINT + COSMOS_KEY.

Vector and hybrid search (post 3) require a real account: the Linux-based
Cosmos DB Emulator does not yet support vector indexing policies. Schema,
TTL, and checkpointing (posts 2 and 4) work against either target.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from azure.cosmos import CosmosClient

# The Cosmos DB Emulator always uses this well-known key. It is not a secret.
EMULATOR_ENDPOINT = "https://localhost:8081"
EMULATOR_KEY = (
    "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="
)

DATABASE_NAME = os.environ.get("COSMOS_DATABASE", "agentmemory")


@dataclass(frozen=True)
class CosmosSettings:
    endpoint: str
    key: str
    database_name: str = DATABASE_NAME

    @property
    def is_emulator(self) -> bool:
        return "localhost" in self.endpoint or "127.0.0.1" in self.endpoint

    @property
    def supports_vector_search(self) -> bool:
        """The emulator does not support vector indexing policies yet.

        Callers that need vector/hybrid search (post 3) should check this
        flag and either skip or point at a real account instead.
        """
        return not self.is_emulator


def load_settings() -> CosmosSettings:
    endpoint = os.environ.get("COSMOS_ENDPOINT", EMULATOR_ENDPOINT)
    key = os.environ.get("COSMOS_KEY", EMULATOR_KEY if "localhost" in endpoint else "")
    if not key:
        raise RuntimeError(
            "COSMOS_KEY is required when COSMOS_ENDPOINT points at a real "
            "Azure Cosmos DB account. Copy .env.example to .env and fill it in."
        )
    return CosmosSettings(endpoint=endpoint, key=key)


@lru_cache(maxsize=1)
def get_client() -> CosmosClient:
    settings = load_settings()
    # The emulator uses a self-signed certificate; disabling verification is
    # standard practice for local emulator use only, never for a real account.
    connection_verify = not settings.is_emulator
    return CosmosClient(
        settings.endpoint, credential=settings.key, connection_verify=connection_verify
    )
