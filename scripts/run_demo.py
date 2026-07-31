#!/usr/bin/env python3
"""CLI: end-to-end demo tying posts 2-4 together.

1. Provisions the database and containers (post 2).
2. Seeds sample turns (post 2).
3. Runs all four search patterns against the seeded thread (post 3) --
   skips vector/hybrid automatically against the Emulator.
4. Runs the triage -> specialist graph through the Cosmos-backed
   checkpointer, then resumes it from a fresh process to prove state
   persisted (post 4).
5. Replays the change feed to demonstrate the handoff trigger (post 4).

Usage: python scripts/run_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cosmos_agent_lab import search
from cosmos_agent_lab.change_feed_demo import process_new_turns
from cosmos_agent_lab.checkpointer import CosmosDBSaver
from cosmos_agent_lab.config import get_client, load_settings
from cosmos_agent_lab.graph import build_graph, run_turn
from cosmos_agent_lab.schema import (
    CHECKPOINT_WRITES_CONTAINER,
    CHECKPOINTS_CONTAINER,
    TURNS_CONTAINER,
    provision_all,
)
from cosmos_agent_lab.seed import SAMPLE_TENANT, SAMPLE_THREAD, fake_embedding, seed


def main() -> None:
    settings = load_settings()
    client = get_client()
    db = client.get_database_client(settings.database_name)

    print(f"--- post 2: provisioning against {settings.endpoint} ---")
    provision_all()
    turns = db.get_container_client(TURNS_CONTAINER)
    print(f"--- post 2: seeding sample turns ---")
    seed(turns)

    if settings.supports_vector_search:
        print("--- post 3: four ways to ask the same question ---")
        vector = fake_embedding("refund policy")
        for label, query in {
            "recency": search.most_recent(SAMPLE_TENANT, SAMPLE_THREAD),
            "semantic": search.semantic(SAMPLE_TENANT, SAMPLE_THREAD, vector),
            "hybrid": search.hybrid(SAMPLE_TENANT, SAMPLE_THREAD, vector, "refund"),
            "keyword": search.keyword(SAMPLE_TENANT, SAMPLE_THREAD, "refund"),
        }.items():
            results = search.run(turns, query, SAMPLE_TENANT, SAMPLE_THREAD)
            print(f"  {label}: {len(results)} result(s)")
    else:
        print("--- post 3: skipped (Emulator does not support vector search) ---")

    print("--- post 4: triage -> specialist graph with Cosmos checkpointing ---")
    checkpoints = db.get_container_client(CHECKPOINTS_CONTAINER)
    writes = db.get_container_client(CHECKPOINT_WRITES_CONTAINER)
    checkpointer = CosmosDBSaver(checkpoints, writes)
    app = build_graph(checkpointer)
    thread_id = f"{SAMPLE_TENANT}:{SAMPLE_THREAD}"
    result = run_turn(app, thread_id, "Can I get a refund on an unopened item?")
    print(f"  final reply: {result['messages'][-1]['content']}")

    print("--- post 4: change feed handoff replay ---")
    handoffs = process_new_turns(turns, SAMPLE_TENANT, SAMPLE_THREAD)
    print(f"  {handoffs} handoff(s) detected via change feed")


if __name__ == "__main__":
    main()
