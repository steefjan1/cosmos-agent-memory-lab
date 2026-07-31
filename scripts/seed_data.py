#!/usr/bin/env python3
"""CLI: seed sample turn-based memory items. Usage: python scripts/seed_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cosmos_agent_lab.config import get_client, load_settings
from cosmos_agent_lab.schema import TURNS_CONTAINER
from cosmos_agent_lab.seed import seed

if __name__ == "__main__":
    settings = load_settings()
    client = get_client()
    container = client.get_database_client(settings.database_name).get_container_client(
        TURNS_CONTAINER
    )
    count = seed(container)
    print(f"Seeded {count} turns into '{TURNS_CONTAINER}'")
