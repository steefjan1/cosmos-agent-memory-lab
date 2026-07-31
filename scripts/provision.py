#!/usr/bin/env python3
"""CLI: create the database and containers (post 2). Usage: python scripts/provision.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cosmos_agent_lab.schema import provision_all

if __name__ == "__main__":
    provision_all()
