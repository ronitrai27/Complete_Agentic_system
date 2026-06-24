import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.composio_agent import get_composio

comp = get_composio()
user_id = "user_v55i61letn6c"

for tk in ["gmail", "slack"]:
    try:
        tools = comp.tools.get(user_id=user_id, toolkits=[tk], limit=100)
        names = [t.name for t in tools]
        print(f"Toolkit {tk} with limit=100 ({len(names)} tools):")
        print(", ".join(sorted(names)))
    except Exception as e:
        print(f"Error for {tk}: {e}")
