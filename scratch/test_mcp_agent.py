import os
import sys
import uuid
import asyncio
from dotenv import load_dotenv

# Ensure we can import from root
sys.path.insert(0, os.path.abspath("."))

load_dotenv()

from ui.app import run_agent_turn
from src.utils import event_bus
from langgraph.errors import GraphInterrupt

query = "Show my latest gmail threads"
conv_id = str(uuid.uuid4())

print("Invoking run_agent_turn for MCP query...")
event_bus.clear()

try:
    result = run_agent_turn(query, None, conv_id)
    print("\n--- Result ---")
    print("Route:", result.get("route"))
    print("Final Answer:", result.get("final_answer"))
    print("Interrupts:", result.get("__interrupt__"))
except GraphInterrupt as gi:
    print("\n--- Caught GraphInterrupt Exception ---")
    print(gi)
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n--- Event Bus Events ---")
for event in event_bus.get_all():
    print(f"[{event.get('type')}] {event.get('message')}")
