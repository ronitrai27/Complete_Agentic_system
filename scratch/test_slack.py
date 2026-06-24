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

try:
    print("Testing GMAIL_SEND_EMAIL...")
    res1 = comp.tools.execute(
        slug="GMAIL_SEND_EMAIL",
        arguments={
            "recipient_email": "ronitrai27@gmail.com",
            "subject": "Test Subject from Workflow Builder",
            "body": "Test Body"
        },
        user_id=user_id,
        dangerously_skip_version_check=True
    )
    print("GMAIL_SEND_EMAIL response:", res1)
except Exception as e:
    print("GMAIL_SEND_EMAIL error:", e)

try:
    print("Testing SLACK_SEND_MESSAGE...")
    res2 = comp.tools.execute(
        slug="SLACK_SEND_MESSAGE",
        arguments={
            "channel": "general",
            "markdown_text": "Test Slack message from Workflow Builder"
        },
        user_id=user_id,
        dangerously_skip_version_check=True
    )
    print("SLACK_SEND_MESSAGE response:", res2)
except Exception as e:
    print("SLACK_SEND_MESSAGE error:", e)
