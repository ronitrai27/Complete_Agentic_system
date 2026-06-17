"""
test_mcp_gmail.py — Arcade Gmail OAuth and Tool Execution Test
Run: poetry run python scripts/test_mcp_gmail.py
"""
import os
import json
from dotenv import load_dotenv
from arcadepy import Arcade

# Load environment variables from .env
load_dotenv()

user_id = "raironit127@gmail.com"
client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))

print(f"Checking authorization status for Gmail tool (Gmail.ListThreads) for user: {user_id}...")

# Request authorization for Gmail tool
auth_response = client.tools.authorize(
    tool_name="Gmail.ListThreads",
    user_id=user_id,
)

if auth_response.status != "completed":
    print("\n" + "=" * 60)
    print("Authorization is REQUIRED to access Gmail!")
    print(f"Please visit this URL to authorize:\n\n  {auth_response.url}\n")
    print("Waiting for authorization to complete...")
    print("=" * 60 + "\n")
    
    # Wait for completion using the auth response ID
    status_response = client.auth.wait_for_completion(auth_response.id)
    if status_response.status == "completed":
        print("SUCCESS: Authorization completed successfully!")
    else:
        print(f"FAILED: Authorization failed with status: {status_response.status}")
        exit(1)
else:
    print("SUCCESS: User is already authorized!")

print("\nExecuting tool: Gmail.ListThreads...")
response = client.tools.execute(
    tool_name="Gmail.ListThreads",
    input={"max_results": 5},
    user_id=user_id,
)

print("\nResponse:")
if response.output and response.output.value:
    print(json.dumps(response.output.value, indent=2))
else:
    if response.output and response.output.error:
        print(f"Error executing tool: {response.output.error.message}")
    else:
        print("No output received or empty response.")

print("\n" + "=" * 60)
print("Gmail test complete!")
print("=" * 60)
