import sys
import os
import time
import threading
import uuid
from typing import Dict, Any

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath('.'))

# Attempt to configure stdout to UTF-8 to handle emojis on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from dotenv import load_dotenv
load_dotenv()

from src.utils import event_bus
from src.agents.rag_agent import compile_agent, resume_agent
from src.agents.state import create_initial_state
from langgraph.types import Command

# Attempt to fix Windows console colors using colorama
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass

# Colors for terminal printing
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_GREY = "\033[90m"

def safe_print(text: str, end: str = "\n"):
    """Prints text safely, falling back to ascii-only if console encoding fails."""
    try:
        print(text, end=end)
        sys.stdout.flush()
    except UnicodeEncodeError:
        clean_text = text.encode('ascii', 'replace').decode('ascii')
        print(clean_text, end=end)
        sys.stdout.flush()

def safe_input(prompt: str) -> str:
    """Prompts for input safely, falling back to ascii-only if prompt fails to print."""
    try:
        safe_print(prompt, end="")
        return input()
    except (UnicodeEncodeError, UnicodeDecodeError):
        clean_prompt = prompt.encode('ascii', 'replace').decode('ascii')
        safe_print(clean_prompt, end="")
        return input()

def print_event(event: dict):
    msg = event.get("message", "")
    
    # Pick color based on emojis or content
    if "✅" in msg:
        color = COLOR_GREEN
    elif "❌" in msg:
        color = COLOR_RED
    elif "⚠️" in msg:
        color = COLOR_YELLOW
    elif "🧭" in msg or "Router" in msg:
        color = COLOR_CYAN
    elif "🤖" in msg or "LLM" in msg:
        color = COLOR_BLUE
    elif "🕸️" in msg or "Neo4j" in msg:
        color = COLOR_CYAN
    elif "🧠" in msg or "Pinecone" in msg:
        color = COLOR_BLUE
    elif "📚" in msg or "BM25" in msg:
        color = COLOR_GREEN
    elif "🛠️" in msg or "MCP" in msg:
        color = COLOR_YELLOW
    else:
        color = COLOR_GREY
        
    safe_print(f"{color}{msg}{COLOR_RESET}")

def run_with_progress(action_fn):
    stop_event = threading.Event()
    result_holder = [None]
    error_holder = [None]
    
    def target():
        try:
            result_holder[0] = action_fn()
        except Exception as e:
            error_holder[0] = e
        finally:
            stop_event.set()
            
    t = threading.Thread(target=target, daemon=True)
    t.start()
    
    # Poll events in real-time
    while not stop_event.is_set():
        events = event_bus.get_all()
        for e in events:
            print_event(e)
        time.sleep(0.1)
        
    # Flush remaining events
    events = event_bus.get_all()
    for e in events:
        print_event(e)
        
    if error_holder[0]:
        raise error_holder[0]
        
    return result_holder[0]

def check_auth_interrupt(result):
    interrupts = result.get("__interrupt__", []) if result else []
    if interrupts:
        for intr in interrupts:
            val = intr.value if hasattr(intr, "value") else intr
            if isinstance(val, dict) and val.get("type") == "authorization_required":
                auth = val.get("auth_response", {})
                return {
                    "tool_name": val.get("tool_name", "unknown tool"),
                    "auth_id":   auth.get("id"),
                    "auth_url":  auth.get("url"),
                }
    return None

def check_hitl_interrupt(result):
    interrupts = result.get("__interrupt__", []) if result else []
    if interrupts:
        for intr in interrupts:
            val = intr.value if hasattr(intr, "value") else intr
            if isinstance(val, dict) and val.get("type") == "hitl_checkpoint":
                return True
    return False

def main():
    safe_print("=" * 80)
    safe_print(f"{COLOR_BOLD}{COLOR_GREEN}🚀 Welcome to RAG Agent CLI Terminal Interface! 🚀{COLOR_RESET}")
    safe_print(f"{COLOR_GREY}Persistent memory is enabled. Ask questions across multiple turns.{COLOR_RESET}")
    safe_print("=" * 80)
    
    # Setup persistent thread/conversation ID
    conv_id = f"cli_{uuid.uuid4().hex[:6]}"
    safe_print(f"{COLOR_GREY}Session Thread ID: {conv_id}{COLOR_RESET}\n")
    
    agent = compile_agent()
    
    while True:
        try:
            query = safe_input(f"{COLOR_BOLD}👤 You: {COLOR_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            safe_print(f"\n{COLOR_GREEN}Goodbye!{COLOR_RESET}")
            break
            
        if not query:
            continue
            
        if query.lower() in ("exit", "quit"):
            safe_print(f"{COLOR_GREEN}Goodbye!{COLOR_RESET}")
            break
            
        # Clear event bus for the new turn
        event_bus.clear()
        
        safe_print(f"\n{COLOR_GREY}--- Running Agent Turn ---{COLOR_RESET}")
        start_time = time.time()
        
        # Define initial action
        initial_state = create_initial_state(
            user_query=query,
            conversation_id=conv_id,
        )
        config = {"configurable": {"thread_id": conv_id}}
        
        try:
            # Run initial turn with progress printer
            result = run_with_progress(lambda: agent.invoke(initial_state, config=config))
            
            # Re-entrant loop to handle authorization or HITL checkpoint interrupts
            while True:
                auth_info = check_auth_interrupt(result)
                if auth_info:
                    elapsed = time.time() - start_time
                    safe_print(f"\n{COLOR_YELLOW}⏱️  Paused after {elapsed:.2f} seconds.{COLOR_RESET}")
                    safe_print("\n" + "🔐 " * 20)
                    safe_print(f"{COLOR_BOLD}{COLOR_YELLOW}🔐 OAuth Authorization Required for: {auth_info['tool_name']}{COLOR_RESET}")
                    safe_print(f"Please visit this URL in your browser to grant permissions:")
                    safe_print(f"{COLOR_BLUE}{auth_info['auth_url']}{COLOR_RESET}")
                    safe_print("🔐 " * 20 + "\n")
                    
                    inp = safe_input("Have you authorized? Press Enter to continue, or type 'cancel' to deny: ").strip().lower()
                    
                    event_bus.clear()
                    start_time = time.time()  # reset timer for resume
                    if inp == "cancel":
                        result = run_with_progress(lambda: agent.invoke(Command(resume={"authorized": False}), config=config))
                    else:
                        result = run_with_progress(lambda: agent.invoke(Command(resume={"authorized": True}), config=config))
                    continue # check for next interrupts (e.g. HITL)
                
                # Check for HITL memory save checkpoint
                if check_hitl_interrupt(result):
                    elapsed = time.time() - start_time
                    safe_print(f"\n{COLOR_YELLOW}⏱️  Paused after {elapsed:.2f} seconds.{COLOR_RESET}")
                    safe_print("\n" + "💾 " * 20)
                    safe_print(f"{COLOR_BOLD}{COLOR_GREEN}💾 Long-term Memory Checkpoint{COLOR_RESET}")
                    safe_print(f"The agent generated an answer and wants to store this turn in memory.")
                    safe_print("💾 " * 20 + "\n")
                    
                    choice = safe_input("Save this turn to Neo4j graph & SQLite memory? (y/n) [default: y]: ").strip().lower()
                    approved = choice != 'n'
                    
                    event_bus.clear()
                    start_time = time.time() # reset timer for resume
                    result = run_with_progress(lambda: resume_agent(conv_id, {"approved": approved, "notes": None}))
                    continue # check if any further interrupts
                    
                # No more interrupts, print final answer and break out of re-entrant loop
                break
                
            elapsed = time.time() - start_time
            safe_print("\n" + "=" * 80)
            safe_print(f"{COLOR_BOLD}{COLOR_GREEN}🤖 Agent Response:{COLOR_RESET}")
            safe_print(result.get("final_answer", "No answer returned."))
            safe_print("=" * 80)
            safe_print(f"{COLOR_GREEN}⏱️  Turn Response Time: {elapsed:.2f} seconds{COLOR_RESET}\n")
            
        except Exception as e:
            safe_print(f"\n{COLOR_RED}❌ Error running agent: {e}{COLOR_RESET}\n")

if __name__ == "__main__":
    main()
