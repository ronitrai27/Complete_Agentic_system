"""
Workflow Creator Streamlit App — chat with an agent to build and run workflows.
"""
import sys

# Force stdout/stderr to use UTF-8 encoding on Windows to prevent UnicodeEncodeErrors
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uuid
import sqlite3
import traceback
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agents.composio_agent import (
    create_user_session,
    get_connect_url,
    get_toolkit_status,
)
from src.config import settings
from workflows.agent import run_workflow_agent_stream


# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic Workflow Creator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
    <style>
    /* Hide automatic multi-page navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Clean layout tweaks */
    .stApp {
        background-color: #0f1115;
        color: #e6e8ea;
    }
    
    /* Chat message container styling */
    .user-msg {
        background-color: #1f232a;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #7928ca;
    }
    .assistant-msg {
        background-color: #17191e;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #0070f3;
    }
    
    /* Visual workflow step design */
    .wf-container {
        background-color: #17191e;
        border: 1px solid #2e333d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Database and State helpers ───────────────────────────────────────────────

def get_or_create_permanent_user_id() -> str:
    db_dir = ROOT / "data"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "conversations.db"
    
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
        
        cursor.execute("SELECT value FROM user_config WHERE key = 'composio_user_id'")
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            import random
            import string
            chars = string.ascii_lowercase + string.digits
            new_id = "user_" + "".join(random.choices(chars, k=12))
            
            cursor.execute(
                "INSERT INTO user_config (key, value) VALUES ('composio_user_id', ?)",
                (new_id,)
            )
            conn.commit()
            return new_id
    finally:
        conn.close()


def save_connected_toolkits_to_sqlite(user_id: str, toolkits: list[dict]):
    db_path = ROOT / "data" / "conversations.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connected_toolkits (
                user_id   TEXT,
                toolkit   TEXT,
                connected INTEGER,
                updated_at TEXT,
                PRIMARY KEY (user_id, toolkit)
            )
        """)
        now = datetime.now(timezone.utc).isoformat()
        for tk in toolkits:
            slug = tk["slug"]
            connected = 1 if tk["connected"] else 0
            cursor.execute("""
                INSERT INTO connected_toolkits (user_id, toolkit, connected, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, toolkit) DO UPDATE SET
                    connected = excluded.connected,
                    updated_at = excluded.updated_at
            """, (user_id, slug, connected, now))
        conn.commit()
    except Exception as exc:
        print(f"[SQLite Toolkit Cache Error] {exc}", flush=True)
    finally:
        conn.close()


def init_state():
    defaults = {
        "user_id": get_or_create_permanent_user_id(),
        "conv_id": str(uuid.uuid4()),
        "messages": [],
        "agent_running": False,
        "connect_url": None,
        "connect_label": None,
        "status_refresh": 0,
        "workflow": {
            "name": "Untitled Workflow",
            "description": "Describe the workflow you want to build in the chat! (e.g. 'Reddit to Slack summaries')",
            "steps": []
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# ─── Environment Checks ───────────────────────────────────────────────────────

missing_keys = []
if not settings.composio_api_key:
    missing_keys.append("COMPOSIO_API_KEY")
if not (settings.openai_api_key or settings.openai_key):
    missing_keys.append("OPENAI_API_KEY")

if missing_keys:
    st.error(f"Missing environment keys: {', '.join(missing_keys)}. Add them to `.env` and restart streamlit.")
    st.stop()


# ─── Composio Cache Helpers ───────────────────────────────────────────────────

@st.cache_resource
def get_cached_session(user_id: str):
    return create_user_session(user_id)


def get_session():
    return get_cached_session(st.session_state.user_id)


def load_all_composio_tools(user_id: str, toolkits: list[str]):
    from src.agents.composio_agent import get_composio
    comp = get_composio()
    all_tools = []
    for tk in toolkits:
        try:
            tools = comp.tools.get(user_id=user_id, toolkits=[tk], limit=200)
            all_tools.extend(tools)
        except Exception as exc:
            print(f"[load_all_composio_tools] Error loading {tk}: {exc}", flush=True)
    return all_tools


@st.cache_resource
def get_tools_map(user_id: str):
    try:
        from src.agents.composio_agent import TOOLKITS
        tools = load_all_composio_tools(user_id, TOOLKITS)
        return {t.name: t for t in tools}
    except Exception as exc:
        st.error(f"Failed to load tools list from Composio: {exc}")
        return {}


def coerce_value_for_widget(val: Any, val_type: str) -> Any:
    val_type_lower = str(val_type).lower() if val_type else "string"
    if val_type_lower == "boolean":
        if isinstance(val, bool):
            return val
        if str(val).lower() in ("true", "1", "yes", "on"):
            return True
        return False
    elif val_type_lower in ("integer", "number", "int", "float"):
        try:
            if val_type_lower in ("integer", "int"):
                return int(float(val)) if val not in (None, "") else 0
            else:
                return float(val) if val not in (None, "") else 0.0
        except Exception:
            return 0 if val_type_lower in ("integer", "int") else 0.0
    else:
        # String, array, object, list, anyOf, etc.
        if val is None:
            return ""
        if isinstance(val, (list, dict)):
            return "" if not val else json.dumps(val)
        return str(val)


def resolve_placeholders(val: Any, run_results: list[dict], target_type: str = "string") -> Any:
    """
    Recursively search and resolve placeholders in the format {{step_N}} or {{step_N.key.subkey}}
    using execution results of previous steps.
    """
    import re
    import json
    
    def pretty_format_data(data: Any) -> str:
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                k_pretty = k.replace("_", " ").title()
                if isinstance(v, (dict, list)):
                    v_str = json.dumps(v, indent=2)
                else:
                    v_str = str(v)
                lines.append(f"- **{k_pretty}**: {v_str}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"- {json.dumps(item)}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)
        return str(data)

    if isinstance(val, str):
        pattern = r"\{\{\s*step_(\d+)(?:\.([a-zA-Z0-9_\-\.]+))?\s*\}\}"
        
        # If the string is exactly a single placeholder, return the resolved value
        match = re.fullmatch(pattern, val.strip())
        if match:
            step_num = int(match.group(1))
            path = match.group(2)
            result = next((r for r in run_results if r["step_idx"] == step_num), None)
            if not result or not result["success"]:
                return ""
            data = result["data"]
            if not path:
                resolved = data
            else:
                # Walk the dot-separated path in data
                parts = path.split(".")
                curr = data
                for part in parts:
                    if isinstance(curr, dict) and part in curr:
                        curr = curr[part]
                    elif isinstance(curr, list):
                        try:
                            curr = curr[int(part)]
                        except (ValueError, IndexError):
                            return ""
                    else:
                        try:
                            curr = getattr(curr, part)
                        except AttributeError:
                            return ""
                resolved = curr

            # Format if target_type is string and the resolved value is complex
            if target_type == "string" and isinstance(resolved, (dict, list)):
                return pretty_format_data(resolved)
            return resolved

        # Otherwise, mixed string replacement
        def replace_match(m):
            step_num = int(m.group(1))
            path = m.group(2)
            result = next((r for r in run_results if r["step_idx"] == step_num), None)
            if not result or not result["success"]:
                return ""
            data = result["data"]
            if not path:
                resolved = data
            else:
                parts = path.split(".")
                curr = data
                for part in parts:
                    if isinstance(curr, dict) and part in curr:
                        curr = curr[part]
                    elif isinstance(curr, list):
                        try:
                            curr = curr[int(part)]
                        except (ValueError, IndexError):
                            return ""
                    else:
                        try:
                            curr = getattr(curr, part)
                        except AttributeError:
                            return ""
                resolved = curr
            
            if isinstance(resolved, (dict, list)):
                return pretty_format_data(resolved)
            return str(resolved)

        return re.sub(pattern, replace_match, val)

    elif isinstance(val, list):
        return [resolve_placeholders(item, run_results, target_type) for item in val]
    elif isinstance(val, dict):
        return {k: resolve_placeholders(v, run_results, target_type) for k, v in val.items()}
    return val



# ─── Tool Branding Helper ─────────────────────────────────────────────────────

def get_tool_branding(tool_name: str) -> tuple[str, str]:
    name = tool_name.lower()
    if "slack" in name:
        return "#E01E5A", "💬 Slack"
    elif "reddit" in name:
        return "#FF4500", "🤖 Reddit"
    elif "gmail" in name:
        return "#EA4335", "✉️ Gmail"
    elif "linkedin" in name:
        return "#0077B5", "💼 LinkedIn"
    elif "meet" in name:
        return "#00897B", "📹 Google Meet"
    elif "google" in name or "calendar" in name:
        return "#4285F4", "📅 Google"
    elif "github" in name:
        return "#24292E", "🐙 GitHub"
    elif "notion" in name:
        return "#000000", "📝 Notion"
    elif "jira" in name:
        return "#0052CC", "📋 Jira"
    elif "linear" in name:
        return "#5E6AD2", "📐 Linear"
    elif "todoist" in name:
        return "#E44332", "✅ Todoist"
    return "#555555", "🔧 Custom Tool"


# ─── Sidebar: Connected Integrations ──────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Workflow Integrations")
    st.caption("Connect tools to build and run actions.")
    
    if st.button("🔄 Refresh Integrations", use_container_width=True):
        st.session_state.status_refresh += 1
        get_cached_session.clear()
        get_tools_map.clear()
        st.rerun()
        
    st.divider()
    
    try:
        sess = get_session()
        toolkits = get_toolkit_status(sess)
        st.session_state["toolkits_status"] = toolkits
        save_connected_toolkits_to_sqlite(st.session_state.user_id, toolkits)
    except Exception as exc:
        st.error(f"Error accessing integrations: {exc}")
        toolkits = []
        st.session_state["toolkits_status"] = []
        
    # Render connection states
    for tk in toolkits:
        icon = tk["icon"]
        name = tk["name"]
        slug = tk["slug"]
        connected = tk["connected"]
        status_label = "🟢 Connected" if connected else "⚪ Disconnected"
        
        with st.container(border=True):
            st.markdown(f"**{icon} {name}**")
            st.caption(status_label)
            if not connected:
                if st.button(f"Connect {name}", key=f"connect_{slug}", use_container_width=True):
                    try:
                        url = get_connect_url(sess, slug)
                        st.session_state.connect_url = url
                        st.session_state.connect_label = name
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initiate OAuth: {e}")
                        
    if st.session_state.connect_url:
        st.divider()
        st.markdown(f"### Link **{st.session_state.connect_label}**")
        st.link_button(
            "Open authorization page",
            st.session_state.connect_url,
            use_container_width=True,
        )
        st.caption("Complete OAuth in browser, then click Refresh status.")
        if st.button("Clear Link", use_container_width=True):
            st.session_state.connect_url = None
            st.session_state.connect_label = None
            st.rerun()
            
    st.divider()
    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.conv_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    if st.button("🗑️ Clear Active Workflow", use_container_width=True):
        st.session_state.workflow = {
            "name": "Untitled Workflow",
            "description": "Describe the workflow you want to build in the chat! (e.g. 'Reddit to Slack summaries')",
            "steps": []
        }
        # Clear fields
        for k in list(st.session_state.keys()):
            if k.startswith("wf_step_"):
                del st.session_state[k]
        st.rerun()


# ─── Main Interface ───────────────────────────────────────────────────────────

st.title("⚡ Connected-Apps Workflow Builder")
st.caption("Discuss a workflow with the agent on the left, then refine parameters and run it on the right.")
st.divider()

# High visibility OAuth check
if st.session_state.connect_url:
    st.warning(f"🔓 Connection required: please authenticate **{st.session_state.connect_label}** via the sidebar button or click [here]({st.session_state.connect_url}) to authorize.")

# Split columns
col_chat, col_workflow = st.columns([1, 1], gap="large")


# ─── LEFT COLUMN: Chat Interface ──────────────────────────────────────────────

with col_chat:
    st.subheader("💬 Conversations")
    
    # Message logs
    chat_container = st.container(height=500, border=True)
    with chat_container:
        for msg in st.session_state.messages:
            role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
            icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(
                f"""
                <div class="{role_class}">
                    <strong>{icon} {msg['role'].capitalize()}:</strong><br/>
                    {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # Chat inputs
    user_input = st.chat_input(
        "Ask the agent to construct a workflow...",
        disabled=st.session_state.agent_running
    )
    
    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()
        
    # Check if needs agent response
    needs_response = (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and not st.session_state.agent_running
    )
    
    if needs_response:
        st.session_state.agent_running = True
        
        with chat_container:
            with st.status("Agent searching and generating...", expanded=True) as status:
                try:
                    session = get_session()
                    conv_id = st.session_state.conv_id
                    
                    # Mutable holder for the agent tool callbacks
                    workflow_holder = {}
                    final_ans = "No reply received."
                    
                    for event in run_workflow_agent_stream(session, st.session_state.messages, conv_id, workflow_holder):
                        ev_type = event["type"]
                        if ev_type == "thought":
                            st.markdown(f"💭 **Thought:** {event['content']}")
                        elif ev_type == "tool_call":
                            st.markdown(f"🔧 **Tool Call:** `{event['name']}` args: `{event['args']}`")
                            print(f"\n[Console - Agent Tool Call] name={event['name']} arguments={event['args']}\n", flush=True)
                        elif ev_type == "tool_output":
                            preview = event["content"][:200] + "..." if len(event["content"]) > 200 else event["content"]
                            st.markdown(f"📦 **Output (`{event['name']}`):**\n```\n{preview}\n```")
                            print(f"\n[Console - Tool Response] name={event['name']} return={event['content']}\n", flush=True)
                        elif ev_type == "final_answer":
                            final_ans = event["content"]
                            
                    status.update(label="Processed", state="complete", expanded=False)
                    
                    # Check if the agent defined a workflow during execution
                    if workflow_holder:
                        # Clear old input fields from state to prevent leakage
                        for k in list(st.session_state.keys()):
                            if k.startswith("wf_step_"):
                                del st.session_state[k]
                        st.session_state["workflow"] = workflow_holder
                        st.toast("⚡ Workflow updated successfully!", icon="🔥")
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
                    
                except Exception as exc:
                    status.update(label="Error in execution", state="error")
                    st.error(f"Agent turn failed: {exc}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Failed due to error: {exc}"})
                finally:
                    st.session_state.agent_running = False
                    
        st.rerun()


# ─── RIGHT COLUMN: Workflow Builder & Param Forms ─────────────────────────────

with col_workflow:
    st.subheader("⚡ Workflow Parameters")
    
    workflow = st.session_state["workflow"]
    steps = workflow.get("steps", [])
    
    # Render Workflow details
    st.markdown(f"### 📋 {workflow.get('name', 'Untitled Workflow')}")
    st.markdown(f"*{workflow.get('description', '')}*")
    st.divider()
    
    if not steps:
        st.info("💡 **No active workflow.** Ask the assistant on the left to design a workflow for you, such as:\n\n*\"Create a workflow that gets the hot posts on python subreddit and drafts an email in gmail with a summary\"*")
    else:
        # Load tools mapping
        tools_map = get_tools_map(st.session_state.user_id)
        
        # Draw workflow forms step-by-step
        for idx, step in enumerate(steps):
            if not isinstance(step, dict) or "tool_name" not in step:
                continue
            tool_name = step["tool_name"]
            desc = step.get("step_description", "")
            pre_filled = step.get("parameters", {})
            fields = step.get("fields", [])
            
            # Fetch branding
            border_color, brand_name = get_tool_branding(tool_name)
            
            st.markdown(
                f"""
                <div style="border-left: 5px solid {border_color}; padding-left: 12px; margin-bottom: 8px;">
                    <strong style="font-size: 1.1em; color: {border_color};">Step {idx+1}: {brand_name}</strong><br/>
                    <small style="color: #8c92a0;">Action: <code>{tool_name}</code></small>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if desc:
                st.caption(desc)
                
            # Form card container
            with st.container(border=True):
                # Search case-insensitively
                t_obj = None
                for name, obj in tools_map.items():
                    if name.lower().strip() == tool_name.lower().strip():
                        t_obj = obj
                        break
                        
                # Determine connection status based on toolkit connection states
                tk_slug = tool_name.split("_")[0].lower() if "_" in tool_name else tool_name.lower()
                if tk_slug == "google":
                    tk_slug = "googlecalendar"
                    
                toolkits_status = st.session_state.get("toolkits_status", [])
                is_connected = False
                for tk in toolkits_status:
                    if tk["slug"].lower() == tk_slug and tk.get("connected"):
                        is_connected = True
                        break
                        
                if not is_connected:
                    st.warning(f"⚠️ {tk_slug.upper()} is not connected. Please connect it in the sidebar to run this step.")
                
                if fields and isinstance(fields, list):
                    # Render parameters based on the fields list provided by the agent
                    for field in fields:
                        try:
                            if not isinstance(field, dict) or "name" not in field:
                                continue
                            param_name = field["name"]
                            param_type = field.get("type", "string")
                            tooltip = field.get("description", "")
                            default_val = field.get("value", "")
                            
                            state_key = f"wf_step_{idx}_{param_name}"
                            
                            # Init value if not already in session state
                            if state_key not in st.session_state:
                                st.session_state[state_key] = coerce_value_for_widget(default_val, param_type)
                                
                            label = f"{param_name.replace('_', ' ').capitalize()} ({param_name})"
                            
                            if param_type == "boolean":
                                st.checkbox(label, key=state_key, help=tooltip)
                            elif param_type in ("integer", "number"):
                                st.number_input(label, key=state_key, help=tooltip, step=1 if param_type == "integer" else 0.1)
                            else:
                                # String or other
                                long_text_triggers = ["text", "body", "message", "description", "content", "summary"]
                                is_long = any(trig in str(param_name).lower() or trig in str(tooltip).lower() for trig in long_text_triggers)
                                if is_long:
                                    st.text_area(label, key=state_key, help=tooltip)
                                else:
                                    st.text_input(label, key=state_key, help=tooltip)
                        except Exception as field_err:
                            st.warning(f"Could not render field: {field_err}")
                elif t_obj:
                    # Fallback: render inputs based on live Composio arguments schema
                    for param_name, param_info in t_obj.args.items():
                        try:
                            state_key = f"wf_step_{idx}_{param_name}"
                            param_type = param_info.get("type", "string")
                            
                            if state_key not in st.session_state:
                                default_val = param_info.get("default", "")
                                if param_name in pre_filled:
                                    default_val = pre_filled[param_name]
                                
                                st.session_state[state_key] = coerce_value_for_widget(default_val, param_type)
                                        
                            label = param_info.get("title", param_name)
                            tooltip = param_info.get("description", "")
                            
                            if "enum" in param_info:
                                st.selectbox(label, options=param_info["enum"], key=state_key, help=tooltip)
                            elif param_type == "boolean":
                                st.checkbox(label, key=state_key, help=tooltip)
                            elif param_type in ("integer", "number"):
                                st.number_input(label, key=state_key, help=tooltip, step=1 if param_type == "integer" else 0.1)
                            elif param_type == "string":
                                long_text_triggers = ["text", "body", "message", "description", "content", "summary"]
                                is_long = any(trig in str(param_name).lower() or trig in str(tooltip).lower() for trig in long_text_triggers)
                                if is_long:
                                    st.text_area(label, key=state_key, help=tooltip)
                                else:
                                    st.text_input(label, key=state_key, help=tooltip)
                            else:
                                st.text_input(label, key=state_key, help=tooltip)
                        except Exception as param_err:
                            st.warning(f"Could not render parameter: {param_err}")
                else:
                    st.info("No parameters defined for this tool step.")
            st.write("") # Spacer
            
        # Action Buttons
        st.divider()
        col_run, col_export = st.columns([1, 1])
        
        with col_run:
            if st.button("🚀 Run Workflow", type="primary", use_container_width=True):
                with st.status("Executing workflow...", expanded=True) as run_status:
                    success = True
                    st.session_state["workflow_run_results"] = [] # Clear previous runs
                    
                    for idx, step in enumerate(steps):
                        t_name = step["tool_name"]
                        run_status.write(f"⏳ **Step {idx+1}:** Invoking `{t_name}`...")
                        
                        # Find the tool case-insensitively
                        t_obj = None
                        for name, obj in tools_map.items():
                            if name.lower().strip() == t_name.lower().strip():
                                t_obj = obj
                                break
                                
                        # Gather actual arguments from state
                        tool_args = {}
                        if t_obj:
                            for arg, arg_info in t_obj.args.items():
                                state_key = f"wf_step_{idx}_{arg}"
                                if state_key in st.session_state:
                                    val = st.session_state[state_key]
                                    arg_type = arg_info.get("type", "string")
                                    
                                    # Resolve step placeholders dynamically
                                    val = resolve_placeholders(val, st.session_state["workflow_run_results"], arg_type)
                                    
                                    if arg_type in ("array", "object") and isinstance(val, str):
                                        val = val.strip()
                                        if not val:
                                            val = [] if arg_type == "array" else {}
                                        else:
                                            try:
                                                val = json.loads(val)
                                            except Exception:
                                                if arg_type == "array":
                                                    val = [v.strip() for v in val.split(",") if v.strip()]
                                    # Omit optional empty values (empty strings, empty lists, empty dicts)
                                    if val not in ("", [], {}):
                                        tool_args[arg] = val
                        else:
                            # Gather from step fields
                            for field in step.get("fields", []):
                                arg = field["name"]
                                arg_type = field.get("type", "string")
                                state_key = f"wf_step_{idx}_{arg}"
                                if state_key in st.session_state:
                                    val = st.session_state[state_key]
                                    
                                    # Resolve step placeholders dynamically
                                    val = resolve_placeholders(val, st.session_state["workflow_run_results"], arg_type)
                                    
                                    if arg_type in ("array", "object", "list") and isinstance(val, str):
                                        val = val.strip()
                                        if not val:
                                            val = [] if arg_type in ("array", "list") else {}
                                        else:
                                            try:
                                                val = json.loads(val)
                                            except Exception:
                                                if arg_type in ("array", "list"):
                                                    val = [v.strip() for v in val.split(",") if v.strip()]
                                    # Omit optional empty values (empty strings, empty lists, empty dicts)
                                    if val not in ("", [], {}):
                                        tool_args[arg] = val
                                    
                        # Check connection status of this specific toolkit
                        tk_slug = t_name.split("_")[0].lower() if "_" in t_name else t_name.lower()
                        if tk_slug == "google":
                            tk_slug = "googlecalendar"
                            
                        toolkits_status = st.session_state.get("toolkits_status", [])
                        is_connected = False
                        for tk in toolkits_status:
                            if tk["slug"].lower() == tk_slug and tk.get("connected"):
                                is_connected = True
                                break
                                
                        if not is_connected:
                            err_msg = f"Toolkit `{tk_slug.upper()}` is not connected in your sidebar."
                            run_status.write(f"❌ **Step {idx+1} Failed!** Error: {err_msg}")
                            print(f"\n[EXECUTE WORKFLOW STEP FAILED] step={idx+1}/{len(steps)} action={t_name} error={err_msg}", flush=True)
                            st.session_state["workflow_run_results"].append({
                                "step_idx": idx + 1,
                                "tool_name": t_name,
                                "success": False,
                                "error": err_msg,
                                "data": None
                            })
                            success = False
                            break
                            
                        try:
                            # Print to terminal
                            print(f"\n[EXECUTE WORKFLOW STEP] step={idx+1}/{len(steps)} action={t_name}", flush=True)
                            print(f"Arguments: {tool_args}", flush=True)
                            
                            # Invoke Composio tool directly via Python SDK with skip check
                            from src.agents.composio_agent import get_composio
                            comp = get_composio()
                            
                            result = comp.tools.execute(
                                slug=t_name,
                                arguments=tool_args,
                                user_id=st.session_state.user_id,
                                dangerously_skip_version_check=True
                            )
                            
                            successful = False
                            error_msg = ""
                            result_data = None
                            
                            if isinstance(result, dict):
                                successful = result.get("successful", False)
                                error_msg = result.get("error")
                                result_data = result.get("data")
                            else:
                                successful = getattr(result, "successful", False)
                                error_msg = getattr(result, "error", None)
                                result_data = getattr(result, "data", None)
                                
                            # Print result to terminal
                            print(f"[EXECUTE WORKFLOW STEP RESPONSE] successful={successful} error={error_msg}", flush=True)
                            print(f"Payload: {result_data}\n", flush=True)
                            
                            st.session_state["workflow_run_results"].append({
                                "step_idx": idx + 1,
                                "tool_name": t_name,
                                "success": successful,
                                "error": error_msg,
                                "data": result_data
                            })
                            
                            if successful:
                                run_status.write(f"✅ **Step {idx+1} Success!** Result:")
                                run_status.code(str(result_data)[:1000])
                            else:
                                run_status.write(f"❌ **Step {idx+1} Failed!** Error: {error_msg or 'Execution unsuccessful'}")
                                if result_data:
                                    run_status.code(str(result_data)[:1000])
                                success = False
                                break
                        except Exception as err:
                            error_trace = traceback.format_exc()
                            print(f"[EXECUTE WORKFLOW STEP EXCEPTION] step={idx+1}/{len(steps)} action={t_name} error={err}\n{error_trace}", flush=True)
                            
                            st.session_state["workflow_run_results"].append({
                                "step_idx": idx + 1,
                                "tool_name": t_name,
                                "success": False,
                                "error": str(err),
                                "data": error_trace
                            })
                            
                            run_status.write(f"❌ **Step {idx+1} Failed!** Error: {err}")
                            run_status.code(error_trace)
                            success = False
                            break
                            
                    if success:
                        run_status.update(label="Workflow completed successfully!", state="complete")
                    else:
                        run_status.update(label="Workflow halted due to error.", state="error")
                        
        with col_export:
            # Prepare export JSON
            export_data = {
                "name": workflow["name"],
                "description": workflow["description"],
                "steps": []
            }
            for idx, step in enumerate(steps):
                t_name = step["tool_name"]
                t_args = {}
                t_obj = None
                for name, obj in tools_map.items():
                    if name.lower().strip() == t_name.lower().strip():
                        t_obj = obj
                        break
                        
                if t_obj:
                    for arg in t_obj.args.keys():
                        state_key = f"wf_step_{idx}_{arg}"
                        if state_key in st.session_state:
                            t_args[arg] = st.session_state[state_key]
                else:
                    for field in step.get("fields", []):
                        arg = field["name"]
                        state_key = f"wf_step_{idx}_{arg}"
                        if state_key in st.session_state:
                            t_args[arg] = st.session_state[state_key]
                            
                export_data["steps"].append({
                    "tool_name": t_name,
                    "step_description": step.get("step_description", ""),
                    "parameters": t_args
                })
                
            st.download_button(
                label="📥 Export Workflow (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"{workflow.get('name', 'workflow').lower().replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
            
    # ─── Persistent Execution Results Proof ───────────────────────────────────────
    if "workflow_run_results" in st.session_state and st.session_state["workflow_run_results"]:
        st.divider()
        st.markdown("### 📝 Execution Logs & Payload Proof")
        for res in st.session_state["workflow_run_results"]:
            status_emoji = "✅" if res["success"] else "❌"
            with st.expander(
                label=f"Step {res['step_idx']}: {res['tool_name']} {status_emoji}",
                expanded=True
            ):
                if res["success"]:
                    st.success(f"Action '{res['tool_name']}' completed successfully!")
                    if res["data"]:
                        st.json(res["data"])
                    else:
                        st.info("Execution returned no response payload.")
                else:
                    st.error(f"Action failed with error: {res['error']}")
                    if res["data"]:
                        if isinstance(res["data"], (dict, list)):
                            st.json(res["data"])
                        else:
                            st.code(str(res["data"]))
