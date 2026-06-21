"""
AI Agent Chat UI — plain Streamlit, no custom CSS.
"""
import os
import sys
import threading
import time
import uuid
from pathlib import Path

# ── Load .env FIRST so every os.environ lookup gets the right values ──────────
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent   # project root  (ai_flow/)
load_dotenv(ROOT / ".env", override=True)       # explicit path, always works

# ── Make src/ importable ──────────────────────────────────────────────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
# Reload all modules in topological order so that changes to utils and configs propagate correctly
RELOAD_MODULES = [
    "src.config",
    "src.utils.entity_extractor",
    "src.utils.graph_store",
    "src.utils.vector_store",
    "src.utils.keyword_search",
    "src.utils.hybrid_search",
    "src.agents.state",
    "src.agents.rag_agent",
    "src.pipelines.ingestion",
]
for mod in RELOAD_MODULES:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass


from src.config import settings

print("[app.py] Active OpenAI API Key loaded:", bool(settings.openai_api_key), flush=True)
from src.utils import event_bus

import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Session state defaults ───────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "events": [],
        "conv_id": str(uuid.uuid4()),
        "agent_running": False,
        "hitl_pending": False,
        "hitl_state": None,
        "uploaded_path": None,
        "uploaded_name": None,
        "auth_pending": False,
        "auth_state": None,   # {"tool_name": ..., "auth_id": ..., "auth_url": ...}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Background ingestion ─────────────────────────────────────────────────────
def run_background_ingestion(file_path: str, doc_id: str, conv_id: str, checksum: str, original_filename: str):
    try:
        from src.pipelines.ingestion import ingest_document_pipeline
        ingest_document_pipeline(
            file_path,
            doc_id,
            conv_id,
            checksum=checksum,
            original_filename=original_filename,
        )
    except Exception as e:
        print(f"[ingestion] Background ingestion failed for {original_filename}: {e}", flush=True)


def start_background_ingestion(file_path: str, doc_id: str, conv_id: str, checksum: str, original_filename: str):
    t = threading.Thread(
        target=run_background_ingestion,
        args=(file_path, doc_id, conv_id, checksum, original_filename),
        daemon=True,
    )
    t.start()


# ─── Agent runner ─────────────────────────────────────────────────────────────
def run_agent_turn(user_query: str, uploaded_path, conv_id: str):
    from src.agents.rag_agent import compile_agent, pre_check_query
    from src.agents.state import create_initial_state

    event_bus.clear()

    # Pre-check guardrails and fast-path greetings
    pre_result = pre_check_query(user_query, conv_id, uploaded_path)
    if pre_result:
        return pre_result

    agent = compile_agent()
    state = create_initial_state(
        user_query=user_query,
        conversation_id=conv_id,
        uploaded_file_path=uploaded_path,
    )
    config = {"configurable": {"thread_id": conv_id}}
    result = agent.invoke(state, config=config)

    # Check if the graph paused for OAuth authorization
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        for intr in interrupts:
            val = intr.value if hasattr(intr, "value") else intr
            if isinstance(val, dict) and val.get("type") == "authorization_required":
                auth = val.get("auth_response", {})
                result["_auth_interrupt"] = {
                    "tool_name": val.get("tool_name", "unknown tool"),
                    "auth_id":   auth.get("id"),
                    "auth_url":  auth.get("url"),
                }
                break
    return result


def resume_agent_after_auth(conv_id: str, authorized: bool):
    """Resume the graph after OAuth is completed (or rejected)."""
    from src.agents.rag_agent import compile_agent
    from langgraph.types import Command
    agent = compile_agent()
    config = {"configurable": {"thread_id": conv_id}}
    return agent.invoke(Command(resume={"authorized": authorized}), config=config)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

@st.fragment(run_every=1.0)
def render_sidebar_progress(conv_id: str):
    from src.pipelines.ingestion import INGESTION_PROGRESS
    if conv_id in INGESTION_PROGRESS:
        docs_progress = INGESTION_PROGRESS[conv_id]
        # Check if any ingestion is running
        running = any(p.get("status") not in ("completed", "failed") for p in docs_progress.values())
        if running:
            st.caption("⚡ Background ingestion active...")
            
        for doc_id, prog in list(docs_progress.items()):
            filename = prog.get("filename", doc_id)
            percent = prog["percent"]
            status = prog["status"]
            details = prog["details"]
            
            expanded = status not in ("completed", "failed")
            with st.expander(f"📄 {filename}", expanded=expanded):
                if status == "completed":
                    st.success("🎉 Ingestion Complete!")
                    st.markdown(details)
                elif status == "failed":
                    st.error("❌ Ingestion Failed")
                    st.caption(f"Reason: {details}")
                else:
                    st.progress(percent / 100.0, text=f"**{percent}% — {status}**")
                    if details:
                        st.caption(details)
    else:
        st.info("No documents uploaded in this thread yet. Upload a file below to build memory!")

# ─── Sidebar Progress ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Knowledge Builder")
    st.markdown(
        "Builds a knowledge base (Pinecone vector store, BM25 index, and Neo4j entities graph) "
        "in the background while you chat."
    )
    st.divider()
    if st.button("View relationships", use_container_width=True):
        st.switch_page("pages/relationships.py")
    st.divider()

    render_sidebar_progress(st.session_state.conv_id)

st.title("⚡ AI Agent")
st.caption("Agentic search, document intelligence, and connected knowledge.")
st.divider()

# ── Chat messages ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

# ── Live events (shown after messages) ───────────────────────────────────────
if st.session_state.events:
    with st.expander("🔍 Agent trace", expanded=True):
        for e in st.session_state.events[-20:]:
            st.caption(e.get("message", ""))

# (HITL long-term memory checkpoint removed)

# ── MCP OAuth authorization interrupt ─────────────────────────────────────────
if st.session_state.auth_pending and st.session_state.auth_state:
    auth = st.session_state.auth_state
    tool_name = auth.get("tool_name", "an external tool")
    auth_url  = auth.get("auth_url", "")
    
    with st.container(border=True):
        st.markdown("### 🔐 Authorization Required")
        st.write(f"The AI Agent needs permission to access external tool: **`{tool_name}`**")
        
        if auth_url:
            st.link_button("🔑 Grant Access / Authorize", auth_url, use_container_width=True)
            
        st.caption("Please click the button above to authorize in the browser tab, then return here and click **I've Authorized**.")
        st.divider()
        
        col_auth, col_deny = st.columns(2)
        with col_auth:
            if st.button("✅ I've Authorized", type="primary", key="auth_accept", use_container_width=True):
                result = resume_agent_after_auth(st.session_state.conv_id, authorized=True)
                answer = result.get("final_answer", "Authorization complete — please re-ask your question.")
                st.session_state.messages.append({"role": "agent", "content": answer})
                st.session_state.auth_pending = False
                st.session_state.auth_state = None
                st.rerun()
        with col_deny:
            if st.button("✗ Cancel", key="auth_deny", use_container_width=True):
                st.session_state.messages.append({
                    "role": "agent",
                    "content": f"❌ Authorization for `{tool_name}` was cancelled."
                })
                st.session_state.auth_pending = False
                st.session_state.auth_state = None
                st.rerun()

st.divider()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Attach a file (optional)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "md"],
    key="file_upload",
)

if uploaded_file is None and st.session_state.uploaded_name is not None:
    # Allow removing and re-selecting the same file after a failed ingestion.
    st.session_state.uploaded_name = None
    st.session_state.uploaded_path = None

if uploaded_file:
    from src.utils.uploads import store_upload

    try:
        stored = store_upload(
            uploaded_file.name,
            uploaded_file.getvalue(),
            st.session_state.conv_id,
        )
        upload_key = f"{stored.document_id}:{stored.safe_name}"
        if upload_key != st.session_state.uploaded_name:
            st.session_state.uploaded_path = stored.path
            st.session_state.uploaded_name = upload_key
            message = (
                f"**{stored.safe_name}** received — indexing in the background…"
                if stored.created
                else f"**{stored.safe_name}** was already uploaded — reusing its safe stored copy."
            )
            st.session_state.messages.append({"role": "agent", "content": message})
            start_background_ingestion(
                stored.path,
                stored.document_id,
                st.session_state.conv_id,
                stored.checksum,
                stored.safe_name,
            )
            st.rerun()
    except ValueError as exc:
        st.error(str(exc))

# ── Chat input (Streamlit native — always at bottom) ─────────────────────────
user_input = st.chat_input("Ask anything…", disabled=st.session_state.agent_running)

if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    st.session_state.events = []
    st.rerun()

# ─── Auto-run agent ───────────────────────────────────────────────────────────
needs_response = (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and not st.session_state.agent_running
    and not st.session_state.hitl_pending
    and not st.session_state.auth_pending
)

if needs_response:
    query         = st.session_state.messages[-1]["content"]
    uploaded_path = st.session_state.uploaded_path
    # Clear file path from state immediately so subsequent turns do not pass it
    st.session_state.uploaded_path = None
    conv_id       = st.session_state.conv_id
    st.session_state.agent_running = True

    with st.status("⚡ Agent thinking…", expanded=True) as status:
        events_ph = st.empty()

        result_holder = [None]
        error_holder  = [None]

        def _agent_thread():
            try:
                result_holder[0] = run_agent_turn(query, uploaded_path, conv_id)
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_agent_thread, daemon=True)
        t.start()

        all_events = []
        while t.is_alive():
            new = event_bus.get_all()
            if new:
                all_events.extend(new)
                st.session_state.events = all_events.copy()
                events_ph.caption("  \n".join(e.get("message", "") for e in all_events[-10:]))
            time.sleep(0.15)

        all_events.extend(event_bus.get_all())
        st.session_state.events = all_events
        result = result_holder[0]
        st.session_state.agent_running = False

        if error_holder[0]:
            status.update(label="❌ Agent failed", state="error")
            st.session_state.messages.append({
                "role": "agent",
                "content": f"Sorry, something went wrong:\n\n{error_holder[0]}"
            })
        elif result:
            answer = result.get("final_answer", "No answer returned.")

            # ── OAuth interrupt: show auth link instead of an answer ──────────
            if result.get("_auth_interrupt"):
                auth_info = result["_auth_interrupt"]
                st.session_state.auth_pending = True
                st.session_state.auth_state = auth_info
                status.update(label="🔐 Authorization required", state="complete", expanded=False)
                st.session_state.messages.append({
                    "role": "agent",
                    "content": (
                        f"🔐 **I need access to `{auth_info['tool_name']}`.**\n\n"
                        f"Please authorize below so I can continue."
                    )
                })
            else:
                status.update(label="✅ Done", state="complete", expanded=False)
                st.session_state.messages.append({"role": "agent", "content": answer})
        else:
            status.update(label="⚠️ No response", state="error")

    st.rerun()
