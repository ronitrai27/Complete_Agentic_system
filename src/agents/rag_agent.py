"""
RAG Agent — Main LangGraph graph.

Architecture:
  START
    │
  [router]  ──────────────────────────────────────────┐
    │                                                  │
    │  route="search"                                  │  route="mcp"
    ▼                                                  ▼
  [web_search_dispatcher] ── Send() ──► [tavily_worker]  [mcp_agent]
                               └──────► [serpapi_worker]     │
                                                │            │
                                        [aggregate_search]   │
                                                │            │
                                      ┌─────────┴────────────┘
                                      │   route="rag"     route="direct"
                                      │      ▼                ▼
                                      │  [rag_retrieve]   (skip)
                                      │      │
                                      └──────┴──────────────────────►
                                                                    [llm_answer]
                                                                        │
                                                                  [hitl_checkpoint]  ← interrupt()
                                                                        │
                                                            ┌───────────┴──────────┐
                                                     approved=True         approved=False
                                                            ▼                      ▼
                                                   [save_conversation]            END
                                                            │
                                                           END
"""

import asyncio
import os
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from loguru import logger

from src.agents.state import AgentState, HitlDecision, Message, RagContext, SearchResult
from src.config import settings
from src.tools.conversation_store import (
    get_recent_conversation_history,
    record_conversation_turn,
    save_search_results,
    upsert_conversation,
)
from src.tools.search import search_web, search_web_tavily
from src.utils.event_bus import emit
from src.utils.hybrid_search import get_hybrid_context
from src.utils.parser import parse_file

# ─── LLM Client ───────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model="gpt-4.1-mini",  # Upgraded OpenAI model
        temperature=temperature,
        api_key=api_key,
    )


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a highly intelligent research assistant with access to:
- Web search results (Tavily + Google/SerpAPI)
- A personal knowledge base (Pinecone vector store + Neo4j knowledge graph + BM25 keyword index)
- External tools (Gmail, Google Docs, Notion via Arcade MCP)

Your job is to give precise, well-structured answers. When you have both web search
and knowledge base context, synthesize them. Always cite your sources if you know them.
If the user uploaded a file, use the file content to answer their question directly.

CRITICAL INSTRUCTIONS:
1. Do NOT hallucinate. Only state facts directly supported by the retrieved context.
2. If the context does not contain enough information to answer the question, state clearly that you do not know the answer based on the provided documents. Do not make up answers.
3. Be highly factual and objective.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Router
# Analyzes the query and decides which path to take.
# ═══════════════════════════════════════════════════════════════════════════════

ROUTER_PROMPT = """Given the user query below, decide which retrieval route to use.
Reply with EXACTLY one of these words (no other text):
  - search    → query needs live/current/external web info, news, prices, weather, recent events, real-time facts not in the database.
                Examples: "latest news about AI", "current Bitcoin price", "what happened yesterday"
  - rag       → query asks about specific people, companies, departments, projects, skills, features, documents, tasks, stored/uploaded data — if the answer lives in a private database, knowledge base, or uploaded file, always pick rag.
                Examples: "who is Sarah?", "what does Rohan work on?", "find AI assistant owner", "what does the uploaded doc say?"
  - mcp       → query specifically needs to interact with or read Gmail, Google Docs, Notion, Outlook.
                Examples: "read my email", "show my latest threads", "check my inbox"
  - direct    → ONLY for simple greetings, small talk, or questions about the assistant's own capabilities. NOT for factual/entity queries.
                Examples: "hello", "what can you do?", "thanks", "who are you"

{document_context}
User query: {query}
Route:"""

def router(state: AgentState) -> Dict:
    emit("🧭 Router: analysing query to decide retrieval path...", "step")
    user_query = state.get("user_query", "")
    uploaded_file_path = state.get("uploaded_file_path")
    fast_path_text = state.get("fast_path_text")
    conversation_id = state.get("conversation_id", "")
    
    doc_context = ""
    if (uploaded_file_path and os.path.exists(uploaded_file_path)) or fast_path_text:
        doc_context = "[Context: A document has been uploaded in this session. If the user query is about the document, or refers to 'the document', 'the file', 'this', or 'it' in the context of the upload, choose 'rag'.]"

    logger.info(f"[Router] Deciding route for: '{user_query[:80]}...'")
    recent_history = get_recent_conversation_history(
        conversation_id,
        max_messages=6,
        max_characters=4000,
    )
    history_text = "\n".join(
        f"{item['role']}: {item['content']}" for item in recent_history
    )
    if history_text:
        doc_context += f"\nRecent conversation:\n{history_text}"

    llm = get_llm(temperature=0.0)
    response = llm.invoke([
        HumanMessage(content=ROUTER_PROMPT.format(query=user_query, document_context=doc_context))
    ])
    route_raw = response.content.strip().lower()

    # Normalize — only allow known routes
    route = "direct"
    for valid in ("search", "rag", "mcp"):
        if valid in route_raw:
            route = valid
            break

    emit(f"✅ Route decided: **{route}**", "success")
    logger.info(f"[Router] → route = '{route}'")
    return {"route": route}


def route_decision(state: AgentState):
    """
    Conditional edge from router — follows the orchestrator-worker pattern.

    For 'search': returns List[Send] to fan-out to tavily + serpapi workers in parallel.
    For all other routes: returns a plain string node name.

    This is identical to the pattern in orchestrator-worker.py:
        builder.add_conditional_edges("orchestrator", route_to_workers)
    where route_to_workers returns List[Send] directly.
    """
    route = state.get("route", "direct")
    user_query = state.get("user_query", "")
    conversation_id = state.get("conversation_id")
    
    if route == "search":
        emit("🔀 Dispatching parallel web searches: Tavily + SerpAPI", "step")
        logger.info("[Router] Fanning out to tavily_worker + serpapi_worker via Send()")
        return [
            Send("tavily_worker",  {"user_query": user_query, "conversation_id": conversation_id, "route": route}),
            Send("serpapi_worker", {"user_query": user_query, "conversation_id": conversation_id, "route": route}),
        ]

    mapping = {
        "rag":    "rag_retrieve",
        "mcp":    "mcp_agent",
        "direct": "llm_answer",
    }
    next_node = mapping.get(route, "llm_answer")
    logger.info(f"[Router] Routing to node: '{next_node}'")
    return next_node


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3a — Tavily Worker
# ═══════════════════════════════════════════════════════════════════════════════

def tavily_worker(state: AgentState) -> Dict:
    """Runs Tavily search and appends structured results to state.search_results."""
    emit("🌐 Tavily: running web search...", "step")
    user_query = state.get("user_query", "")
    logger.info(f"[Tavily] Searching: '{user_query}'")
    try:
        raw = search_web_tavily.invoke({"query": user_query})
        results = _parse_tavily_output(raw, user_query)
        emit(f"✅ Tavily: got {len(results)} result(s)", "success")
    except Exception as e:
        emit(f"⚠️ Tavily failed: {e}", "warning")
        logger.error(f"[Tavily] Failed: {e}")
        results = []
    return {"search_results": results}


def _parse_tavily_output(raw: str, query: str) -> List[SearchResult]:
    results = []
    lines = raw.strip().split("\n")
    title = url = snippet = ""
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            if title:
                results.append(SearchResult(source="tavily", title=title, url=url, snippet=snippet))
            title = line.split(".", 1)[-1].strip()
            url = snippet = ""
        elif line.startswith("🔗"):
            url = line.replace("🔗", "").strip().split(" ")[0]
        elif line and not line.startswith("Tavily"):
            snippet = (snippet + " " + line).strip()
    if title:
        results.append(SearchResult(source="tavily", title=title, url=url, snippet=snippet))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3b — SerpAPI Worker
# ═══════════════════════════════════════════════════════════════════════════════

def serpapi_worker(state: AgentState) -> Dict:
    """Runs SerpAPI Google Search and appends structured results to state.search_results."""
    emit("🔍 SerpAPI: running Google search...", "step")
    user_query = state.get("user_query", "")
    logger.info(f"[SerpAPI] Searching: '{user_query}'")
    try:
        raw = search_web.invoke({"query": user_query})
        results = _parse_serpapi_output(raw, user_query)
        emit(f"✅ SerpAPI: got {len(results)} result(s)", "success")
    except Exception as e:
        emit(f"⚠️ SerpAPI failed: {e}", "warning")
        logger.error(f"[SerpAPI] Failed: {e}")
        results = []
    return {"search_results": results}


def _parse_serpapi_output(raw: str, query: str) -> List[SearchResult]:
    results = []
    lines = raw.strip().split("\n")
    title = url = snippet = ""
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            if title:
                results.append(SearchResult(source="serpapi", title=title, url=url, snippet=snippet))
            title = line.split(".", 1)[-1].strip()
            url = snippet = ""
        elif line.startswith("🔗"):
            url = line.replace("🔗", "").strip().split(" ")[0]
        elif line and not line.startswith("Google"):
            snippet = (snippet + " " + line).strip()
    if title:
        results.append(SearchResult(source="serpapi", title=title, url=url, snippet=snippet))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Aggregate Search Results
# Runs after both parallel workers complete (fan-in point).
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_search(state: AgentState) -> Dict:
    """Deduplicates and logs merged search results from both workers."""
    seen_urls = set()
    unique = []
    search_results = state.get("search_results", [])
    for r in search_results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique.append(r)
    emit(f"🔗 Aggregated {len(unique)} unique results from Tavily + SerpAPI", "success")
    logger.info(f"[Aggregate] {len(unique)} unique results from {len(search_results)} total")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — RAG Retrieve
# Runs Hybrid Search: Pinecone + BM25 + Neo4j neighbors.
# Also handles fast-path parsing if a file was uploaded.
# ═══════════════════════════════════════════════════════════════════════════════

def rag_retrieve(state: AgentState) -> Dict:
    updates: Dict = {}
    uploaded_file_path = state.get("uploaded_file_path")
    fast_path_text = state.get("fast_path_text")
    user_query = state.get("user_query", "")

    # Fast-path: parse the uploaded file if a path is supplied
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        emit(f"📄 LlamaParse: parsing uploaded file...", "step")
        logger.info(f"[RAG] Fast-parsing uploaded file: {uploaded_file_path}")
        try:
            fast_text = parse_file(uploaded_file_path)
            updates["fast_path_text"] = fast_text
            emit(f"✅ LlamaParse: extracted {len(fast_text)} characters from document", "success")
        except Exception as e:
            emit(f"⚠️ LlamaParse failed: {e}", "warning")
            logger.error(f"[RAG] Fast-parse failed: {e}")

    # Hybrid retrieval from Pinecone + BM25 + Neo4j
    emit("🧠 Semantic search: querying Pinecone vector store...", "step")
    logger.info(f"[RAG] Running hybrid retrieval for: '{user_query}'")
    try:
        context = get_hybrid_context(
            user_query,
            top_k=5,
            conversation_id=state.get("conversation_id"),
        )
        chunks = context.get("text_chunks", [])
        graph = context.get("graph_context", [])
        extracted_entities = context.get("extracted_entities", [])
        graph_error = context.get("graph_error")
        
        # 1. Pinecone results emission
        vector_chunks = [c for c in chunks if c.get("source") == "vector"]
        emit(f"✅ Pinecone: retrieved {len(vector_chunks)} chunk(s)", "success")
        for i, vc in enumerate(vector_chunks[:3]):
            text_preview = vc["text"][:100].replace('\n', ' ')
            score = vc.get("score", 0.0)
            emit(f"   └─ [{i+1}] (Score: {score:.4f}) {text_preview}...", "step")
            
        # 2. Neo4j results emission
        if graph_error:
            emit(f"❌ Neo4j graph: query failed: {graph_error}", "warning")
        else:
            if extracted_entities:
                emit(f"🕸️ Neo4j graph: extracted entities: {extracted_entities}", "step")
            else:
                emit("🕸️ Neo4j graph: no entities extracted from query", "step")
                
            if graph:
                emit(f"🕸️ Neo4j graph: found {len(graph)} relationship(s)", "success")
                for i, rel in enumerate(graph[:10]):  # Limit trace output to avoid cluttering UI
                    ent = rel.get("entity")
                    r_type = rel.get("relation")
                    neigh = rel.get("neighbor")
                    emit(f"   └─ [{i+1}] ({ent})--[{r_type}]-->({neigh})", "success")
            else:
                emit("🕸️ Neo4j graph: no matching relationships found in database", "step")
                
        # 3. BM25 results emission
        bm25_chunks = [c for c in chunks if c.get("source") == "bm25"]
        emit(f"📚 BM25 keyword search: retrieved {len(bm25_chunks)} match(es)", "success")
        for i, bc in enumerate(bm25_chunks[:3]):
            text_preview = bc["text"][:100].replace('\n', ' ')
            score = bc.get("score", 0.0)
            emit(f"   └─ [{i+1}] (Score: {score:.2f}) {text_preview}...", "step")

        updates["rag_context"] = RagContext(
            text_chunks=chunks,
            graph_context=graph,
        )
    except Exception as e:
        emit(f"⚠️ Hybrid search failed: {e}", "warning")
        logger.error(f"[RAG] Hybrid search failed: {e}")

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — MCP Agent
# Calls Arcade MCP tools (Gmail, Google Docs, Notion) using a ReAct loop.
# ═══════════════════════════════════════════════════════════════════════════════

def mcp_agent(state: AgentState, config: RunnableConfig) -> Dict:
    """
    Invokes Arcade MCP tools asynchronously.
    Uses a simple tool-calling loop: LLM decides which tool to call,
    executes it, appends result, repeats until LLM signals done.

    Authorization interrupts (OAuth) are re-raised so LangGraph can
    pause the graph and surface the auth URL to the UI.
    """
    user_query = state.get("user_query", "")
    emit("🔌 MCP Agent: loading Arcade tools (Gmail, Google Docs, Notion)...", "step")
    logger.info(f"[MCP] Starting MCP agent for: '{user_query}'")
    mcp_results = []

    # Import here to avoid circular imports and to allow optional usage
    from src.tools.mcp import get_arcade_tools
    from arcadepy import AsyncArcade
    # LangGraph interrupt raises this internally — we must NOT catch it
    from langgraph.errors import GraphInterrupt

    async def _run_mcp():
        arcade_client = AsyncArcade(api_key=os.getenv("ARCADE_API_KEY"))
        tools = await get_arcade_tools(arcade_client=arcade_client)

        if not tools:
            logger.warning("[MCP] No tools loaded from Arcade.")
            await arcade_client.close()
            return []

        # Bind tools to LLM
        llm = get_llm().bind_tools(tools)

        # Use a persistent user_id so we don't trigger OAuth for every new conversation
        user_id = "raironit127@gmail.com"

        # Merge graph config with user_id to preserve LangGraph context
        merged_config = config.copy() if config else {}
        if "configurable" not in merged_config:
            merged_config["configurable"] = {}
        merged_config["configurable"]["user_id"] = user_id

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]

        # ReAct loop — max 3 tool calls to avoid runaway
        results = []
        for _ in range(3):
            response = await llm.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break   # LLM is done calling tools

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                matched = next((t for t in tools if t.name == tool_name), None)
                if matched:
                    emit(f"🛠️ MCP: calling tool **{tool_name}**...", "step")
                    logger.info(f"[MCP] Calling tool: {tool_name}")
                    # NOTE: GraphInterrupt must propagate — do NOT catch it here
                    tool_output = await matched.ainvoke(
                        tool_args,
                        config=merged_config
                    )
                    if isinstance(tool_output, dict) and "error" in tool_output:
                        emit(f"⚠️ MCP tool **{tool_name}** returned error: {tool_output['error']}", "warning")
                    else:
                        emit(f"✅ MCP: **{tool_name}** returned result", "success")
                    results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "output": str(tool_output)[:2000],
                    })
                    # Add ToolMessage to the messages history so the ReAct loop is valid in LangChain
                    import json
                    messages.append(
                        ToolMessage(
                            content=json.dumps(tool_output) if not isinstance(tool_output, str) else tool_output,
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    )

        await arcade_client.close()
        return results

    try:
        # Run async MCP logic; GraphInterrupt will propagate through asyncio.run
        mcp_results = asyncio.run(_run_mcp())
    except GraphInterrupt:
        # Re-raise GraphInterrupt so LangGraph can pause the graph and handle it
        raise
    except Exception as e:
        # Only catch genuine errors (network failures, API errors, etc.)
        import traceback
        traceback.print_exc()
        emit(f"⚠️ MCP Agent failed: {e}", "warning")
        logger.error("[MCP] Agent failed: {}", e, exc_info=True)

    emit(f"✅ MCP Agent: completed with {len(mcp_results)} tool result(s)", "success")
    logger.info(f"[MCP] Completed with {len(mcp_results)} tool result(s)")
    return {"mcp_results": mcp_results}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — LLM Answer
# Synthesizes all available context into a final answer.
# ═══════════════════════════════════════════════════════════════════════════════

def llm_answer(state: AgentState) -> Dict:
    emit("🤖 LLM: synthesising answer from all retrieved context...", "step")
    logger.info("[LLM] Building final answer...")
    llm = get_llm(temperature=0.3)

    fast_path_text = state.get("fast_path_text")
    rag_context = state.get("rag_context")
    search_results = state.get("search_results", [])
    mcp_results = state.get("mcp_results", [])
    user_query = state.get("user_query", "")
    conversation_id = state.get("conversation_id", "")

    # Build context sections
    context_parts = []

    recent_history = get_recent_conversation_history(
        conversation_id,
        max_messages=12,
        max_characters=12000,
    )
    if recent_history:
        history_lines = [
            f"{item['role'].title()}: {item['content']}" for item in recent_history
        ]
        context_parts.append("## Recent Conversation\n" + "\n\n".join(history_lines) + "\n")

    # 1. Uploaded file (fast path)
    if fast_path_text:
        preview = fast_path_text[:3000]
        context_parts.append(f"## Uploaded Document Content\n{preview}\n")

    # 2. RAG context
    if rag_context:
        if rag_context.text_chunks:
            chunk_parts = []
            for c in rag_context.text_chunks[:6]:  # Show up to 6 chunks for better coverage
                doc_name = c.get("metadata", {}).get("filename") or c.get("metadata", {}).get("document_id", "Unknown Document")
                chunk_text = c.get("text", "")
                chunk_parts.append(f"[Source Document: {doc_name}]\n{chunk_text}")
            chunks_text = "\n---\n".join(chunk_parts)
            context_parts.append(f"## Knowledge Base Excerpts\n{chunks_text}\n")

        if rag_context.graph_context:
            graph_lines = [
                f"- {g['entity']} --[{g['relation']}]--> {g['neighbor']}"
                for g in rag_context.graph_context[:15]
            ]
            context_parts.append(f"## Knowledge Graph Context\n" + "\n".join(graph_lines) + "\n")

    # 3. Web search results
    if search_results:
        search_lines = []
        for r in search_results[:6]:
            search_lines.append(f"- [{r.source.upper()}] {r.title}\n  {r.url}\n  {r.snippet}")
        context_parts.append(f"## Web Search Results\n" + "\n\n".join(search_lines) + "\n")

    # 4. MCP tool outputs
    if mcp_results:
        mcp_lines = [
            f"- Tool: {r.get('tool', 'unknown')}\n  Output: {r.get('output', r.get('error', ''))}"
            for r in mcp_results
        ]
        context_parts.append(f"## External Tool Results\n" + "\n\n".join(mcp_lines) + "\n")

    # Build the full prompt
    context_block = "\n".join(context_parts) if context_parts else "No additional context available."
    full_prompt = f"""
{SYSTEM_PROMPT}

---

{context_block}

---

User Question: {user_query}

Please give a thorough and well-structured answer based on all the context above.
"""

    response = llm.invoke([HumanMessage(content=full_prompt)])
    answer = response.content

    try:
        record_conversation_turn(
            conversation_id=conversation_id,
            turn_id=state.get("turn_id", ""),
            user_query=user_query,
            final_answer=answer,
            route=state.get("route", "direct"),
            uploaded_file=state.get("uploaded_file_path"),
        )
    except Exception as exc:
        logger.error(f"[History] Failed to persist conversation turn: {exc}")

    emit(f"✅ Answer ready ({len(answer)} characters)", "success")
    logger.info(f"[LLM] Answer generated ({len(answer)} chars)")
    return {
        "final_answer": answer,
        "messages": [
            Message(role="user", content=user_query),
            Message(role="assistant", content=answer),
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — HITL Checkpoint  ← interrupt() pauses graph here
# Human reviews the answer and decides whether to save to memory.
# ═══════════════════════════════════════════════════════════════════════════════

def hitl_checkpoint(state: AgentState) -> Dict:
    """
    Pauses the graph and asks the human:
      "Do you want to save this conversation to long-term memory?"
    
    The UI must resume the graph by passing:
      {"approved": True/False, "notes": "optional annotation"}
    """
    final_answer = state.get("final_answer", "")
    conversation_id = state.get("conversation_id")
    emit("⏸️ Pausing for your review — save this conversation to memory?", "step")
    logger.info("[HITL] Interrupting for human review...")

    human_response = interrupt({
        "question": "Save this conversation to long-term memory (Pinecone + Neo4j)?",
        "answer_preview": (final_answer or "")[:500],
        "conversation_id": conversation_id,
        "instructions": "Reply with {'approved': true/false, 'notes': 'optional note'}"
    })

    # Parse response (UI sends back a dict)
    if isinstance(human_response, dict):
        decision = HitlDecision(
            approved=human_response.get("approved", False),
            notes=human_response.get("notes"),
        )
    else:
        decision = HitlDecision(approved=False)

    logger.info(f"[HITL] Human decision: approved={decision.approved}, notes={decision.notes}")
    return {
        "hitl_decision": decision,
        "save_requested": decision.approved,
    }


def hitl_route(state: AgentState) -> Literal["save_conversation", "__end__"]:
    """After HITL: route to save if approved, else end."""
    return "save_conversation" if state.get("save_requested") else END


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 — Save Conversation
# Persists the full turn to SQLite.
# ═══════════════════════════════════════════════════════════════════════════════

def save_conversation(state: AgentState) -> Dict:
    emit("💾 Saving conversation to SQLite...", "step")
    conversation_id = state.get("conversation_id")
    user_query = state.get("user_query")
    final_answer = state.get("final_answer")
    route = state.get("route")
    uploaded_file_path = state.get("uploaded_file_path")
    hitl_decision = state.get("hitl_decision")
    search_results = state.get("search_results", [])

    logger.info(f"[Save] Persisting conversation {conversation_id} to SQLite...")

    try:
        # 1. Upsert the main conversation record in SQLite
        upsert_conversation(
            conversation_id=conversation_id,
            user_query=user_query,
            final_answer=final_answer,
            route=route,
            uploaded_file=uploaded_file_path,
            hitl_approved=True,
            hitl_notes=None,
        )

        # Save web search results to SQLite
        if search_results:
            save_search_results(
                conversation_id,
                [{"source": r.source, "title": r.title, "url": r.url, "snippet": r.snippet}
                 for r in search_results]
            )

        emit(f"✅ Conversation saved to SQLite", "success")
        logger.info(f"[Save] ✅ Conversation {conversation_id} saved successfully.")
    except Exception as e:
        emit(f"⚠️ Save failed: {e}", "warning")
        logger.error(f"[Save] Failed to save conversation: {e}")

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register all nodes ───────────────────────────────────────────────────
    graph.add_node("router",          router)
    graph.add_node("tavily_worker",   tavily_worker)
    graph.add_node("serpapi_worker",  serpapi_worker)
    graph.add_node("aggregate_search", aggregate_search)
    graph.add_node("rag_retrieve",    rag_retrieve)
    graph.add_node("mcp_agent",       mcp_agent)
    graph.add_node("llm_answer",      llm_answer)
    graph.add_node("hitl_checkpoint", hitl_checkpoint)
    graph.add_node("save_conversation", save_conversation)

    # ── Edges ────────────────────────────────────────────────────────────────

    # Entry point
    graph.add_edge(START, "router")

    # Router → conditional edge
    graph.add_conditional_edges(
        "router",
        route_decision,
        # ["tavily_worker", "serpapi_worker", "rag_retrieve", "mcp_agent", "llm_answer"],
    )

    # Fan-in: both workers → aggregate → LLM
    graph.add_edge("tavily_worker",    "aggregate_search")
    graph.add_edge("serpapi_worker",   "aggregate_search")
    graph.add_edge("aggregate_search", "llm_answer")

    # RAG and MCP → LLM
    graph.add_edge("rag_retrieve", "llm_answer")
    graph.add_edge("mcp_agent",    "llm_answer")

    # LLM → Save conversation directly (no HITL / long-term memory checkpoint)
    graph.add_edge("llm_answer", "save_conversation")
    graph.add_edge("save_conversation", END)

    return graph


# ─── Compile the graph (with interrupt at HITL) ───────────────────────────────

if "_COMPILED_AGENT" not in globals():
    _COMPILED_AGENT = None


def compile_agent():
    """Returns the compiled LangGraph agent ready to invoke."""
    global _COMPILED_AGENT
    if _COMPILED_AGENT is not None:
        return _COMPILED_AGENT

    from langgraph.checkpoint.memory import MemorySaver
    graph = build_graph()
    memory = MemorySaver()
    _COMPILED_AGENT = graph.compile(
        checkpointer=memory,
    )
    return _COMPILED_AGENT


# ─── Convenience runner ───────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    conversation_id: str | None = None,
    uploaded_file_path: str | None = None,
) -> Dict:
    """
    Single-turn runner for the agent.
    Returns the final state dict.

    Usage:
        result = run_agent("What is the latest news about AI?")
        print(result["final_answer"])
    """
    import uuid
    conv_id = conversation_id or str(uuid.uuid4())

    # Pre-check guardrails and fast-path greetings
    pre_result = pre_check_query(user_query, conv_id, uploaded_file_path)
    if pre_result:
        return pre_result

    from src.agents.state import create_initial_state
    agent = compile_agent()

    initial_state = create_initial_state(
        user_query=user_query,
        conversation_id=conv_id,
        uploaded_file_path=uploaded_file_path,
    )
    config = {"configurable": {"thread_id": conv_id}}

    # Run the agent to completion (no HITL interrupt)
    result = agent.invoke(initial_state, config=config)
    logger.info(f"[Agent] Completed run for conversation {conv_id}")
    return result


def resume_agent(conversation_id: str, hitl_response: Dict) -> Dict:
    """
    Resumes the agent after HITL interrupt with the human's decision.

    Usage:
        resume_agent(conv_id, {"approved": True, "notes": "Save this!"})
    """
    agent = compile_agent()
    config = {"configurable": {"thread_id": conversation_id}}
    result = agent.invoke(Command(resume=hitl_response), config=config)
    return result


def pre_check_query(
    user_query: str,
    conversation_id: str,
    uploaded_file_path: str | None = None,
) -> Dict | None:
    """
    Checks the user query against fast-path greetings and NeMo input guardrails.
    If matched or blocked, persists the interaction to SQLite and returns the response immediately.
    Otherwise, returns None.
    """
    from src.utils.guardrails import check_fast_path_greeting, check_input_guardrails
    from src.tools.conversation_store import record_conversation_turn
    import uuid

    # 1. Greetings fast path
    greeting_response = check_fast_path_greeting(user_query)
    if greeting_response:
        try:
            record_conversation_turn(
                conversation_id=conversation_id,
                turn_id=str(uuid.uuid4()),
                user_query=user_query,
                final_answer=greeting_response,
                route="direct",
                uploaded_file=uploaded_file_path,
            )
        except Exception as exc:
            logger.error(f"[History] Failed to persist fast-path turn: {exc}")
        return {
            "final_answer": greeting_response,
            "route": "direct",
            "messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": greeting_response},
            ]
        }

    # 2. Input Guardrails
    is_blocked, refusal_message = check_input_guardrails(user_query)
    if is_blocked and refusal_message:
        try:
            record_conversation_turn(
                conversation_id=conversation_id,
                turn_id=str(uuid.uuid4()),
                user_query=user_query,
                final_answer=refusal_message,
                route="direct",
                uploaded_file=uploaded_file_path,
            )
        except Exception as exc:
            logger.error(f"[History] Failed to persist guardrail turn: {exc}")
        return {
            "final_answer": refusal_message,
            "route": "direct",
            "messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": refusal_message},
            ]
        }

    return None

