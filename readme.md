# AI Flow — Setup, Architecture, and Operations Guide

![NVIDIA NeMo Guardrails](https://img.shields.io/badge/NVIDIA%20NeMo--Guardrails-76B900?style=flat-square&logo=nvidia&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-orange?style=flat-square&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-blueviolet?style=flat-square)
![LlamaIndex / LlamaParse](https://img.shields.io/badge/LlamaIndex-brightgreen?style=flat-square)
![Tavily](https://img.shields.io/badge/Tavily-0052FF?style=flat-square)
![Arcade MCP](https://img.shields.io/badge/Arcade%20MCP-4B0082?style=flat-square)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017AEC?style=flat-square&logo=apacheairflow&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat-square)

This project is an agentic RAG application with:

- A Streamlit chat interface
- A LangGraph agent and router
- Tavily and SerpAPI web search tools
- Arcade MCP tools for Gmail, Google Docs, Notion, and Outlook
- LlamaCloud/LlamaParse document parsing
- Pinecone semantic retrieval
- BM25 lexical retrieval
- Neo4j knowledge-graph storage and visualization
- SQLite conversation history
- NeMo Guardrails input validation and pre-check greetings bypass
- Human approval before saving a turn to long-term graph/vector memory

## 1. Requirements

- Python `>=3.12,<3.14`
- Poetry
- API credentials configured in `.env`
- Pinecone and Neo4j instances
- Windows users: Visual Studio C++ Build Tools (specifically the **Desktop development with C++** workload) is required to compile NeMo Guardrails dependencies (e.g., `hnswlib`).
- Windows users: run the main application normally, but use WSL2 or Docker for Airflow

Install dependencies:

```powershell
poetry install
```

Verify the environment:

```powershell
poetry run python --version
poetry run pytest tests
```

## 2. Environment Configuration

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=
TAVILY_API_KEY=
SERPAPI_API_KEY=
LLAMA_CLOUD_API_KEY=
ARCADE_API_KEY=

PINECONE_API_KEY=
PINECONE_INDEX_NAME=agentic-system

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
```

Never commit `.env`. The application no longer displays any part of the OpenAI key in the UI.

Initialize Pinecone and the Neo4j entity constraint:

```powershell
poetry run python scripts/init_databases.py
```

## 3. Running the Application

### Streamlit application

From the project root:

```powershell
poetry run streamlit run ui/app.py
```

Open:

```text
http://localhost:8501
```

The UI supports:

- Chatting with the agent
- Uploading supported documents
- Watching background-ingestion progress
- Approving or rejecting long-term memory
- Completing MCP OAuth authorization
- Opening the Neo4j relationship explorer

### CLI chat

```powershell
poetry run python scripts/chat_cli.py
```

The CLI creates a conversation ID, runs the same LangGraph agent, prints progress events, handles authorization interrupts, and asks whether a turn should be saved to long-term memory.

### Run tests

```powershell
poetry run pytest tests -q
```

Use `pytest tests`, rather than unrestricted `pytest`, because files named `test_*.py` under `scripts/` and `src/utils/` are live connectivity utilities and may call external services during collection.

## 4. Updated Project Structure

```text
ai_flow/
├── dags/
│   └── document_ingestion_dag.py
├── data/
│   ├── uploads/                 # Safe content-addressed uploads; gitignored
│   ├── parsed/                  # Cached LlamaCloud markdown; gitignored
│   ├── conversations.db         # Chat history; gitignored
│   └── bm25_index.pkl           # Local lexical index; gitignored
├── scripts/
│   ├── chat_cli.py
│   ├── clean_neo4j.py
│   ├── init_databases.py
│   ├── reingest_docs.py
│   ├── test_cli.py
│   ├── test_mcp_gmail.py
│   └── test_mcp_math.py
├── src/
│   ├── agents/
│   │   ├── rag_agent.py
│   │   └── state.py
│   ├── guardrails/
│   │   ├── config.yml           # NeMo Guardrails configuration
│   │   └── disallowed.co       # Colang disallowed intents and flows
│   ├── pipelines/
│   │   └── ingestion.py
│   ├── tools/
│   │   ├── conversation_store.py
│   │   ├── mcp.py
│   │   └── search.py
│   └── utils/
│       ├── entity_extractor.py
│       ├── graph_store.py
│       ├── guardrails.py        # Pre-check greetings & NeMo Guardrails helper functions
│       ├── hybrid_search.py
│       ├── keyword_search.py
│       ├── parser.py
│       ├── uploads.py           # New safe-upload service
│       └── vector_store.py
├── tests/
│   ├── test_agents.py
│   ├── test_rag_utilities.py
│   └── test_uploads_and_history.py
└── ui/
    ├── app.py
    └── pages/
        └── relationships.py     # New Neo4j graph explorer
```

## 5. Recent Updates

### 5.1 Neo4j relationship explorer

The chat sidebar now contains a **View relationships** button.

It opens `ui/pages/relationships.py`, which:

- Reads connected entities and relationships from Neo4j
- Displays an interactive directed PyVis graph
- Colors nodes by entity type
- Supports entity/relationship search
- Supports entity-type filtering
- Limits graph size between 100 and 5,000 relationships
- Shows node, relationship, and relationship-type metrics
- Provides a relationship table for easier inspection

The page is read-only. It does not modify or delete Neo4j data.

Relevant implementation:

- `ui/pages/relationships.py`
- `src/utils/graph_store.py`
  - `get_graph_snapshot()`
  - `get_entity_labels()`

### 5.2 Safer uploads

Uploads now pass through `src/utils/uploads.py`.

Safety improvements:

- Filename path traversal is removed
- Filenames are sanitized
- File extensions are allowlisted
- Empty files are rejected
- Upload size is limited to 25 MB
- PDF, DOCX, PNG, and JPEG signatures are checked
- Files are stored atomically using a temporary `.part` file
- Local paths use generated IDs rather than user-controlled filenames
- Upload identity uses a SHA-256 checksum

Supported extensions:

```text
.pdf .docx .txt .png .jpg .jpeg .md
```

Stored uploads use this structure:

```text
data/uploads/<conversation_id>/doc_<sha256>.<extension>
```

### 5.3 Repeatable ingestion

Each accepted upload runs the ingestion pipeline. The same file checksum still
produces the same document ID, so writes remain repeatable:

- Pinecone vector IDs are deterministic and are upserted
- BM25 replaces chunks for the existing document ID
- Neo4j uses `MERGE` for entities and relationships
- A missing Pinecone index is recreated automatically

This keeps the mini-project easy to understand: no separate ingestion registry
or document-marker nodes are required.

LlamaCloud parsing is also cached by file checksum:

```text
data/parsed/<sha256>.md
```

The chat fast path and background ingestion can therefore reuse the same parsed result instead of paying for and waiting on two identical LlamaCloud parses.

### 5.4 Conversation history

Each generated turn now has its own `turn_id`.

Every successful answer is atomically written to SQLite as:

```text
conversation
└── turn
    ├── user message
    └── assistant message
```

Improvements:

- Each turn is persisted even before optional long-term-memory approval
- Replaying the same turn does not duplicate messages
- Follow-up questions receive recent conversation context
- The router receives a smaller recent-history window
- The answer model receives a larger bounded history window
- Prompt history has a character budget to avoid uncontrolled growth
- Approved conversation memories use per-turn Pinecone IDs
- Conversation vectors can be retrieved only for their own conversation ID

Human approval still controls whether a turn is additionally extracted into Neo4j and indexed as long-term Pinecone conversation memory.

### 5.5 NeMo input guardrails

We implemented **NVIDIA NeMo Guardrails** (`v0.22.0`) to check and sanitise incoming user queries before they reach the main LangGraph agent.

#### Installation & Prerequisites
NeMo Guardrails is specified in [pyproject.toml](file:///r:/python/ai_flow/pyproject.toml) and is automatically installed when you run `poetry install`. However, on Windows, installing it has specific compilation requirements:

1. **C++ Build Tools Requirement**:
   NeMo Guardrails installs underlying packages like `hnswlib` (a C++ vector database binding) and others that compile native C++ extensions during installation.
   * **Error Symptoms**: Without C++ compilers, installation will fail with errors such as `error: Microsoft Visual C++ 14.0 or greater is required.` or compilation errors for `hnswlib`.
   * **Solution**: You must install the **Build Tools for Visual Studio** (Visual Studio 2022 Build Tools or newer).
     * Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
     * During the installation, select the **Desktop development with C++** workload.
     * Ensure the **MSVC v143 - VS 2022 C++ x64/x86 build tools** and **Windows 11 SDK** (or Windows 10 SDK) components are checked.
     
2. **Python Version**: NeMo Guardrails has strict dependencies and requires Python `>=3.12,<3.14`.
3. **Active Loops Setup**: It requires `nest_asyncio` to run within Streamlit or other environments with active asyncio loops. This is automatically handled dynamically in [src/utils/guardrails.py](file:///r:/python/ai_flow/src/utils/guardrails.py).

#### How It Works
NeMo Guardrails coordinates validation flows using a combination of configuration files, semantic embedding search, and Colang script logic:

1. **Configuration ([config.yml](file:///r:/python/ai_flow/src/guardrails/config.yml))**:
   - Defines the engines used: `gpt-4o-mini` for the LLM-based rails and `text-embedding-3-small` for generating sentence embeddings.
   - Restricts Dialog flows by setting `embeddings_only: true` with a `similarity_threshold` of `0.82`.
   - Specifies a fallback intent (`unhandled_user_intent`) if the query does not map to any defined intent.
2. **Colang Intent Mapping ([disallowed.co](file:///r:/python/ai_flow/src/guardrails/disallowed.co))**:
   - Maps standard user query variations to specific disallowed intents:
     - `user ask cooking recipe`: Intercepts cooking/food instructions.
     - `user ask developer mode`: Intercepts system prompts, developer simulations, system overrides, and jailbreak attempts.
     - `user ask python code`: Intercepts general coding requests.
     - `user ask violence`: Intercepts safety-critical queries.
   - Defines a standard refusal response:
     - `bot refuse request`: *"I cannot help you with that request. I am designed to assist only with technical documentation and tech-related web searches, not for cooking, coding, developer simulation, or harmful queries."*
   - Defines dialog **flows** connecting each disallowed intent to the refusal action (e.g., if a user asks a cooking recipe, the bot replies with the refusal response).
3. **Semantic Matching**: When a query is checked, NeMo Guardrails calculates its embedding and checks the similarity against the canonical examples in the `.co` file. If the similarity meets or exceeds `0.82`, it triggers the associated block flow.

#### How It Is Used in the Codebase
- **Pre-Check Hook**: Before the query initiates the LangGraph compiler and states in [src/agents/rag_agent.py](file:///r:/python/ai_flow/src/agents/rag_agent.py), `run_agent()` triggers `pre_check_query()`.
- **Fast-Path Greeting Filter**: Before making any API requests, [src/utils/guardrails.py](file:///r:/python/ai_flow/src/utils/guardrails.py) performs a regex check against `GREETINGS_MAP`. If a greeting is matched, the predefined reply is returned instantly. This prevents latency and avoids any LLM / OpenAI token costs.
- **Guardrails Invocation**: If it's a standard user query, `check_input_guardrails(query)` is called:
  - It temporarily injects `NEMO_API_KEY` (configured in [src/utils/guardrails.py](file:///r:/python/ai_flow/src/utils/guardrails.py)) as the `OPENAI_API_KEY` environment variable.
  - It retrieves the cached `LLMRails` instance from `get_rails_instance()`.
  - It calls `rails.generate()` with the user query.
  - It inspects the returned string. If the response contains any of the known refusal markers (such as `"I cannot help you with that request"` or `"I am designed to assist only with technical documentation"`), the helper marks the query as blocked and returns the refusal message.
  - The refusal message is saved to the SQLite conversation table using `record_conversation_turn()`, and the execution terminates immediately, short-circuiting the LangGraph router.
- **Fail-Safe Fallback**: If an exception or connection failure occurs during the NeMo check, the exception is caught, logged, and the query is allowed through to prevent disrupting the user flow.

#### Files Created for NeMo Guardrails
- [src/guardrails/config.yml](file:///r:/python/ai_flow/src/guardrails/config.yml): Sets up the LLM engines, embedding engines, and embedding similarity parameters.
- [src/guardrails/disallowed.co](file:///r:/python/ai_flow/src/guardrails/disallowed.co): Contains user intents (Colang definitions) for disallowed topics and maps them to flows triggering `bot refuse request`.
- [src/utils/guardrails.py](file:///r:/python/ai_flow/src/utils/guardrails.py): Houses the fast-path regex greetings mapping (`GREETINGS_MAP`), functions `check_fast_path_greeting()`, and `check_input_guardrails()` which initialize and execute the NeMo guardrails.
- [src/agents/rag_agent.py](file:///r:/python/ai_flow/src/agents/rag_agent.py): Calls the `pre_check_query()` hook at the entrance of `run_agent()` to execute the guardrail validation.

## 6. Application Data Flow

### Chat request

```text
User question
    ↓
Pre-check Fast-path Greetings (Local dictionary check)
    ├── Matched → Predefined response (Bypasses LLM/Guardrails completely)
    └── No Match
         ↓
NeMo Input Guardrails
    ├── Blocked → Predefined refusal response (Bypasses LangGraph)
    └── Allowed
         ↓
LangGraph router
    ├── search → Tavily + SerpAPI
    ├── rag    → Pinecone + BM25 + Neo4j
    ├── mcp    → Arcade tools
    └── direct → LLM
    ↓
Answer generation with recent conversation history
    ↓
Atomic SQLite turn persistence
    ↓
Human memory approval (optional)
    ├── approved → Neo4j + Pinecone long-term memory
    └── rejected → conversation history remains in SQLite only
```

### Upload request

```text
Upload bytes
    ↓
Size, extension, signature, and filename validation
    ↓
SHA-256 document identity
    ↓
Safe atomic local storage
    ↓
LlamaCloud parse or cached markdown
    ↓
SentenceSplitter chunks
    ├── OpenAI embeddings → Pinecone
    ├── spaCy entities/relations → Neo4j
    └── tokens → BM25
```

## 7. Script Reference

### `scripts/init_databases.py`

Creates or verifies:

- The Pinecone index
- The Neo4j unique constraint on `Entity.name`

```powershell
poetry run python scripts/init_databases.py
```

### `scripts/chat_cli.py`

Runs the main agent in an interactive terminal:

```powershell
poetry run python scripts/chat_cli.py
```

### `scripts/reingest_docs.py`

Reprocesses the project’s configured sample documents through the ingestion pipeline:

```powershell
poetry run python scripts/reingest_docs.py
```

Review the file paths in the script before running it.

### `scripts/clean_neo4j.py`

Destructively clears graph data. Inspect the script and confirm the target database before use:

```powershell
poetry run python scripts/clean_neo4j.py
```

Do not run this against a production database without a backup.

### `scripts/test_mcp_math.py`

Runs an Arcade tool that does not require OAuth:

```powershell
poetry run python scripts/test_mcp_math.py
```

### `scripts/test_mcp_gmail.py`

Tests Gmail authorization and lists Gmail threads:

```powershell
poetry run python scripts/test_mcp_gmail.py
```

This accesses the configured external account and should be treated as a live integration check, not a unit test.

### `scripts/test_cli.py`

Runs a broad ingestion and interactive-agent diagnostic flow:

```powershell
poetry run python scripts/test_cli.py
```

This can call LlamaCloud, OpenAI, Pinecone, and Neo4j and may incur API usage.

## 8. Apache Airflow: What It Does Here

Airflow is not the chat server and should not run each user question.

Use Airflow for durable, observable background workflows such as:

- Scheduled document ingestion
- Retrying failed ingestion
- Nightly re-indexing
- Rebuilding BM25
- Knowledge-graph maintenance
- Data-quality checks
- Cleaning old upload files
- Sending failure notifications

The Streamlit thread is useful for immediate background work during development. Airflow becomes valuable when ingestion must survive application restarts and provide retries, logs, schedules, ownership, and operational visibility.

## 9. Current Airflow Status

Airflow `3.1.0` is installed, but the current DAG requires repair before use:

```text
dags/document_ingestion_dag.py
```

Current issues:

1. It imports `ingest_documents_pipeline`, but the implemented function is `ingest_document_pipeline`.
2. `ingest_document_pipeline()` requires a file path and document context, but the DAG currently supplies no arguments.
3. The DAG uses an older `PythonOperator` import path.
4. A daily schedule is not enough to tell Airflow which uploaded document should be processed.

Do not expect this DAG to appear successfully until those points are addressed.

The best next implementation is an ingestion queue:

```text
Streamlit upload
    ↓
Create queued ingestion record
    ↓
Airflow scheduled DAG reads pending records
    ↓
One task processes each document
    ↓
Registry becomes completed or failed
```

This preserves the existing ingestion function while moving durability and retries into Airflow.

## 10. Practising Airflow Locally

Apache Airflow does not natively support Windows. Use WSL2 or Docker Desktop.

### Recommended beginner path: WSL2 standalone

Inside Ubuntu/WSL:

```bash
export AIRFLOW_HOME=~/airflow
airflow standalone
```

`airflow standalone` initializes the metadata database, creates an admin user, and starts the required local components.

Open:

```text
http://localhost:8080
```

Put or link this project’s `dags` folder into:

```text
$AIRFLOW_HOME/dags
```

Useful learning commands:

```bash
airflow version
airflow dags list
airflow dags list-import-errors
airflow dags show document_ingestion_pipeline
airflow tasks list document_ingestion_pipeline
airflow tasks test <dag_id> <task_id> 2026-06-18
airflow dags test <dag_id> 2026-06-18
airflow dags trigger <dag_id>
airflow dags pause <dag_id>
airflow dags unpause <dag_id>
```

Start with this practice loop:

1. Write a tiny DAG with one Python task.
2. Confirm it appears in the UI.
3. Run `airflow dags list-import-errors`.
4. Test the task with `airflow tasks test`.
5. Test the complete DAG with `airflow dags test`.
6. Trigger it manually in the UI.
7. Add retries and deliberately fail the first attempt.
8. Read task logs in the UI.
9. Add a schedule only after manual execution works.

### Docker Compose learning environment

Airflow’s official Docker Compose quick start uses multiple services including PostgreSQL, Redis, scheduler, DAG processor, API server, worker, and triggerer.

Typical commands:

```bash
docker compose up airflow-init
docker compose up -d
docker compose ps
docker compose logs -f airflow-scheduler
docker compose run airflow-cli airflow dags list
docker compose down
```

This is excellent for learning CeleryExecutor and distributed task execution, but the official quick-start Compose file is not a secure production deployment.

## 11. Production-Like Airflow Architecture

For a serious deployment:

```text
Git repository
    ↓
CI tests DAG imports and tasks
    ↓
Versioned Airflow image
    ↓
Kubernetes + official Airflow Helm chart
    ├── API server
    ├── Scheduler
    ├── DAG processor
    ├── Triggerer
    └── Workers
         ↓
PostgreSQL metadata database
Object storage for logs
Secret manager for credentials
Monitoring and alerts
```

Production recommendations:

- Use PostgreSQL, not SQLite, for Airflow metadata
- Build a pinned custom image containing project dependencies
- Store secrets in Airflow Connections or an external secret backend
- Do not hardcode API keys in DAG files
- Keep DAG files thin; call business logic from `src/`
- Make tasks idempotent
- Set explicit retries and retry delays
- Configure timeouts and concurrency limits
- Use remote task logging
- Add failure notifications
- Test DAG imports in CI
- Use separate development, staging, and production environments
- Avoid local files as the only handoff between distributed workers

For this project, uploaded files should eventually move from local `data/uploads` storage to shared object storage such as S3, GCS, or Azure Blob before Airflow workers process them.

## 12. Suggested Airflow Learning Project

Build this in four stages:

### Stage 1 — Health-check DAG

- Task 1: verify required environment settings
- Task 2: connect to Pinecone
- Task 3: connect to Neo4j

### Stage 2 — Single-document ingestion

- Supply one known file path through DAG parameters
- Call `ingest_document_pipeline()`
- Observe retries and logs

### Stage 3 — Queue-based ingestion

- Read pending records from an ingestion queue table
- Dynamically map one task per document
- Record completion or failure

### Stage 4 — Production simulation

- Run Airflow in Docker Compose
- Use PostgreSQL
- Run multiple workers
- Add remote/shared document storage
- Add alerts and dashboard monitoring

This progression teaches the real Airflow concepts without rewriting the agent or ingestion logic.

## 13. Tests Added for the Updates

`tests/test_uploads_and_history.py` verifies:

- Unsafe filenames are sanitized
- Identical uploads receive the same document ID
- Duplicate local storage is avoided
- Spoofed PDFs are rejected
- Conversation turns are ordered correctly
- Replaying a turn does not duplicate its messages

Run:

```powershell
poetry run pytest tests/test_uploads_and_history.py -q
```

Run the full unit-test folder:

```powershell
poetry run pytest tests -q
```

## 14. Useful Official Airflow References

- Airflow 3.1 Quick Start: https://airflow.apache.org/docs/apache-airflow/3.1.0/start.html
- Airflow 3.1 Docker Compose: https://airflow.apache.org/docs/apache-airflow/3.1.0/howto/docker-compose/index.html
- Airflow Best Practices: https://airflow.apache.org/docs/apache-airflow/3.1.0/best-practices.html
- Airflow CLI Reference: https://airflow.apache.org/docs/apache-airflow/3.1.0/cli-and-env-variables-ref.html
