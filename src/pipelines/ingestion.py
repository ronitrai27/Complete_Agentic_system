import os
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from loguru import logger
from src.config import settings
from src.utils.parser import parse_file
from src.utils.vector_store import chunk_text, index_chunks
from src.utils.entity_extractor import extract_knowledge_graph_elements
from src.utils.graph_store import upsert_entities_and_relations
from src.utils.keyword_search import add_to_bm25_index

# Preserve the dictionary reference across hot-reloads so Streamlit doesn't wipe progress updates
if "src.pipelines.ingestion" in sys.modules and hasattr(sys.modules["src.pipelines.ingestion"], "INGESTION_PROGRESS"):
    INGESTION_PROGRESS = sys.modules["src.pipelines.ingestion"].INGESTION_PROGRESS
else:
    INGESTION_PROGRESS = {}

_PROGRESS_LOCK = threading.Lock()
_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "ingestion_registry.db",
)

def _update_progress(conversation_id: str, document_id: str, percent: int, status: str, details: str = "", filename: str = ""):
    if conversation_id and document_id:
        with _PROGRESS_LOCK:
            if conversation_id not in INGESTION_PROGRESS:
                INGESTION_PROGRESS[conversation_id] = {}
            INGESTION_PROGRESS[conversation_id][document_id] = {
                "percent": percent,
                "status": status,
                "details": details,
                "filename": filename or os.path.basename(document_id),
            }


def _registry_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    connection = sqlite3.connect(_REGISTRY_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestions (
            document_id TEXT PRIMARY KEY,
            checksum TEXT,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 1,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        )
        """
    )
    return connection


def _claim_ingestion(document_id: str, checksum: str, filename: str) -> str:
    """Atomically claim a document, or report an existing ingestion."""
    now = datetime.now(timezone.utc).isoformat()
    connection = _registry_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT status, started_at FROM ingestions WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row and row[0] == "completed":
            connection.commit()
            return "completed"
        if row and row[0] == "running":
            try:
                started_at = datetime.fromisoformat(row[1])
                still_active = datetime.now(timezone.utc) - started_at < timedelta(hours=1)
            except (TypeError, ValueError):
                still_active = False
            if still_active:
                connection.commit()
                return "running"

        connection.execute(
            """
            INSERT INTO ingestions
                (document_id, checksum, filename, status, attempts, started_at, completed_at, error)
            VALUES (?, ?, ?, 'running', 1, ?, NULL, NULL)
            ON CONFLICT(document_id) DO UPDATE SET
                status = 'running',
                attempts = ingestions.attempts + 1,
                started_at = excluded.started_at,
                completed_at = NULL,
                error = NULL
            """,
            (document_id, checksum, filename, now),
        )
        connection.commit()
        return "claimed"
    finally:
        connection.close()


def _finish_ingestion(document_id: str, status: str, error: str | None = None) -> None:
    connection = _registry_connection()
    try:
        connection.execute(
            """
            UPDATE ingestions
            SET status = ?, completed_at = ?, error = ?
            WHERE document_id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(), error, document_id),
        )
        connection.commit()
    finally:
        connection.close()

def ingest_document_pipeline(
    file_path: str,
    document_id: str = None,
    conversation_id: str = None,
    checksum: str = "",
    original_filename: str = "",
) -> dict:
    """
    Background (slow path) pipeline:
    1. Parse file using LlamaParse
    2. Split into chunks using SentenceSplitter
    3. Generate embeddings & upload to Pinecone
    4. Extract entities & relations using spaCy
    5. Load entities & relations into Neo4j
    6. Index chunks into BM25
    """
    if not document_id:
        document_id = os.path.basename(file_path)

    filename = original_filename or os.path.basename(file_path)
    claim = _claim_ingestion(document_id, checksum, filename)
    if claim == "completed":
        details = "This exact file was already indexed. Reused the existing knowledge."
        _update_progress(conversation_id, document_id, 100, "completed", details, filename)
        return {"document_id": document_id, "status": "already_indexed"}
    if claim == "running":
        details = "This exact file is already being indexed."
        _update_progress(conversation_id, document_id, 5, "Ingestion already running", details, filename)
        return {"document_id": document_id, "status": "already_running"}

    logger.info(f"Starting document ingestion pipeline for: {document_id}")
    _update_progress(conversation_id, document_id, 0, "Starting ingestion...", "", filename)

    try:
        # 1. Parsing
        _update_progress(conversation_id, document_id, 10, "Parsing document using LlamaParse...", "Running LlamaParse parser on document...", filename)
        logger.info("Step 1: Parsing document using LlamaParse...")
        text_content = parse_file(file_path)
        if not text_content:
            _update_progress(conversation_id, document_id, 100, "failed", "No text extracted from file", filename)
            raise ValueError(f"No text extracted from file: {file_path}")
            
        # 2. Chunking
        _update_progress(conversation_id, document_id, 35, "Splitting text into chunks...", "Splitting parsed markdown text using SentenceSplitter...", filename)
        logger.info("Step 2: Splitting text into chunks...")
        chunks = chunk_text(text_content)
        chunks_count = len(chunks)
        _update_progress(conversation_id, document_id, 45, "Chunks generated", f"Successfully split document into {chunks_count} chunks.", filename)
        
        # 3. Vector Database Indexing (Pinecone)
        _update_progress(conversation_id, document_id, 50, "Generating embeddings & indexing in Pinecone...", f"Generating OpenAI embeddings and indexing {chunks_count} chunks in Pinecone...", filename)
        logger.info("Step 3: Indexing chunks in Pinecone...")
        metadata_base = {
            "filename": filename,
            "checksum": checksum,
            "conversation_id": conversation_id or "",
            "type": "document",
        }
        index_chunks(document_id, chunks, metadata_base)
        _update_progress(conversation_id, document_id, 65, "Vector store updated", f"Successfully indexed {chunks_count} chunks in Pinecone vector store.", filename)
        
        # 4. Entity Extraction
        _update_progress(conversation_id, document_id, 70, "Extracting entities & relationships (spaCy)...", "Extracting named entities and relationships from text using spaCy NER...", filename)
        logger.info("Step 4: Extracting entities & relationships...")
        kg_elements = extract_knowledge_graph_elements(text_content)
        entities = kg_elements.get("entities", [])
        relations = kg_elements.get("relations", [])
        
        # Format a preview of extracted entities and relationships
        entity_names = [e["name"] for e in entities]
        entity_preview = ", ".join(entity_names[:8]) + ("..." if len(entity_names) > 8 else "")
        
        relation_parts = []
        for r in relations[:3]:
            relation_parts.append(f"{r['source']}->{r['target']}")
        relation_preview = f"Found {len(relations)} unique relationship(s) (e.g., {', '.join(relation_parts)})" if relations else "No relationships found"
        
        details_extraction = f"Extracted {len(entities)} entities: {entity_preview}\n\n{relation_preview}"
        _update_progress(conversation_id, document_id, 80, "Entity extraction complete", details_extraction, filename)
        
        # 5. Graph Database Ingestion (Neo4j)
        _update_progress(conversation_id, document_id, 85, "Upserting graph into Neo4j database...", f"Loading {len(entities)} entities and {len(relations)} relations into Neo4j graph db...", filename)
        logger.info("Step 5: Loading entities and relations into Neo4j...")
        if entities or relations:
            upsert_entities_and_relations(entities, relations)
        else:
            logger.warning("No entities or relations found to load into Neo4j.")
        _update_progress(conversation_id, document_id, 90, "Neo4j graph database updated", f"Loaded {len(entities)} entities and {len(relations)} relations into Neo4j graph store.", filename)
            
        # 6. Keyword Index Ingestion (BM25)
        _update_progress(conversation_id, document_id, 95, "Adding chunks to BM25 index...", f"Adding {chunks_count} chunks to disk-backed BM25 keyword index...", filename)
        logger.info("Step 6: Adding chunks to BM25 index...")
        add_to_bm25_index(document_id, chunks, metadata_base)
        
        logger.info(f"Ingestion pipeline completed successfully for document: {document_id}")
        
        final_details = (
            f"✅ **Ingestion Summary:**\n"
            f"- **Chunks Ingested:** {chunks_count}\n"
            f"- **Entities Extracted:** {len(entities)}\n"
            f"- **Unique Relationships Extracted:** {len(relations)}\n"
            f"- **Graph stored:** Neo4j\n"
            f"- **Lexical index:** BM25 updated\n"
            f"\n**Entities Preview:**\n{entity_preview}"
        )
        _update_progress(conversation_id, document_id, 100, "completed", final_details, filename)
        _finish_ingestion(document_id, "completed")
        
        return {
            "document_id": document_id,
            "chunks_count": len(chunks),
            "entities_count": len(entities),
            "relations_count": len(relations)
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {document_id}: {e}")
        _update_progress(conversation_id, document_id, 100, "failed", str(e), filename)
        _finish_ingestion(document_id, "failed", str(e)[:2000])
        raise
