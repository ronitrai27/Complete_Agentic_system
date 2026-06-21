import os
import sys
import time
import uuid
from dotenv import load_dotenv

# Try to apply nest_asyncio to avoid "asyncio.run() cannot be called from a running event loop"
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

# Load environment variables first
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"), override=True)
sys.path.insert(0, ROOT)

from src.utils.parser import parse_file
from src.utils.vector_store import chunk_text, index_chunks, query_vector_store
from src.utils.entity_extractor import extract_knowledge_graph_elements, extract_entities
from src.utils.graph_store import upsert_entities_and_relations, get_neighbors
from src.utils.keyword_search import add_to_bm25_index, query_bm25
from src.agents.rag_agent import compile_agent
from src.agents.state import create_initial_state

def run_ingestion(file_path: str):
    print("\n" + "="*80)
    print(f"[START] STARTING INGESTION PIPELINE FOR: {file_path}")
    print("="*80)
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Error: File not found at {file_path}")
        sys.exit(1)
        
    document_id = os.path.basename(file_path)
    metadata_base = {"filename": document_id}
    
    total_start = time.time()
    
    # Step 1: Parsing
    print("\n[Step 1/6] Parsing document using LlamaParse...")
    start = time.time()
    text_content = parse_file(file_path)
    parsing_time = time.time() - start
    print(f"[OK] Parsing Complete! Extracted {len(text_content)} characters.")
    print(f"[TIME] Time taken: {parsing_time:.2f} seconds")
    
    # Step 2: Chunking
    print("\n[Step 2/6] Chunking text using SentenceSplitter...")
    start = time.time()
    chunks = chunk_text(text_content)
    chunking_time = time.time() - start
    print(f"[OK] Chunking Complete! Generated {len(chunks)} chunks.")
    print(f"[TIME] Time taken: {chunking_time:.2f} seconds")
    
    # Step 3: Pinecone Indexing
    print("\n[Step 3/6] Indexing chunks into Pinecone vector store...")
    start = time.time()
    index_chunks(document_id, chunks, metadata_base)
    pinecone_time = time.time() - start
    print(f"[OK] Pinecone Indexing Complete!")
    print(f"[TIME] Time taken: {pinecone_time:.2f} seconds")
    
    # Step 4: spaCy Entity Extraction
    print("\n[Step 4/6] Extracting entities & relationships using spaCy NER...")
    start = time.time()
    kg_elements = extract_knowledge_graph_elements(text_content)
    entities = kg_elements.get("entities", [])
    relations = kg_elements.get("relations", [])
    spacy_time = time.time() - start
    print(f"[OK] Extraction Complete! Found {len(entities)} entities and {len(relations)} relationships.")
    print(f"[TIME] Time taken: {spacy_time:.2f} seconds")
    
    # Step 5: Neo4j Ingestion
    print("\n[Step 5/6] Upserting entities and relations into Neo4j graph database...")
    start = time.time()
    if entities or relations:
        upsert_entities_and_relations(entities, relations)
    else:
        print("[WARN] No entities or relations to upsert.")
    neo4j_time = time.time() - start
    print(f"[OK] Neo4j Ingestion Complete!")
    print(f"[TIME] Time taken: {neo4j_time:.2f} seconds")
    
    # Step 6: BM25 Indexing
    print("\n[Step 6/6] Indexing chunks in BM25 keyword index...")
    start = time.time()
    add_to_bm25_index(document_id, chunks, metadata_base)
    bm25_time = time.time() - start
    print(f"[OK] BM25 Indexing Complete!")
    print(f"[TIME] Time taken: {bm25_time:.2f} seconds")
    
    total_time = time.time() - total_start
    print("\n" + "="*80)
    print(f"[SUCCESS] DOCUMENT INGESTION PIPELINE FULLY COMPLETED IN {total_time:.2f} seconds!")
    print("="*80)

def chat_loop():
    print("\n" + "="*80)
    print("[CHAT] ENTERING AGENT CHAT INTERACTIVE MODE")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*80)
    
    agent = compile_agent()
    conversation_id = str(uuid.uuid4())
    
    while True:
        try:
            print("\n" + "-"*50)
            user_query = input("User: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
                
            # Time the query execution
            query_start = time.time()
            
            # Pre-check guardrails and fast-path greetings
            from src.agents.rag_agent import pre_check_query
            pre_result = pre_check_query(user_query, conversation_id)
            if pre_result:
                result = pre_result
            else:
                # Prepare state and configuration
                initial_state = create_initial_state(
                    user_query=user_query,
                    conversation_id=conversation_id
                )
                config = {"configurable": {"thread_id": conversation_id}}
                
                # Run the agent turn until HITL checkpoint
                result = agent.invoke(initial_state, config=config)
            query_time = time.time() - query_start
            
            # Print Route Decision
            route = result.get("route", "unknown")
            print(f"\n[ROUTE] Router Route Decided: {route.upper()}")
            print(f"[TIME] Turn Execution Time: {query_time:.2f} seconds")
            
            # If the route was RAG, print retrieval diagnostics
            if route == "rag":
                print("\n[DIAG] --- RAG RETRIEVAL DIAGNOSTICS ---")
                
                # 1. Pinecone Vector Results
                print("\n1. [VECTOR] Pinecone Vector Search (Semantic) Results:")
                try:
                    vector_results = query_vector_store(user_query, top_k=4)
                    if vector_results:
                        for i, r in enumerate(vector_results):
                            print(f"   [{i+1}] Score: {r['score']:.4f} | Content: {r['text'][:180].replace('\n', ' ')}...")
                    else:
                        print("   No vector results returned.")
                except Exception as e:
                    print(f"   [ERROR] Failed to query Pinecone: {e}")
                    
                # 2. BM25 Results
                print("\n2. [BM25] BM25 Keyword Search Results:")
                try:
                    bm25_results = query_bm25(user_query, top_k=4)
                    if bm25_results:
                        for i, r in enumerate(bm25_results):
                            print(f"   [{i+1}] Content: {r['text'][:180].replace('\n', ' ')}...")
                    else:
                        print("   No BM25 results returned.")
                except Exception as e:
                    print(f"   [ERROR] Failed to query BM25: {e}")
                    
                # 3. Graph Results
                print("\n3. [GRAPH] Neo4j Graph Neighbor Results:")
                try:
                    query_ents = extract_entities(user_query)
                    if query_ents:
                        print(f"   Extracted query entities: {[e['name'] for e in query_ents]}")
                        found_any_graph = False
                        for ent in query_ents:
                            name = ent["name"]
                            neighbors = get_neighbors(name, limit=5)
                            if neighbors:
                                found_any_graph = True
                                print(f"   Connections for '{name}':")
                                for neighbor in neighbors:
                                    print(f"     - ({name})--[{neighbor.get('rel_type')}]-->({neighbor.get('neighbor_name')})")
                        if not found_any_graph:
                            print("   No graph neighbors found for the extracted query entities.")
                    else:
                        print("   No entities extracted from query to look up in Neo4j.")
                except Exception as e:
                    print(f"   [ERROR] Failed to query Neo4j: {e}")
                print("------------------------------------\n")
            
            # Print Final LLM Answer
            final_answer = result.get("final_answer", "No answer returned.")
            print(f"\n[AGENT] Agent:\n{final_answer}")
            
            # Auto-approve HITL checkpoint if pending
            # (In interactive mode, we auto-approve so that subsequent queries can run smoothly in the same thread)
            interrupts = result.get("__interrupt__", [])
            if interrupts:
                # Resume the thread with approval
                from langgraph.types import Command
                agent.invoke(Command(resume={"approved": True, "notes": "CLI auto-approve"}), config=config)
                print("\n[SAVE] [CLI Auto-saved conversation turn to long-term memory]")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Error processing turn: {e}")

if __name__ == "__main__":
    # Ingestion path default
    default_path = os.path.join(ROOT, "data", "enterprise_knowledge_base.txt")
    
    # Process ingestion
    run_ingestion(default_path)
    
    # Start chat loop
    chat_loop()
