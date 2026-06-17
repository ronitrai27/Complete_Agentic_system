import sys
import os
import re

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from src.utils.graph_store import run_write_query, run_read_query, upsert_entities_and_relations
from src.utils.entity_extractor import extract_knowledge_graph_elements, _clean_and_validate_node
from src.utils.parser import parse_file

def main():
    print("================================================================================")
    print("[CLEAN] Purging Neo4j Graph Database...")
    try:
        run_write_query("MATCH (n) DETACH DELETE n")
        print("[OK] Purge complete!")
    except Exception as e:
        print(f"[ERROR] Neo4j purge failed: {e}")
        return

    # 1. Ingest enterprise_knowledge_base.txt
    txt_path = "data/enterprise_knowledge_base.txt"
    print("\n================================================================================")
    print(f"[FILE] Processing: {txt_path}")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
            
        print("[SEARCH] Extracting entities and relations from TXT...")
        kg_txt = extract_knowledge_graph_elements(txt_content)
        txt_entities = kg_txt.get("entities", [])
        txt_relations = kg_txt.get("relations", [])
        
        print(f"Found {len(txt_entities)} entities and {len(txt_relations)} relationships in TXT.")
        print("\n--- Extracted Entities (TXT) ---")
        for ent in sorted(txt_entities, key=lambda x: x["name"]):
            safe_name = ent['name'].encode('ascii', 'replace').decode()
            print(f"  [{ent['label']}] {safe_name}")
            
        print("\n--- Extracted Relationships (TXT) ---")
        for rel in txt_relations[:50]:  # print first 50
            safe_src = rel['source'].encode('ascii', 'replace').decode()
            safe_tgt = rel['target'].encode('ascii', 'replace').decode()
            print(f"  ({safe_src}) -[{rel['type']}]-> ({safe_tgt})")
        if len(txt_relations) > 50:
            print(f"  ... and {len(txt_relations) - 50} more relations.")
            
        # Load into Neo4j
        if txt_entities or txt_relations:
            print("\n[INGEST] Loading TXT elements into Neo4j...")
            upsert_entities_and_relations(txt_entities, txt_relations)
            print("[OK] Loaded TXT elements successfully!")
    else:
        print(f"[ERROR] File not found: {txt_path}")

    # 2. Ingest test_prd.docx
    docx_path = "data/test_prd.docx"
    print("\n================================================================================")
    print(f"[FILE] Processing: {docx_path}")
    if os.path.exists(docx_path):
        print("LlamaParse running on DOCX...")
        docx_content = parse_file(docx_path)
        
        print("[SEARCH] Extracting entities and relations from DOCX...")
        kg_docx = extract_knowledge_graph_elements(docx_content)
        docx_entities = kg_docx.get("entities", [])
        docx_relations = kg_docx.get("relations", [])
        
        print(f"Found {len(docx_entities)} entities and {len(docx_relations)} relationships in DOCX.")
        print("\n--- Extracted Entities (DOCX) ---")
        for ent in sorted(docx_entities, key=lambda x: x["name"]):
            safe_name = ent['name'].encode('ascii', 'replace').decode()
            print(f"  [{ent['label']}] {safe_name}")
            
        print("\n--- Extracted Relationships (DOCX) ---")
        for rel in docx_relations:
            safe_src = rel['source'].encode('ascii', 'replace').decode()
            safe_tgt = rel['target'].encode('ascii', 'replace').decode()
            print(f"  ({safe_src}) -[{rel['type']}]-> ({safe_tgt})")
            
        # Load into Neo4j
        if docx_entities or docx_relations:
            print("\n[INGEST] Loading DOCX elements into Neo4j...")
            upsert_entities_and_relations(docx_entities, docx_relations)
            print("[OK] Loaded DOCX elements successfully!")
    else:
        print(f"[ERROR] File not found: {docx_path}")

    print("\n================================================================================")
    print("[STATS] Current Graph Database Statistics:")
    try:
        node_count = run_read_query("MATCH (n:Entity) RETURN count(n) AS total")[0]["total"]
        rel_count = run_read_query("MATCH ()-[r]->() RETURN count(r) AS total")[0]["total"]
        print(f"  Total Entity Nodes: {node_count}")
        print(f"  Total Relationships: {rel_count}")
    except Exception as e:
        print(f"[ERROR] Failed to get database stats: {e}")

if __name__ == "__main__":
    main()
