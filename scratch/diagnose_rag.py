import sys, os
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv()

from src.utils.entity_extractor import extract_entities, extract_knowledge_graph_elements
from src.utils.graph_store import run_read_query

# ── 1. Entity extraction on TXT document ─────────────────────────────────────
with open('data/enterprise_knowledge_base.txt', 'r', encoding='utf-8') as f:
    txt = f.read()[:3000]

print('=== TXT FILE (first 3000 chars) ENTITIES ===')
ents = extract_entities(txt)
print(f'Found {len(ents)} entities:')
for e in ents[:20]:
    print(f'  [{e["label"]}] {e["name"]}')

# ── 2. Query-level entity extraction ─────────────────────────────────────────
queries = [
    "who is sarah and what skill she has",
    "who is the manager of the department responsible for the AI Assistant feature",
    "What projects does Rohan contribute to",
]
print()
for q in queries:
    qents = extract_entities(q)
    print(f'QUERY: "{q}"')
    print(f'  -> Extracted entities: {[e["name"] for e in qents]}')

# ── 3. Neo4j total node/rel count ─────────────────────────────────────────────
print()
print('=== NEO4J STATS ===')
try:
    count = run_read_query("MATCH (n:Entity) RETURN count(n) AS total")
    print(f'  Entity nodes: {count[0]["total"]}')
    rels = run_read_query("MATCH ()-[r]->() RETURN count(r) AS total")
    print(f'  Relations: {rels[0]["total"]}')
    sample = run_read_query("MATCH (n:Entity) RETURN n.name AS name, n.label AS label LIMIT 15")
    print('  Sample nodes:')
    for r in sample:
        print(f'    [{r["label"]}] {r["name"]}')
except Exception as e:
    print(f'  Neo4j error: {e}')
