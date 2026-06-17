import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from src.utils.graph_store import run_read_query

def check_nodes():
    nodes = run_read_query("MATCH (n:Entity) RETURN n.name AS name, n.label AS label LIMIT 50")
    print(f"Total nodes in database: {len(nodes)}")
    for n in nodes:
        name = n["name"]
        label = n["label"]
        safe_name = name.encode('ascii', 'replace').decode()
        print(f"  [{label}] {safe_name}")

if __name__ == "__main__":
    check_nodes()
