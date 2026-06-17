import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from src.utils.graph_store import run_write_query
from loguru import logger

def purge_neo4j():
    logger.info("Purging all nodes and relationships from Neo4j...")
    query = "MATCH (n) DETACH DELETE n"
    try:
        run_write_query(query)
        logger.info("✅ Neo4j database successfully purged!")
    except Exception as e:
        logger.error(f"❌ Failed to purge Neo4j database: {e}")

if __name__ == "__main__":
    purge_neo4j()
