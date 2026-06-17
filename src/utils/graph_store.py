import re
from typing import List, Dict, Any, Tuple
from loguru import logger
from neo4j import GraphDatabase
from src.config import settings

_NEO4J_DRIVER = None

def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        if not settings.neo4j_uri:
            raise ValueError("NEO4J_URI is not set in settings.")
        _NEO4J_DRIVER = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
    return _NEO4J_DRIVER

def run_write_query(query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    db_name = settings.neo4j_database or "neo4j"
    driver = get_neo4j_driver()
    with driver.session(database=db_name) as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]

def run_read_query(query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    db_name = settings.neo4j_database or "neo4j"
    driver = get_neo4j_driver()
    with driver.session(database=db_name) as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]

def upsert_entities_and_relations(entities: List[Dict[str, str]], relations: List[Dict[str, str]]) -> None:
    """
    Saves extracted entities and relations to Neo4j.
    """
    logger.info(f"Upserting {len(entities)} entities and {len(relations)} relations to Neo4j...")
    
    # 1. Collect all unique entities from both lists (to prevent MATCH failures in step 2)
    entity_map = {}
    for ent in entities:
        entity_map[ent["name"]] = ent.get("label", "Entity")
        
    for rel in relations:
        src = rel["source"]
        tgt = rel["target"]
        if src not in entity_map:
            entity_map[src] = "Entity"
        if tgt not in entity_map:
            entity_map[tgt] = "Entity"
            
    entities_payload = [{"name": name, "label": label} for name, label in entity_map.items()]
    
    # Upsert all entities first
    entity_query = """
    UNWIND $entities AS ent
    MERGE (e:Entity {name: ent.name})
    SET e.label = ent.label
    """
    run_write_query(entity_query, {"entities": entities_payload})
    
    # 2. Group relations by sanitized type and upsert them using MATCH (preventing lock contention)
    from collections import defaultdict
    relations_by_type = defaultdict(list)
    for rel in relations:
        source = rel["source"]
        target = rel["target"]
        rel_type = rel["type"]
        
        # Sanitize rel_type for Cypher safety (only allow alphanumeric and underscore)
        clean_rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type).upper()
        if not clean_rel_type:
            clean_rel_type = "RELATED_TO"
            
        relations_by_type[clean_rel_type].append({"source": source, "target": target})
        
    for clean_rel_type, rels in relations_by_type.items():
        relation_query = f"""
        UNWIND $rels AS rel
        MATCH (s:Entity {{name: rel.source}})
        MATCH (t:Entity {{name: rel.target}})
        MERGE (s)-[r:{clean_rel_type}]->(t)
        """
        run_write_query(relation_query, {"rels": rels})
        
    logger.info("Graph DB upsert completed successfully.")

def get_neighbors(entity_name: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Retrieves all direct neighbors of a specific entity using fuzzy matching.
    """
    clean_name = entity_name.strip()
    if clean_name.lower().startswith("the "):
        clean_name = clean_name[4:].strip()
        
    query = """
    MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name)
    MATCH (e)-[r]-(n:Entity)
    RETURN type(r) AS rel_type, n.name AS neighbor_name, n.label AS neighbor_label
    LIMIT $limit
    """
    records = run_read_query(query, {"name": clean_name, "limit": limit})
    return records

def get_two_hop_neighbors(entity_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves 2-hop neighbors of a specific entity using fuzzy matching, representing paths:
    (Entity) <-> (Neighbor) <-> (Neighbor's Neighbor)
    """
    clean_name = entity_name.strip()
    if clean_name.lower().startswith("the "):
        clean_name = clean_name[4:].strip()
        
    query = """
    MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name)
    MATCH (e)-[r1]-(n1:Entity)
    OPTIONAL MATCH (n1)-[r2]-(n2:Entity)
    WHERE n2 <> e
    RETURN e.name AS entity_name, 
           type(r1) AS r1_type, 
           n1.name AS n1_name, 
           n1.label AS n1_label,
           type(r2) AS r2_type, 
           n2.name AS n2_name, 
           n2.label AS n2_label
    LIMIT $limit
    """
    records = run_read_query(query, {"name": clean_name, "limit": limit})
    return records

def find_shortest_path(start_entity: str, end_entity: str, max_depth: int = 5) -> Dict[str, Any]:
    """
    Finds the shortest path between start_entity and end_entity.
    Returns the path details (nodes and relations).
    """
    clean_start = start_entity.strip()
    if clean_start.lower().startswith("the "):
        clean_start = clean_start[4:].strip()
    clean_end = end_entity.strip()
    if clean_end.lower().startswith("the "):
        clean_end = clean_end[4:].strip()

    # Safe interpolation of max_depth
    depth_str = f"*..{int(max_depth)}"
    query = f"""
    MATCH (start:Entity) WHERE toLower(start.name) CONTAINS toLower($start_entity)
    MATCH (end:Entity) WHERE toLower(end.name) CONTAINS toLower($end_entity)
    MATCH p = shortestPath((start)-[{depth_str}]-(end))
    RETURN p
    """
    records = run_read_query(query, {"start_entity": clean_start, "end_entity": clean_end})
    
    if not records or not records[0].get("p"):
        return {"found": False, "path": []}
    
    path = records[0]["p"]
    # Extract nodes and relationship types from path
    path_nodes = []
    path_rels = []
    
    if isinstance(path, list):
        # record.data() converts path to a list of alternating dicts (nodes) and strings (relationship types)
        # e.g., [{'name': 'Alice', 'label': 'Person'}, 'KNOWS', {'name': 'Bob', 'label': 'Person'}]
        for i, item in enumerate(path):
            if i % 2 == 0:
                # Node
                path_nodes.append({
                    "name": item.get("name"),
                    "label": item.get("label", "Entity")
                })
            else:
                # Relationship type string connecting previous node to next node
                prev_node = path[i - 1]
                next_node = path[i + 1]
                path_rels.append({
                    "start": prev_node.get("name"),
                    "end": next_node.get("name"),
                    "type": item if isinstance(item, str) else str(item)
                })
    else:
        # Fallback in case it's somehow a Neo4j Path object directly
        for node in getattr(path, "nodes", []):
            path_nodes.append({
                "name": node.get("name"),
                "label": list(node.labels)[0] if getattr(node, "labels", None) else "Entity"
            })
        for rel in getattr(path, "relationships", []):
            start_node = rel.nodes[0] if getattr(rel, "nodes", None) else None
            end_node = rel.nodes[1] if getattr(rel, "nodes", None) and len(rel.nodes) > 1 else None
            path_rels.append({
                "start": start_node.get("name") if start_node else None,
                "end": end_node.get("name") if end_node else None,
                "type": getattr(rel, "type", "RELATED_TO")
            })
        
    return {
        "found": True,
        "nodes": path_nodes,
        "relations": path_rels
    }
