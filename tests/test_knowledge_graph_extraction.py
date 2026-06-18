from unittest.mock import patch

from src.utils.entity_extractor import (
    extract_entities,
    extract_knowledge_graph_elements,
)
from src.utils.graph_store import upsert_entities_and_relations


WEKRAFT_TEXT = """
Ronit Rai leads Project WeKraft.
Akash Sharma develops the WeKraft API.
Riya Kapoor designs the WeKraft user experience.
Mia Chen tests the WeKraft release.
Project WeKraft depends on AWS infrastructure.
Project WeKraft extends Project Atlas.
Project WeKraft supports Project Orion.
"""


def test_wekraft_entities_and_connections_are_extracted():
    graph = extract_knowledge_graph_elements(WEKRAFT_TEXT)
    entity_names = {entity["name"] for entity in graph["entities"]}
    triples = {
        (relation["source"], relation["type"], relation["target"])
        for relation in graph["relations"]
    }

    assert {"Project WeKraft", "Ronit Rai", "Akash Sharma", "Riya Kapoor", "Mia Chen"} <= entity_names
    assert ("Ronit Rai", "LEAD", "Project WeKraft") in triples
    assert ("Project WeKraft", "EXTEND", "Project Atlas") in triples
    assert ("Project WeKraft", "SUPPORT", "Project Orion") in triples


def test_duplicate_relations_are_collapsed_before_neo4j():
    graph = extract_knowledge_graph_elements(
        "Ronit Rai leads Project WeKraft. Ronit Rai leads Project WeKraft."
    )
    assert graph["relations"] == [
        {"source": "Ronit Rai", "type": "LEAD", "target": "Project WeKraft"}
    ]


def test_wekraft_query_is_available_for_graph_retrieval():
    names = {
        entity["name"]
        for entity in extract_entities(
            "Who works on Project WeKraft and how is it connected to Project Atlas?"
        )
    }
    assert {"Project WeKraft", "Project Atlas"} <= names


def test_upsert_sends_each_unique_relation_once():
    graph = extract_knowledge_graph_elements(WEKRAFT_TEXT)
    calls = []

    def capture(query, parameters=None):
        calls.append((query, parameters or {}))
        return []

    with patch("src.utils.graph_store.run_write_query", side_effect=capture):
        upsert_entities_and_relations(graph["entities"], graph["relations"])

    relation_total = sum(len(parameters["rels"]) for _, parameters in calls[1:])
    assert relation_total == len(graph["relations"])
