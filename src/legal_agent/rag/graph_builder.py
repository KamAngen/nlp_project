from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


def build_citation_graph(chunks: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, str], nx.DiGraph]:
    graph = nx.DiGraph()
    doc_to_chunks: dict[str, list[str]] = defaultdict(list)
    chunk_to_doc: dict[str, str] = {}

    for chunk in chunks:
        doc_node = f"doc::{chunk['document_id']}"
        chunk_node = f"chunk::{chunk['chunk_id']}"
        graph.add_node(
            doc_node,
            kind="document",
            title=chunk["document_title"],
            effect_level=chunk["effect_level"],
            effect_rank=chunk["effect_rank"],
        )
        graph.add_node(
            chunk_node,
            kind="chunk",
            title=chunk["document_title"],
            chunk_id=chunk["chunk_id"],
        )
        graph.add_edge(doc_node, chunk_node, relation="has_chunk")
        graph.add_edge(chunk_node, doc_node, relation="belongs_to")
        doc_to_chunks[chunk["normalized_title"]].append(chunk["chunk_id"])
        chunk_to_doc[chunk["chunk_id"]] = chunk["normalized_title"]

    doc_nodes_by_title = {
        attrs.get("title"): node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "document"
    }
    for chunk in chunks:
        source_doc_node = f"doc::{chunk['document_id']}"
        for ref in chunk.get("cross_references", []):
            target_node = doc_nodes_by_title.get(ref)
            if target_node is not None:
                graph.add_edge(source_doc_node, target_node, relation="cites")

    return dict(doc_to_chunks), chunk_to_doc, graph


def graph_to_json(graph: nx.DiGraph) -> dict[str, Any]:
    return json_graph.node_link_data(graph)


def graph_from_json(payload: dict[str, Any]) -> nx.DiGraph:
    return json_graph.node_link_graph(payload)
