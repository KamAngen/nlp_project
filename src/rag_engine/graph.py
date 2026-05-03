from __future__ import annotations

import networkx as nx
from pathlib import Path
from typing import List, Set, Dict, Any
import json

class LegalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def build_from_cases(self, case_records: List[Dict[str, Any]]):
        """
        Build the graph from case records. 
        Each record should have 'case_id' and 'statutes' (list of names).
        """
        for record in case_records:
            case_id = record.get("case_id")
            if not case_id:
                continue
                
            self.graph.add_node(case_id, type="case", title=record.get("title"))
            
            statutes = record.get("statutes", [])
            for statute in statutes:
                statute_name = str(statute).strip()
                if not statute_name:
                    continue
                
                self.graph.add_node(statute_name, type="statute")
                self.graph.add_edge(case_id, statute_name, relation="references")

    def get_related_statutes(self, case_id: str) -> List[str]:
        if not self.graph.has_node(case_id):
            return []
        return [neighbor for neighbor in self.graph.neighbors(case_id) 
                if self.graph.nodes[neighbor].get("type") == "statute"]

    def get_related_cases(self, case_id: str, depth: int = 1) -> Set[str]:
        """
        Find cases related to the given case via common statutes.
        """
        if not self.graph.has_node(case_id):
            return set()
            
        related_cases = set()
        # Find statutes connected to this case
        statutes = self.get_related_statutes(case_id)
        for statute in statutes:
            # Find other cases connected to these statutes
            for neighbor in self.graph.neighbors(statute):
                if neighbor != case_id and self.graph.nodes[neighbor].get("type") == "case":
                    related_cases.add(neighbor)
                    
        return related_cases

    def save(self, path: str | Path):
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> LegalKnowledgeGraph:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.graph = nx.node_link_graph(data)
        return obj
