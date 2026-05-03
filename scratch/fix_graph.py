from pathlib import Path
import sys
from typing import Dict, Any

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.service import KnowledgeService
from rag_engine.graph import LegalKnowledgeGraph

def fix():
    print("Initializing Service to read raw data...")
    root = PROJECT_ROOT
    service = KnowledgeService(
        question_bank_path=root/'data/legal_study_agent/question_bank.jsonl',
        case_bank_path=root/'data/legal_study_agent/case_bank.jsonl',
        common_knowledge_path=root/'data/legal_study_agent/common_knowledge.jsonl'
    )
    
    print(f"Building Graph from {len(service.case_bank)} cases...")
    graph = LegalKnowledgeGraph()
    
    # Convert KnowledgeRecord to the dict format expected by build_from_cases
    case_dicts = []
    for r in service.case_bank:
        # The build_from_cases expects 'case_id' and 'statutes'
        # In our schema, record_id is case_id, and statutes are in tags or metadata
        statutes = r.tags
        if r.metadata and "statutes" in r.metadata:
            statutes = r.metadata["statutes"]
            
        case_dicts.append({
            "case_id": r.record_id,
            "title": r.title,
            "statutes": statutes
        })
    
    graph.build_from_cases(case_dicts)
    
    output_path = root / 'data/indices/legal_graph.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(output_path)
    print(f"✅ Graph rebuilt and saved to {output_path}")

if __name__ == "__main__":
    fix()
