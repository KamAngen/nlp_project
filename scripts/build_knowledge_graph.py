import sys
import json
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.graph import LegalKnowledgeGraph

def main():
    case_bank_path = PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl"
    output_path = PROJECT_ROOT / "data/indices/legal_graph.json"
    
    print(f"Reading cases from {case_bank_path}...")
    cases = []
    with open(case_bank_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
                
    question_bank_path = PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl"
    print(f"Reading questions from {question_bank_path}...")
    questions = []
    with open(question_bank_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                # Map question tags to statutes field for graph builder
                q["case_id"] = q["question_id"]
                q["statutes"] = q.get("tags", [])
                questions.append(q)

    print(f"Building graph for {len(cases)} cases and {len(questions)} questions...")
    graph = LegalKnowledgeGraph()
    graph.build_from_cases(cases)
    graph.build_from_cases(questions)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(output_path)
    print(f"Graph saved to {output_path}")
    print(f"Nodes: {len(graph.graph.nodes)}, Edges: {len(graph.graph.edges)}")

if __name__ == "__main__":
    main()
