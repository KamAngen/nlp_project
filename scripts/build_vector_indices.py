import sys
import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.service import KnowledgeService, KnowledgeRecord

def main():
    index_path = PROJECT_ROOT / "data/indices"
    emb_model_path = PROJECT_ROOT / "models/qwen/Qwen3_4B"
    
    # 1. Clear old indices
    if index_path.exists():
        print(f"Cleaning old indices at {index_path}...")
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Service (Auto-detect MPS/GPU)
    print("Initializing Service and Loading Qwen-4B to GPU (MPS)...")
    service = KnowledgeService(
        question_bank_path=PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl",
        case_bank_path=PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl",
        common_knowledge_path=PROJECT_ROOT / "data/legal_study_agent/common_knowledge.jsonl"
    )
    # This will trigger hardware-accelerated EmbeddingEngine
    service.load_indices(index_path, emb_model_path)
    
    # 3. Build indices with Progress Bar
    for bank_name in ["question_bank", "case_bank", "common_knowledge"]:
        records = getattr(service, bank_name)
        if not records: continue
        
        print(f"\nIndexing {bank_name} ({len(records)} records)...")
        from rag_engine.indexer import VectorIndex
        idx = VectorIndex(bank_name, service.embedding_engine)
        
        # We use the new dynamic batching from the engine
        # But we still need to loop for the progress bar to show something
        # We'll ask the engine what it thinks is a good batch size
        dyn_batch_size = service.embedding_engine._calculate_dynamic_batch_size()
        print(f"Auto-selected batch size: {dyn_batch_size}")
        
        for i in tqdm(range(0, len(records), dyn_batch_size), desc=f"Building {bank_name}"):
            batch = records[i : i + dyn_batch_size]
            idx.add_records(batch)
            
        idx.save(index_path / bank_name)
    
    print("\n[SUCCESS] All indices rebuilt with high performance (MPS + bfloat16 + BatchSize 32)!")

if __name__ == "__main__":
    main()
