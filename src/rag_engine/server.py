from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
import uvicorn
from rag_engine.service import KnowledgeService
from rag_engine.schema import KnowledgeRecord
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Legal RAG Standalone Service")

# Mount the current directory to serve local JS files
# This allows serving http://localhost:2266/cytoscape.min.js
app.mount("/lib", StaticFiles(directory="src/rag_engine"), name="lib")

# Enable CORS for the test webpage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global service
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
service = KnowledgeService(
    question_bank_path=PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl",
    case_bank_path=PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl",
    common_knowledge_path=PROJECT_ROOT / "data/legal_study_agent/common_knowledge.jsonl"
)

# Load assets
index_path = PROJECT_ROOT / "data/indices"
emb_model_path = PROJECT_ROOT / "models/qwen/Qwen3_4B"
reranker_path = PROJECT_ROOT / "models/reranker/bge-reranker-base"
graph_path = PROJECT_ROOT / "data/indices/legal_graph.json"

# Debug: Print Paths
print(f">>> Server Root: {PROJECT_ROOT}")
print(f">>> Index Path: {index_path} (exists: {index_path.exists()})")
print(f">>> Model Path: {emb_model_path} (exists: {emb_model_path.exists()})")

service.load_indices(index_path, emb_model_path)
service.load_reranker(reranker_path)
service.load_graph(graph_path)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True
    mode: str = "hybrid"
    sources: Optional[List[str]] = None

class UpdateRequest(BaseModel):
    source_type: str
    record_id: Optional[str] = None
    title: str
    content: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

@app.get("/stats")
def get_stats():
    return service.summary()

@app.post("/search")
def unified_search(req: SearchRequest):
    hits = service.search(req.query, mode=req.mode, sources=req.sources, rerank=req.rerank, top_k=req.top_k)
    import math
    def sigmoid(x): return 1 / (1 + math.exp(-x))
    for hit in hits:
        if req.mode != "lexical" and req.rerank:
            if hit.score > 15: hit.score = 0.99
            elif hit.score < -15: hit.score = 0.01
            else: hit.score = sigmoid(hit.score)
        # Enrich with graph data if available
        if service.graph:
            hit.metadata["related_statutes"] = service.graph.get_related_statutes(hit.record_id)
            hit.metadata["related_cases"] = list(service.graph.get_related_cases(hit.record_id))
            
    return hits

@app.post("/search/lexical")
def search_lexical(req: SearchRequest):
    lexical_hits = service.search(req.query, mode="lexical", sources=req.sources, top_k=req.top_k)
    return lexical_hits

@app.post("/search/embedding")
def search_embedding(req: SearchRequest):
    embedding_hits = service.search(req.query, mode="embedding", sources=req.sources, rerank=req.rerank, top_k=req.top_k)
    
    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for hit in embedding_hits:
        if req.rerank:
            hit.score = sigmoid(hit.score)
            
        if service.graph:
            hit.metadata["related_statutes"] = service.graph.get_related_statutes(hit.record_id)
            hit.metadata["related_cases"] = list(service.graph.get_related_cases(hit.record_id))
            
    return embedding_hits

@app.post("/update")
def update_knowledge(request: UpdateRequest):
    actual_id = service.generate_id(request.source_type, request.record_id)
    record = KnowledgeRecord(
        source_type=request.source_type,
        record_id=actual_id,
        title=request.title,
        content=request.content,
        tags=request.tags,
        metadata=request.metadata
    )
    service.update_record(record, persist=True)
    return {"status": "success", "record_id": actual_id}

upload_tasks = {}

def process_batch_task(task_id: str, records: List[KnowledgeRecord]):
    try:
        chunk_size = 32
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            service.update_records_batch(chunk, persist=True)
            upload_tasks[task_id]["processed"] += len(chunk)
            
        upload_tasks[task_id]["status"] = "completed"
    except Exception as e:
        upload_tasks[task_id]["status"] = "failed"
        upload_tasks[task_id]["error"] = str(e)

@app.post("/update_batch")
def update_knowledge_batch(requests: List[UpdateRequest], background_tasks: BackgroundTasks):
    results = []
    records = []
    for req in requests:
        actual_id = service.generate_id(req.source_type, req.record_id)
        record = KnowledgeRecord(
            source_type=req.source_type,
            record_id=actual_id,
            title=req.title,
            content=req.content,
            tags=req.tags,
            metadata=req.metadata
        )
        records.append(record)
        results.append(actual_id)
    
    task_id = str(uuid.uuid4())
    upload_tasks[task_id] = {
        "status": "processing",
        "total": len(records),
        "processed": 0,
        "ids": results
    }
    
    if records:
        background_tasks.add_task(process_batch_task, task_id, records)
        
    return {"status": "accepted", "task_id": task_id, "count": len(results)}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in upload_tasks:
        return {"status": "not_found"}
    return upload_tasks[task_id]

@app.get("/record/{record_id}")
def get_record(record_id: str):
    record = service.get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record.to_dict()

@app.get("/graph/{node_id}")
def get_graph_detail(node_id: str):
    try:
        if not service.graph:
            return {"center": {"id": node_id, "title": "图谱引擎未就绪"}, "statutes": [], "related_cases": []}
        
        record = service.get_record_by_id(node_id)
        record_title = str(record.title) if record else f"节点:{node_id[:8]}"
        
        statutes = list(service.graph.get_related_statutes(node_id))
        
        related_cases_ids = list(service.graph.get_related_cases(node_id))
        related_cases_data = []
        for rc_id in related_cases_ids[:10]: # Return up to 10
            rc_record = service.get_record_by_id(str(rc_id))
            related_cases_data.append({
                "id": str(rc_id),
                "title": str(rc_record.title) if rc_record else f"节点:{str(rc_id)[:8]}"
            })
            
        similar_cases_data = []
        similar_questions_data = []
        
        if record:
            # Construct a comprehensive query from the node's text
            query_text = f"{record.title}\n{record.content}"
            
            # Find semantically similar cases
            case_hits = service.search(query_text, sources=["case_bank"], mode="embedding", top_k=4)
            for hit in case_hits:
                if hit.record_id != node_id and hit.record_id not in [rc["id"] for rc in related_cases_data]:
                    similar_cases_data.append({"id": hit.record_id, "title": hit.title, "score": float(hit.score)})
            
            # Find semantically similar questions
            question_hits = service.search(query_text, sources=["question_bank"], mode="embedding", top_k=3)
            for hit in question_hits:
                if hit.record_id != node_id:
                    similar_questions_data.append({"id": hit.record_id, "title": hit.title, "score": float(hit.score)})

        return {
            "center": {"id": str(node_id), "title": record_title},
            "statutes": [str(s) for s in statutes],
            "related_cases": related_cases_data,
            "similar_cases": similar_cases_data[:3],
            "similar_questions": similar_questions_data[:3]
        }
    except Exception as e:
        print(f"Graph API Error: {e}")
        return {"center": {"id": node_id, "title": f"系统错误: {str(e)}"}, "statutes": [], "related_cases": []}

@app.get("/sample")
def sample(mastery: float = 0.5, topic: str | None = None):
    # Call sample_questions from service
    # If topic is provided, it will prioritize matching questions
    questions = service.sample_questions(
        question_count=5, 
        user_mastery_level=mastery,
        topic=topic
    )
    return questions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2266)
