import json
from pathlib import Path

def cleanup_file(path: Path, id_key: str):
    if not path.exists():
        return
    
    print(f"Cleaning {path}...")
    seen_ids = set()
    unique_records = []
    
    # Read all lines first
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Process from newest to oldest to keep latest version
    for line in reversed(lines):
        try:
            record = json.loads(line)
            rid = str(record.get(id_key) or record.get("id") or record.get("record_id"))
            if rid not in seen_ids:
                unique_records.append(line)
                seen_ids.add(rid)
        except:
            continue
            
    # Write back in original order
    with open(path, "w", encoding="utf-8") as f:
        for line in reversed(unique_records):
            f.write(line)
    
    print(f"Done. Kept {len(unique_records)} unique records (removed {len(lines) - len(unique_records)} duplicates).")

def main():
    root = Path("/Users/mmb/Downloads/nlp_project-main/data/legal_study_agent")
    cleanup_file(root / "question_bank.jsonl", "question_id")
    cleanup_file(root / "case_bank.jsonl", "case_id")
    cleanup_file(root / "common_knowledge.jsonl", "id")

if __name__ == "__main__":
    main()
