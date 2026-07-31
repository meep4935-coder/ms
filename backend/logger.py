import os
import json
import time
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
JSONL_PATH = os.path.join(LOGS_DIR, "pipeline.jsonl")

def write_scan_log(log_entry: dict) -> str:
    """
    Writes a comprehensive scan log entry both as an individual human-readable JSON
    and as a single-line appended entry in pipeline.jsonl.
    Returns the file path of the per-scan JSON.
    """
    timestamp_ms = int(time.time() * 1000)
    scan_id = f"scan_{timestamp_ms}"
    log_entry["scan_id"] = scan_id

    # 1. Write per-scan JSON file
    filename = f"{scan_id}.json"
    filepath = os.path.join(LOGS_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[Logger] Error writing per-scan JSON: {e}")

    # 2. Append to pipeline.jsonl
    try:
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Logger] Error writing to pipeline.jsonl: {e}")

    return filepath
