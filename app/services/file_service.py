# 文件服务模块，负责处理文件的读写,存储操作
import json
import os
from app.config import INDEX_FILE, RESULTS_DIR

def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return []

def save_index(data):
    with open(INDEX_FILE, "w") as f:
        json.dump(data, f, indent=2)

def save_experiment_detail(exp_id, detail):
    (RESULTS_DIR / f"exp_{exp_id}.json").write_text(json.dumps(detail, indent=2))

def load_experiment_detail(exp_id):
    file = RESULTS_DIR / f"exp_{exp_id}.json"
    if file.exists():
        return json.loads(file.read_text())
    return None

def delete_experiment_files(exp_id):
    file = RESULTS_DIR / f"exp_{exp_id}.json"
    if file.exists():
        file.unlink()