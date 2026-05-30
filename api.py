# api.py (项目根目录)
import os
from pathlib import Path
os.chdir(Path(__file__).parent)  # 保持工作目录

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")