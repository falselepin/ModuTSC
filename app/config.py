#提取配置常量
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent   # 指向 ModuTSC_new
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
INDEX_FILE = RESULTS_DIR / "index.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DATASET_META = {
    "Atlanta": {"typeTag": "动脉", "description": "亚特兰大城市主干道，潮汐车流"},
    "grid": {"typeTag": "网格", "description": "4×4网格，16个路口，均匀车流"},
    "Hangzhou1": {"typeTag": "环形", "description": "杭州城区路网，车流波动"},
    "Hangzhou2": {"typeTag": "环形", "description": "杭州郊区路网，中等流量"},
    "Jinan": {"typeTag": "动脉", "description": "济南主干道，高饱和度"},
    "LargeGrid": {"typeTag": "网格", "description": "8×8大型网格，多路口协同"},
    "LosAngeles": {"typeTag": "动脉", "description": "洛杉矶高速路网，复杂拓扑"},
    "Manhattan": {"typeTag": "网格", "description": "曼哈顿标准网格，经典场景"},
    "Monaco": {"typeTag": "单点", "description": "摩纳哥单一路口，高密度"},
    "test_simple": {"typeTag": "测试", "description": "简单测试路网，基础功能验证"},
    "test_tls": {"typeTag": "测试", "description": "交通灯测试专用"},
}

AVAILABLE_MODELS = [
    {"id": "dqn", "name": "DQN", "type": "DRL"},
    {"id": "ma2c", "name": "MA2C", "type": "MARL"},
    {"id": "frap", "name": "FRAP", "type": "RL"},
    {"id": "colight", "name": "CoLight", "type": "MARL"},
    {"id": "igrl", "name": "IGRL", "type": "RL"},
    {"id": "fixed_time", "name": "Fixed-Time", "type": "Rule"},
    {"id": "max_pressure", "name": "MaxPressure", "type": "Rule"},
]

# 默认训练 / 评估参数
DEFAULT_TRAINING = {"warmup_steps": 0, "num_epochs": 2, "episodes_per_epoch": 1}
DEFAULT_EVALUATION = {"eval_frequency": 1, "eval_steps": 20}

# LLM Agent 配置
# 支持 OpenAI / DeepSeek 等兼容 OpenAI API 格式的服务
LLM_API_KEY = "sk-409ae3b4b2ea45a985f6032bb71c1d16"  # 或直接填写 API Key
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
