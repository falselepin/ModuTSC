from fastapi import APIRouter, HTTPException
from app.config import AVAILABLE_MODELS
from modutsc.api import scaffold_config
import yaml

router = APIRouter(prefix="/api", tags=["models"])

AVAILABLE_METRICS = [
    {"value": "reward", "label": "平均奖励 (avg_reward)", "type": "scalar", "source": "reward_curve"},
    {"value": "ATT", "label": "平均行程时间 (ATT)", "type": "scalar", "source": "metrics.ATT"},
    {"value": "AQL", "label": "平均排队长度 (AQL)", "type": "scalar", "source": "metrics.AQL"},
    {"value": "Throughput", "label": "吞吐量 (Throughput)", "type": "scalar", "source": "metrics.Throughput"},
    {"value": "RealDelay", "label": "实际延迟 (RealDelay)", "type": "scalar", "source": "metrics.RealDelay"},
    {"value": "TripFlow", "label": "行程流量 (TripFlow)", "type": "scalar", "source": "metrics.TripFlow"},
    {"value": "avg_reward", "label": "平均奖励", "type": "scalar", "source": "metrics.avg_reward"},
    {"value": "queue", "label": "平均排队长度 (queue)", "type": "scalar", "source": "metrics.queue"},
    {"value": "arrived", "label": "到达车辆数", "type": "scalar", "source": "metrics.arrived"},
    {"value": "departed", "label": "出发车辆数", "type": "scalar", "source": "metrics.departed"},
    {"value": "sim_time", "label": "仿真时间 (s)", "type": "scalar", "source": "metrics.sim_time"},
    {"value": "total_loss", "label": "总损失 (total_loss)", "type": "scalar", "source": "metrics.total_loss"},
    {"value": "steps", "label": "决策步数", "type": "scalar", "source": "metrics.steps"},
    {"value": "epsilon", "label": "探索率 (epsilon)", "type": "scalar", "source": "metrics.epsilon"},
]

# ── 论文复现预设 ──

PAPER_PRESETS = [
    {
        "id": "fixed_time",
        "title": "固定配时信号控制 (Fixed-Time)",
        "authors": "经典交通工程方法",
        "year": "—",
        "conference": "基线方法",
        "algorithm": "Fixed-Time",
        "description": "固定周期的信号配时方案，作为所有强化学习方法的下界基线。",
        "orchestrator": "rule",
        "selections": {
            "observer": "flat_lane",
            "actor": "phase",
            "reward": "composite",
            "collector": "replay",
            "algorithm": "fixed_time",
        },
        "config_params": {
            "algorithm": {"num_phase": 4, "interval": 20},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 1.0},
        },
        "dataset_id": "Monaco",
        "training": {"warmup_steps": 0, "num_epochs": 1, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 1, "eval_steps": 500},
        "tags": ["规则", "基线"],
    },
    {
        "id": "max_pressure",
        "title": "MaxPressure: 基于压力的自适应信号控制",
        "authors": "Varaiya, P.",
        "year": 2013,
        "conference": "Transportation Research",
        "algorithm": "MaxPressure",
        "description": "基于车道压力的自适应信号控制算法，选择压力最大的相位组合。",
        "orchestrator": "rule",
        "selections": {
            "observer": "flat_lane",
            "actor": "phase",
            "reward": "composite",
            "collector": "replay",
            "algorithm": "max_pressure",
        },
        "config_params": {
            "algorithm": {"num_phase": 4, "min_duration": 10},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 1.0},
        },
        "dataset_id": "Monaco",
        "training": {"warmup_steps": 0, "num_epochs": 1, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 1, "eval_steps": 500},
        "tags": ["规则", "自适应"],
    },
    {
        "id": "sotl",
        "title": "SOTL: 自适应信号控制",
        "authors": "经典自适应方法",
        "year": "—",
        "conference": "基线方法",
        "algorithm": "SOTL",
        "description": "基于车辆数阈值的自适应信号控制算法，根据排队车辆数动态切换相位。",
        "orchestrator": "rule",
        "selections": {
            "observer": "flat_lane",
            "actor": "phase",
            "reward": "composite",
            "collector": "replay",
            "algorithm": "sotl",
        },
        "config_params": {
            "algorithm": {"num_phase": 4, "min_duration": 10, "min_green_veh": 20, "max_red_veh": 0},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 1.0},
        },
        "dataset_id": "Monaco",
        "training": {"warmup_steps": 0, "num_epochs": 1, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 1, "eval_steps": 500},
        "tags": ["规则", "自适应"],
    },
    {
        "id": "dqn",
        "title": "深度 Q 网络在交通信号控制中的应用",
        "authors": "Mnih et al. / 经典 DRL 方法",
        "year": 2015,
        "conference": "Nature / 经典方法",
        "algorithm": "DQN",
        "description": "使用深度 Q 网络 (DQN) 学习信号控制策略，epsilon-greedy 探索 + 经验回放。",
        "orchestrator": "single",
        "selections": {
            "observer": "flat_lane",
            "actor": "phase",
            "reward": "composite",
            "collector": "replay",
            "algorithm": "dqn",
        },
        "config_params": {
            "algorithm": {"lr": 0.0001, "gamma": 0.95, "tau": 0.005, "hidden_size": 128},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 20.0},
        },
        "dataset_id": "Hangzhou1",
        "training": {"warmup_steps": 500, "num_epochs": 20, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 5, "eval_steps": 500},
        "tags": ["DRL", "值函数"],
    },
    {
        "id": "colight",
        "title": "CoLight: 基于注意力机制的多路口协作信号控制",
        "authors": "Wei, H. et al.",
        "year": 2019,
        "conference": "AAAI 2019",
        "algorithm": "CoLight",
        "description": "利用注意力机制建模路口间交互，实现多路口协作信号控制。",
        "orchestrator": "single",
        "selections": {
            "observer": "colight",
            "actor": "phase",
            "reward": "composite",
            "collector": "replay",
            "algorithm": "colight",
        },
        "config_params": {
            "algorithm": {"lr": 0.0001, "gamma": 0.95, "tau": 0.005},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 20.0},
        },
        "dataset_id": "Hangzhou1",
        "training": {"warmup_steps": 500, "num_epochs": 20, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 5, "eval_steps": 500},
        "tags": ["DRL", "注意力", "多路口"],
    },
    {
        "id": "frap",
        "title": "FRAP: 基于相位竞争关系的信号控制",
        "authors": "Zheng, G. et al.",
        "year": 2019,
        "conference": "CIKM 2019",
        "algorithm": "FRAP",
        "description": "通过建模相位间的竞争关系 (Phase Competition) 来学习信号控制策略。",
        "orchestrator": "single",
        "selections": {
            "observer": "frap",
            "actor": "phase",
            "reward": "composite",
            "collector": "replay",
            "algorithm": "frap",
        },
        "config_params": {
            "observer": {"features": ["num"]},
            "algorithm": {"lr": 0.0005, "gamma": 0.95, "tau": 0.01},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 20.0},
        },
        "dataset_id": "LosAngeles",
        "training": {"warmup_steps": 500, "num_epochs": 20, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 5, "eval_steps": 500},
        "tags": ["DRL", "相位竞争", "多路口"],
    },
    {
        "id": "ma2c",
        "title": "MA2C: 多智能体 Actor-Critic 协作信号控制",
        "authors": "Chu, T. et al.",
        "year": 2020,
        "conference": "AAAI 2020",
        "algorithm": "MA2C",
        "description": "基于 A2C 架构的多智能体强化学习算法，利用相邻路口信息进行协作。",
        "orchestrator": "ma2c",
        "selections": {
            "observer": "ma2c",
            "actor": "phase",
            "reward": "composite",
            "collector": "ma2c",
            "algorithm": "ma2c_agent",
        },
        "config_params": {
            "algorithm": {"gamma": 0.99, "learning_rate": 0.0001, "value_coef": 1.0, "max_grad_norm": 40.0},
            "reward": {"metrics": {"waiting": -1.0}, "reward_norm": 20.0},
            "collector": {"batch_size": 120, "n_agent": 16},
        },
        "dataset_id": "grid",
        "training": {"warmup_steps": 500, "num_epochs": 20, "episodes_per_epoch": 1},
        "evaluation": {"eval_frequency": 5, "eval_steps": 500},
        "tags": ["MARL", "多智能体", "Actor-Critic"],
    },
]


@router.get("/models")
def get_models():
    return AVAILABLE_MODELS


@router.get("/metrics")
def get_metrics():
    return AVAILABLE_METRICS


@router.post("/models/scaffold")
def scaffold_config_endpoint(data: dict):
    orch_name = data.get("orch_name")
    selections = data.get("selections", {})
    config_params = data.get("config_params", {})

    if not orch_name:
        raise HTTPException(status_code=400, detail="缺少编排器名称")

    cfg = scaffold_config(orch_name, selections, config_params)
    yaml_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return {"yaml": yaml_text}


@router.get("/papers")
def get_papers():
    """返回论文复现预设列表，每个预设包含完整配置。"""
    results = []
    for preset in PAPER_PRESETS:
        try:
            cfg = scaffold_config(
                preset["orchestrator"],
                {**preset["selections"], "environment": "sumo"},
                preset.get("config_params", {}),
            )
            cfg["environment"]["config"]["roadnet_file"] = f"data/{preset['dataset_id']}/roadnet.net.xml"
            cfg["environment"]["config"]["flow_file"] = f"data/{preset['dataset_id']}/flow_0.rou.xml"
            cfg["training"] = preset["training"]
            cfg["evaluation"] = preset["evaluation"]
            cfg["evaluation"]["checkpoint_dir"] = f"checkpoints/{preset['id']}_reproduce/"

            yaml_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)

            results.append({
                "id": preset["id"],
                "title": preset["title"],
                "authors": preset["authors"],
                "year": preset["year"],
                "conference": preset["conference"],
                "algorithm": preset["algorithm"],
                "description": preset["description"],
                "tags": preset.get("tags", []),
                "dataset_id": preset["dataset_id"],
                "orchestrator": preset["orchestrator"],
                "selections": preset["selections"],
                "config": cfg,
                "yaml": yaml_text,
            })
        except Exception as e:
            results.append({
                "id": preset["id"],
                "title": preset["title"],
                "error": str(e),
            })
    return results
