from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import httpx
import asyncio

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.services.file_service import load_experiment_detail
from app.routers.models import AVAILABLE_METRICS

router = APIRouter(prefix="/api/agent", tags=["agent"])


class AnalyzeRequest(BaseModel):
    experiment_ids: List[str]
    metrics: Optional[List[str]] = None
    question: Optional[str] = None
    history: Optional[List[dict]] = None


SYSTEM_PROMPT = """你是 ModuTSC 交通信号控制系统的智能分析助手。你的职责是帮助用户对比分析不同实验的结果。

你需要：
1. 根据提供的实验数据，进行专业、深入的对比分析
2. 指出各实验在不同指标上的优劣
3. 分析可能的原因（如算法差异、参数设置、数据集特征等）
4. 给出改进建议
5. 用中文回答，语言简洁专业

分析时请注意：
- ATT（平均行程时间）越低越好
- AQL（平均排队长度）越低越好
- Throughput（吞吐量）越高越好
- avg_reward 越高越好
- 关注奖励曲线的趋势（是否收敛、是否震荡等）
"""


def _build_experiment_context(exp_ids: List[str], metrics: Optional[List[str]] = None) -> str:
    """从实验详情中构建上下文信息"""
    contexts = []
    for eid in exp_ids:
        detail = load_experiment_detail(eid)
        if not detail:
            contexts.append(f"实验 {eid}: 数据不可用\n")
            continue

        ctx = f"## 实验 {eid}\n"
        ctx += f"- 名称: {detail.get('name', 'N/A')}\n"
        ctx += f"- 数据集: {detail.get('datasetName', 'N/A')}\n"
        ctx += f"- 模型: {detail.get('modelName', 'N/A')}\n"

        # 指标数据
        m = detail.get("metrics", {})
        if m:
            ctx += "- 指标:\n"
            for k, v in m.items():
                if v is not None and v != "n/a":
                    ctx += f"  - {k}: {v}\n"

        # 奖励曲线摘要
        rc = detail.get("reward_curve", [])
        if rc:
            ctx += f"- 奖励曲线: 共 {len(rc)} 个数据点\n"
            if len(rc) > 0:
                ctx += f"  - 起始: {rc[0]:.2f}\n"
                ctx += f"  - 终止: {rc[-1]:.2f}\n"
                ctx += f"  - 最大: {max(rc):.2f}\n"
                ctx += f"  - 最小: {min(rc):.2f}\n"
                # 趋势判断
                if len(rc) >= 3:
                    last_third = rc[len(rc)*2//3:]
                    first_third = rc[:len(rc)//3]
                    avg_last = sum(last_third) / len(last_third)
                    avg_first = sum(first_third) / len(first_third)
                    if avg_last > avg_first * 1.1:
                        ctx += f"  - 趋势: 上升（前1/3均值{avg_first:.2f} → 后1/3均值{avg_last:.2f}）\n"
                    elif avg_last < avg_first * 0.9:
                        ctx += f"  - 趋势: 下降（前1/3均值{avg_first:.2f} → 后1/3均值{avg_last:.2f}）\n"
                    else:
                        ctx += f"  - 趋势: 稳定（前1/3均值{avg_first:.2f} → 后1/3均值{avg_last:.2f}）\n"

        # 配置信息
        config_path = detail.get("config_path")
        if config_path:
            ctx += f"- 配置文件: {config_path}\n"

        contexts.append(ctx)

    return "\n".join(contexts)


@router.post("/analyze")
async def analyze_experiments(req: AnalyzeRequest):
    """使用 LLM 分析实验对比结果"""
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 LLM API Key，请在 app/config.py 中设置 LLM_API_KEY")

    # 构建实验上下文
    context = _build_experiment_context(req.experiment_ids, req.metrics)

    # 构建用户消息
    if req.question:
        user_msg = req.question
    else:
        metric_names = []
        if req.metrics:
            for m_name in req.metrics:
                meta = next((m for m in AVAILABLE_METRICS if m["value"] == m_name), None)
                metric_names.append(meta["label"] if meta else m_name)
        if metric_names:
            user_msg = f"请对比分析以下实验在 [{', '.join(metric_names)}] 指标上的表现：\n\n{context}"
        else:
            user_msg = f"请全面对比分析以下实验的表现：\n\n{context}"

    # 构建消息列表
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": user_msg})

    # 调用 LLM API（流式）
    async def stream_response():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 2000,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        yield json.dumps({"error": f"LLM API 错误: {resp.status_code} - {error_text.decode()}"}, ensure_ascii=False) + "\n"
                        return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield json.dumps({"content": content}, ensure_ascii=False) + "\n"
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError:
            yield json.dumps({"error": "无法连接到 LLM API，请检查网络或 API 地址配置"}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"分析失败: {str(e)}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@router.get("/status")
def agent_status():
    """检查 Agent 是否可用"""
    return {
        "available": bool(LLM_API_KEY),
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
    }
