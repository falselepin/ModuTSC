"""
解析 SUMO .rou.xml 流量文件，提取真实车辆出发时间分布。

支持三种格式：
  1. 逐车辆定义：<vehicle id="..." depart="..."/>
  2. 基于 vehsPerHour 的 flow：<flow begin="..." end="..." vehsPerHour="..."/>
  3. 基于 probability 的 flow：<flow begin="..." end="..." probability="..."/>
"""

import os
import xml.etree.ElementTree as ET
from math import ceil
from typing import Dict, List, Tuple


def parse_rou_file(filepath: str) -> Dict:
    """解析单个 .rou.xml 文件，返回车辆出发时间统计。

    Returns:
        {
            "total_vehicles": int,        # 总车辆数（或估算值）
            "time_range": (start, end),   # 仿真时间范围（秒）
            "depart_histogram": [int],    # 按时间窗口统计的车辆出发数
            "window_sec": int,            # 每个时间窗口的秒数
        }
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return _empty_result()

    depart_times = []       # 格式1：直接收集 depart 时间
    flow_specs = []         # 格式2/3：收集 flow 定义

    for elem in root.iter():
        if elem.tag == "vehicle":
            depart = elem.get("depart")
            if depart is not None:
                try:
                    depart_times.append(float(depart))
                except ValueError:
                    pass

        elif elem.tag == "flow":
            begin = elem.get("begin", "0")
            end = elem.get("end", "3600")
            vph = elem.get("vehsPerHour")
            prob = elem.get("probability")
            number = elem.get("number")
            try:
                b, e = float(begin), float(end)
            except ValueError:
                continue
            flow_specs.append({
                "begin": b, "end": e,
                "vehsPerHour": float(vph) if vph else None,
                "probability": float(prob) if prob else None,
                "number": int(number) if number else None,
            })

    # 确定仿真时间范围
    all_times = list(depart_times)
    for fs in flow_specs:
        all_times.append(fs["begin"])
        all_times.append(fs["end"])

    if not all_times:
        return _empty_result()

    t_min = int(min(all_times))
    t_max = int(max(all_times))
    if t_max <= t_min:
        t_max = t_min + 3600

    # 时间窗口大小：将仿真时间分成约 100 个区间
    total_sec = t_max - t_min
    window = max(1, ceil(total_sec / 100))
    n_bins = ceil(total_sec / window)

    # 格式1：直接统计 depart 时间直方图
    histogram = [0] * n_bins
    for t in depart_times:
        idx = int((t - t_min) / window)
        if 0 <= idx < n_bins:
            histogram[idx] += 1

    # 格式2/3：从 flow 定义估算每个时间窗口的车辆数
    for fs in flow_specs:
        fb, fe = fs["begin"], fs["end"]
        if fs["number"] is not None:
            # number 属性：精确车辆数
            count = fs["number"]
            duration = max(fe - fb, 1)
            for i in range(n_bins):
                w_start = t_min + i * window
                w_end = w_start + window
                overlap = max(0, min(w_end, fe) - max(w_start, fb))
                if overlap > 0:
                    histogram[i] += round(count * overlap / duration)
        elif fs["vehsPerHour"] is not None:
            # vehsPerHour：每小时车辆数
            vph = fs["vehsPerHour"]
            for i in range(n_bins):
                w_start = t_min + i * window
                w_end = w_start + window
                overlap = max(0, min(w_end, fe) - max(w_start, fb))
                if overlap > 0:
                    histogram[i] += round(vph * overlap / 3600)
        elif fs["probability"] is not None:
            # probability：每秒生成车辆的概率
            prob = fs["probability"]
            for i in range(n_bins):
                w_start = t_min + i * window
                w_end = w_start + window
                overlap = max(0, min(w_end, fe) - max(w_start, fb))
                if overlap > 0:
                    histogram[i] += round(prob * overlap)

    total_vehicles = sum(histogram)

    return {
        "total_vehicles": total_vehicles,
        "time_range": (t_min, t_max),
        "depart_histogram": histogram,
        "window_sec": window,
    }


def parse_dataset_flows(dataset_dir: str) -> Dict:
    """解析数据集目录下所有 .rou.xml 文件，返回汇总的车辆出发分布。

    Returns:
        {
            "vehicleCurve": [int],       # 100 个数据点，供前端折线图使用
            "total_vehicles": int,       # 总车辆数
            "sim_duration_sec": int,     # 仿真总时长（秒）
            "flow_files": int,           # flow 文件数量
        }
    """
    rou_files = []
    if os.path.isdir(dataset_dir):
        for f in sorted(os.listdir(dataset_dir)):
            if f.endswith(".rou.xml"):
                rou_files.append(os.path.join(dataset_dir, f))

    if not rou_files:
        return {"vehicleCurve": [], "total_vehicles": 0, "sim_duration_sec": 0, "flow_files": 0}

    # 解析所有 flow 文件，合并 depart 时间
    all_departs = []
    all_flows = []

    for filepath in rou_files:
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError:
            continue

        for elem in root.iter():
            if elem.tag == "vehicle":
                depart = elem.get("depart")
                if depart is not None:
                    try:
                        all_departs.append(float(depart))
                    except ValueError:
                        pass
            elif elem.tag == "flow":
                begin = elem.get("begin", "0")
                end = elem.get("end", "3600")
                vph = elem.get("vehsPerHour")
                prob = elem.get("probability")
                number = elem.get("number")
                try:
                    b, e = float(begin), float(end)
                except ValueError:
                    continue
                all_flows.append({
                    "begin": b, "end": e,
                    "vehsPerHour": float(vph) if vph else None,
                    "probability": float(prob) if prob else None,
                    "number": int(number) if number else None,
                })

    # 确定时间范围
    all_times = list(all_departs)
    for fs in all_flows:
        all_times.append(fs["begin"])
        all_times.append(fs["end"])

    if not all_times:
        return {"vehicleCurve": [], "total_vehicles": 0, "sim_duration_sec": 0, "flow_files": len(rou_files)}

    t_min = int(min(all_times))
    t_max = int(max(all_times))
    if t_max <= t_min:
        t_max = t_min + 3600

    total_sec = t_max - t_min
    window = max(1, ceil(total_sec / 100))
    n_bins = ceil(total_sec / window)

    # 统计 vehicle 格式
    histogram = [0] * n_bins
    for t in all_departs:
        idx = int((t - t_min) / window)
        if 0 <= idx < n_bins:
            histogram[idx] += 1

    # 统计 flow 格式
    for fs in all_flows:
        fb, fe = fs["begin"], fs["end"]
        if fs["number"] is not None:
            count = fs["number"]
            duration = max(fe - fb, 1)
            for i in range(n_bins):
                w_start = t_min + i * window
                w_end = w_start + window
                overlap = max(0, min(w_end, fe) - max(w_start, fb))
                if overlap > 0:
                    histogram[i] += round(count * overlap / duration)
        elif fs["vehsPerHour"] is not None:
            vph = fs["vehsPerHour"]
            for i in range(n_bins):
                w_start = t_min + i * window
                w_end = w_start + window
                overlap = max(0, min(w_end, fe) - max(w_start, fb))
                if overlap > 0:
                    histogram[i] += round(vph * overlap / 3600)
        elif fs["probability"] is not None:
            prob = fs["probability"]
            for i in range(n_bins):
                w_start = t_min + i * window
                w_end = w_start + window
                overlap = max(0, min(w_end, fe) - max(w_start, fb))
                if overlap > 0:
                    histogram[i] += round(prob * overlap)

    # 重采样为固定 100 个点，供前端折线图使用
    curve_100 = _resample_to_100(histogram)

    return {
        "vehicleCurve": curve_100,
        "total_vehicles": sum(histogram),
        "sim_duration_sec": total_sec,
        "flow_files": len(rou_files),
    }


def _resample_to_100(data: List[int], target_len: int = 100) -> List[int]:
    """将任意长度的数组重采样为固定 target_len 个点。"""
    if not data:
        return [0] * target_len
    n = len(data)
    if n == target_len:
        return data
    result = []
    for i in range(target_len):
        # 线性插值
        src_idx = i * (n - 1) / (target_len - 1) if target_len > 1 else 0
        lo = int(src_idx)
        hi = min(lo + 1, n - 1)
        frac = src_idx - lo
        val = data[lo] * (1 - frac) + data[hi] * frac
        result.append(max(0, round(val)))
    return result


def _empty_result() -> Dict:
    return {
        "total_vehicles": 0,
        "time_range": (0, 3600),
        "depart_histogram": [],
        "window_sec": 36,
    }
