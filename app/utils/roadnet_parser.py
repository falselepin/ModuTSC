"""
解析 SUMO .net.xml 路网文件，提取节点、边、车道、信号灯数据。
不依赖 TraCI，可离线使用。
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


def parse_roadnet(net_xml_path: str) -> Dict:
    """解析 .net.xml 文件，返回路网可视化所需的全部数据。

    Returns:
        {
            "nodes": [{"id": str, "x": float, "y": float, "type": str}],
            "edges": [{"id": str, "from": str, "to": str,
                       "lanes": [{"id": str, "index": int, "shape": [[x,y],...]}]}],
            "traffic_lights": [{"id": str, "phases": [{"duration": float, "state": str}]}],
            "bounds": {"xMin": float, "yMin": float, "xMax": float, "yMax": float},
        }
    """
    if not os.path.isfile(net_xml_path):
        return _empty_roadnet()

    try:
        tree = ET.parse(net_xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return _empty_roadnet()

    bounds = _parse_bounds(root)
    nodes = _parse_nodes(root)
    edges = _parse_edges(root)
    traffic_lights = _parse_tl_logic(root)

    return {
        "nodes": nodes,
        "edges": edges,
        "traffic_lights": traffic_lights,
        "bounds": bounds,
    }


def _parse_bounds(root) -> Dict:
    """提取路网坐标边界。"""
    loc = root.find("location")
    if loc is not None:
        cb = loc.get("convBoundary", "")
        if cb:
            parts = cb.split(",")
            if len(parts) == 4:
                try:
                    return {
                        "xMin": float(parts[0]), "yMin": float(parts[1]),
                        "xMax": float(parts[2]), "yMax": float(parts[3]),
                    }
                except ValueError:
                    pass
    return {"xMin": 0, "yMin": 0, "xMax": 1000, "yMax": 1000}


def _parse_nodes(root) -> List[Dict]:
    """提取路口/节点信息。"""
    nodes = []
    for j in root.iter("junction"):
        jtype = j.get("type", "")
        # 跳过内部虚拟节点
        if jtype == "internal":
            continue
        jid = j.get("id", "")
        try:
            x = float(j.get("x", "0"))
            y = float(j.get("y", "0"))
        except ValueError:
            continue
        nodes.append({
            "id": jid,
            "x": x,
            "y": y,
            "type": jtype,  # "traffic_light" or "priority"
        })
    return nodes


def _parse_edges(root) -> List[Dict]:
    """提取道路边和车道信息。跳过内部边。"""
    edges = []
    for e in root.iter("edge"):
        func = e.get("function", "")
        if func == "internal":
            continue
        eid = e.get("id", "")
        from_id = e.get("from", "")
        to_id = e.get("to", "")
        lanes = []
        for lane in e.iter("lane"):
            lane_id = lane.get("id", "")
            index = int(lane.get("index", "0"))
            shape_str = lane.get("shape", "")
            shape = _parse_shape(shape_str)
            lanes.append({
                "id": lane_id,
                "index": index,
                "shape": shape,
            })
        if lanes:
            edges.append({
                "id": eid,
                "from": from_id,
                "to": to_id,
                "lanes": lanes,
            })
    return edges


def _parse_shape(shape_str: str) -> List[List[float]]:
    """解析 shape 字符串为 [[x,y], ...] 坐标列表。"""
    points = []
    if not shape_str:
        return points
    for pair in shape_str.split():
        parts = pair.split(",")
        if len(parts) == 2:
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return points


def _parse_tl_logic(root) -> List[Dict]:
    """提取交通信号灯逻辑。"""
    tls = []
    for tl in root.iter("tlLogic"):
        tlid = tl.get("id", "")
        phases = []
        for phase in tl.iter("phase"):
            try:
                duration = float(phase.get("duration", "0"))
            except ValueError:
                duration = 0
            state = phase.get("state", "")
            phases.append({"duration": duration, "state": state})
        tls.append({"id": tlid, "phases": phases})
    return tls


def _empty_roadnet() -> Dict:
    return {
        "nodes": [],
        "edges": [],
        "traffic_lights": [],
        "bounds": {"xMin": 0, "yMin": 0, "xMax": 1000, "yMax": 1000},
    }
