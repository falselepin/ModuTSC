# 提取道路网络信息
def extract_road_network(env):
    try:
        import traci
        nodes = []
        edges = []
        for jid in env.ids():
            pos = traci.junction.getPosition(jid)
            nodes.append({"id": jid, "x": pos[0], "y": pos[1]})
        for edge_id in env.all_edge_ids():
            n_lanes = traci.edge.getLaneNumber(edge_id)
            for lane_idx in range(n_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                shape = traci.lane.getShape(lane_id)
                edges.append({"lane_id": lane_id, "points": shape})
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"Error extracting road network: {e}")
        return {"nodes": [], "edges": []}