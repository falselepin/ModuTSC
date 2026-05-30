from typing import List, Optional
import numpy as np
from modutsc.env import Env
from modutsc.plugins.observers import Observer
from modutsc.scheduling.registry import register


try:
    import dgl
    _DGL_GRAPH_TYPE = dgl.heterograph.DGLHeteroGraph
except ImportError:
    import numpy as np
    _DGL_GRAPH_TYPE = np.ndarray


@register("observer", "igrl_graph")
class IgrlGraphObserver(Observer):
    __output_type__ = _DGL_GRAPH_TYPE

    def setup(self, cfg: dict) -> None:
        self._device = cfg.get("device", "cpu")
        self._node_state_dim = cfg.get("node_state_dim", 0)
        self._tsc_ids: List[str] = []
        self._tsc_to_idx: dict = {}
        self._lane_to_idx: dict = {}
        self._conn_to_idx: dict = {}
        self._conn_info: dict = {}
        self._graph_built = False
        self._dgl_graph = None
        self._have_dgl = False
        try:
            import dgl
            self._have_dgl = True
        except ImportError:
            self._have_dgl = False
        self._last_phase_time: dict = {}
        self._features = cfg.get("features", ["num", "waiting", "wait_time", "speed"])
        self._cached_dim = 0

    def _build_graph_static(self, tsc_ids: List[str]):
        """Build DGL heterograph topology from Env（受控连接拓扑）。"""
        import torch

        if self._env is None:
            raise RuntimeError(
                "IgrlGraphObserver: 构建时须在 setup(cfg, env=...) 传入已 launch 的 Env"
            )

        self._tsc_ids = tsc_ids
        n_tsc = len(tsc_ids)
        self._tsc_to_idx = {jid: i for i, jid in enumerate(tsc_ids)}

        all_links: List[dict] = []
        all_lanes: List[str] = []
        all_conn_ids: List[str] = []

        for jid in tsc_ids:
            controlled_links = self._env.traffic_light_controlled_links(jid)
            if not controlled_links:
                continue

            for grp_idx, group in enumerate(controlled_links):
                for link in group:
                    in_lane = link[0]
                    out_lane = link[1]
                    conn_id = f"{jid}_{in_lane}_{out_lane}"
                    all_links.append({
                        "conn_id": conn_id, "jid": jid, "grp_idx": grp_idx,
                        "in_lane": in_lane, "out_lane": out_lane,
                    })
                    all_lanes.append(in_lane)
                    all_lanes.append(out_lane)
                    all_conn_ids.append(conn_id)

        all_lanes = sorted(set(all_lanes))
        all_conn_ids = sorted(set(all_conn_ids))

        n_lanes = len(all_lanes)
        n_conns = len(all_conn_ids)

        self._lane_to_idx = {lid: i for i, lid in enumerate(all_lanes)}
        self._conn_to_idx = {cid: i for i, cid in enumerate(all_conn_ids)}
        self._conn_info = {cid: lk for lk in all_links}

        tsc_to_conn_src, tsc_to_conn_dst = [], []
        conn_to_tsc_src, conn_to_tsc_dst = [], []
        conn_to_inlane_src, conn_to_inlane_dst = [], []
        inlane_to_conn_src, inlane_to_conn_dst = [], []
        conn_to_outlane_src, conn_to_outlane_dst = [], []
        outlane_to_conn_src, outlane_to_conn_dst = [], []

        for lk in all_links:
            cid = lk["conn_id"]
            jid = lk["jid"]
            in_l = lk["in_lane"]
            out_l = lk["out_lane"]

            if cid not in self._conn_to_idx:
                continue
            ci = self._conn_to_idx[cid]
            ti = self._tsc_to_idx[jid]

            tsc_to_conn_src.append(ti); tsc_to_conn_dst.append(ci)
            conn_to_tsc_src.append(ci); conn_to_tsc_dst.append(ti)

            if in_l in self._lane_to_idx:
                li = self._lane_to_idx[in_l]
                conn_to_inlane_src.append(ci); conn_to_inlane_dst.append(li)
                inlane_to_conn_src.append(li); inlane_to_conn_dst.append(ci)
            if out_l in self._lane_to_idx:
                lo = self._lane_to_idx[out_l]
                conn_to_outlane_src.append(ci); conn_to_outlane_dst.append(lo)
                outlane_to_conn_src.append(lo); outlane_to_conn_dst.append(ci)

        data_dict = {}
        if tsc_to_conn_src:
            data_dict[('tsc', 'to', 'connection')] = \
                (torch.tensor(tsc_to_conn_src), torch.tensor(tsc_to_conn_dst))
            data_dict[('connection', 'to', 'tsc')] = \
                (torch.tensor(conn_to_tsc_src), torch.tensor(conn_to_tsc_dst))
            data_dict[('connection', 'to', 'lane_in')] = \
                (torch.tensor(conn_to_inlane_src), torch.tensor(conn_to_inlane_dst))
            data_dict[('lane', 'to', 'connection_in')] = \
                (torch.tensor(inlane_to_conn_src), torch.tensor(inlane_to_conn_dst))
        if conn_to_outlane_src:
            data_dict[('connection', 'to', 'lane_out')] = \
                (torch.tensor(conn_to_outlane_src), torch.tensor(conn_to_outlane_dst))
            data_dict[('lane', 'to', 'connection_out')] = \
                (torch.tensor(outlane_to_conn_src), torch.tensor(outlane_to_conn_dst))

        self._n_tsc = n_tsc
        self._n_lanes = n_lanes
        self._n_conns = n_conns

        self._node_state_dim = self._node_state_dim or max(
            self._compute_tsc_feat_dim(),
            self._compute_lane_feat_dim(),
            self._compute_conn_feat_dim(),
        ) or 10

        num_nodes_dict = {'tsc': n_tsc, 'lane': n_lanes, 'connection': n_conns}

        self._dgl_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        self._dgl_graph.nodes['tsc'].data['feat'] = \
            torch.zeros(n_tsc, self._node_state_dim)
        self._dgl_graph.nodes['lane'].data['feat'] = \
            torch.zeros(n_lanes, self._node_state_dim)
        self._dgl_graph.nodes['connection'].data['feat'] = \
            torch.zeros(n_conns, self._node_state_dim)

        self._graph_built = True
        self._cached_dim = max(1, n_tsc * 4)

    def _compute_lane_feat_dim(self) -> int:
        return 4

    def _compute_tsc_feat_dim(self) -> int:
        return 2

    def _compute_conn_feat_dim(self) -> int:
        return 4

    # ===================================================================
    #  Observer API
    # ===================================================================

    def observe(self, env) -> List[dict]:
        if not self._graph_built:
            try:
                import dgl
                self._have_dgl = True
                self._build_graph_static(env.ids())
            except ImportError:
                self._have_dgl = False

        if self._have_dgl and self._dgl_graph is not None:
            return self._observe_graph(env)
        else:
            return self._observe_vector(env)

    def dim(self) -> int:
        if not self._have_dgl and self._cached_dim == 0:
            return 48
        return self._cached_dim if self._cached_dim > 0 else 48

    def reset(self) -> None:
        self._last_phase_time.clear()

    # ===================================================================
    #  DGL graph observation (full IGRLGraphBuilder.update_features)
    # ===================================================================

    def _observe_graph(self, env) -> List[dict]:
        import torch

        if self._env is None:
            raise RuntimeError(
                "IgrlGraphObserver: 构建时须在 setup(cfg, env=...) 传入已 launch 的 Env"
            )

        t = self._env.time()
        for jid in self._tsc_ids:
            if jid not in self._tsc_to_idx:
                continue
            ti = self._tsc_to_idx[jid]
            try:
                phase_state = self._env.traffic_light_state_string(jid)
                is_yellow = 1.0 if 'y' in phase_state.lower() else 0.0
            except Exception:
                is_yellow = 0.0

            prev_t = self._last_phase_time.get(jid, t)
            self._last_phase_time[jid] = t
            tsla = min((t - prev_t) / 120.0, 1.0)

            feat = torch.zeros(self._node_state_dim)
            feat[0] = is_yellow
            feat[1] = tsla
            self._dgl_graph.nodes['tsc'].data['feat'][ti] = feat.to(
                self._dgl_graph.nodes['tsc'].data['feat'].device)

        for lid, li in self._lane_to_idx.items():
            feat = torch.zeros(self._node_state_dim)
            try:
                feat[0] = self._env.lane_length(lid) / 200.0
                feat[1] = self._env.lane_vehicle_count(lid) / 20.0
                vids = self._env.lane_vehicle_ids(lid)
                speeds = [self._env.vehicle_speed(v) for v in vids]
                feat[2] = (np.mean(speeds) if speeds else 0.0) / 15.0
                feat[3] = self._env.lane_halting_count(lid) / 20.0
            except Exception:
                pass
            self._dgl_graph.nodes['lane'].data['feat'][li] = feat.to(
                self._dgl_graph.nodes['lane'].data['feat'].device)

        for cid, ci in self._conn_to_idx.items():
            info = self._conn_info.get(cid)
            feat = torch.zeros(self._node_state_dim)
            if info is None:
                self._dgl_graph.nodes['connection'].data['feat'][ci] = feat
                continue
            jid = info["jid"]
            grp_idx = info["grp_idx"]
            try:
                phase_state = self._env.traffic_light_state_string(jid)
                if grp_idx < len(phase_state):
                    ch = phase_state[grp_idx]
                    feat[0] = 1.0 if ch in 'Gg' else 0.0
                    feat[2] = 1.0 if ch == 'G' else 0.0
            except Exception:
                pass
            self._dgl_graph.nodes['connection'].data['feat'][ci] = feat.to(
                self._dgl_graph.nodes['connection'].data['feat'].device)

        self._dgl_graph = self._dgl_graph.to(self._device)

        results = []
        for jid in self._tsc_ids:
            tsc_idx = self._tsc_to_idx.get(jid, 0)
            results.append({"id": jid,
                            "features": self._dgl_graph,
                            "extras": {"tsc_idx": tsc_idx, "n_tsc": self._n_tsc}})
        return results

    # ===================================================================
    #  Vector fallback (no DGL)
    # ===================================================================

    def _observe_vector(self, env) -> List[dict]:
        results = []
        lane_states = env.all_incoming_lane_states()
        for jid in env.ids():
            feats = []
            for lid, lane in sorted(lane_states.get(jid, {}).items()):
                for fn in self._features:
                    feats.append(float(lane.get(fn, 0.0)))
            results.append(feats)

        max_len = max((len(f) for f in results), default=1)
        self._cached_dim = max(self._cached_dim, max_len)

        obs_list = []
        jids = env.ids()
        for i, feats in enumerate(results):
            while len(feats) < self._cached_dim:
                feats.append(0.0)
            obs_list.append({"id": jids[i], "features": np.array(feats, dtype=np.float32)})
        return obs_list
