import cv2
import numpy as np
import math
from typing import List, Tuple, Set
from collections import defaultdict
from ai_module.src.utils.timer import Timer, Stats
from ai_module.src.utils.visualizer import _color_palette
from ai_module.src.visual_grounding.scripts.structures.entity import Entities
from ai_module.src.visual_grounding.scripts.structures.dsu import DSU
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from dataclasses import dataclass
from ai_module.src.visual_grounding.scripts.utils.utils_active_perception import visible_edges_from_pose, visualize_visibility
from ai_module.src.utils.utils_pose import theta_from_agent_pose
from ai_module.src.visual_grounding.scripts.structures.scene_graph import NodeLevel


def _visible_arcs_from_mask(mask: np.ndarray):
    """
    mask: (M,) bool for edges
    return: list of (start_idx, end_idx) inclusive along CW (in index space)
            여기서는 'index 증가 방향'을 CW로 간주해도 됩니다.
            (CCW/CW는 일관되게만 쓰면 OK)
    """
    M = len(mask)
    if M == 0 or not mask.any():
        return []

    # 원형 처리: 시작점을 'False 다음 True' 지점으로 잡아 선형화
    prev = np.roll(mask, 1)
    starts = np.where((~prev) & mask)[0]
    if len(starts) == 0:
        # 전부 True
        return [(0, M - 1)]

    s0 = int(starts[0])
    m2 = np.concatenate([mask[s0:], mask[:s0]])

    arcs = []
    i = 0
    while i < M:
        if not m2[i]:
            i += 1
            continue
        j = i
        while j < M and m2[j]:
            j += 1
        # [i, j-1] is a True run in m2
        a = (s0 + i) % M
        b = (s0 + (j - 1)) % M
        arcs.append((a, b))
        i = j
    return arcs

def _idx_in_arc(i: int, a: int, b: int, M: int) -> bool:
    """index 증가 방향으로 a->b 구간(원형)에 i가 포함되면 True."""
    if a <= b:
        return a <= i <= b
    return (i >= a) or (i <= b)
def _idx_in_cw_arc(i: int, a: int, b: int, M: int) -> bool:
    """
    arc: a -> b 를 index 증가 방향으로 따라갈 때 포함되면 True (원형)
    """
    if a <= b:
        return a <= i <= b
    else:
        # wrap-around
        return (i >= a) or (i <= b)
def _pt_sig(p, ndigits=4):
    return tuple(np.round(np.asarray(p, dtype=np.float32), ndigits))

NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def _edge_signature(p0, p1, ndigits=4):
    a = tuple(np.round(p0, ndigits))
    b = tuple(np.round(p1, ndigits))
    return (a, b) if a <= b else (b, a)

def build_offsets(radius_cells: int, metric: str = "euclid") -> list:
    """
    dilation offsets. metric:
      - "cheby": |dx|<=r, |dy|<=r (정사각형 커널, 빠름/보수적)
      - "manhattan": |dx|+|dy|<=r (다이아몬드)
      - "euclid": dx^2+dy^2<=r^2 (원에 가까움)
    """
    offs = []
    r = int(radius_cells)
    if r <= 0:
        return [(0, 0)]
    if metric == "cheby":
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offs.append((dx, dy))
    elif metric == "manhattan":
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    offs.append((dx, dy))
    else:  # euclid
        r2 = r * r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r2:
                    offs.append((dx, dy))
    return offs


def _xy_to_rc(x: float, y: float, xmin: float, ymax: float, scale: float, pad: int) -> Tuple[int, int]:
    col = int(round((x - xmin) * scale)) + pad
    row = int(round((ymax - y) * scale)) + pad
    return row, col


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points.astype(float), axis=0)
    if len(pts) <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1], dtype=float)


class GridGrouper:
    def __init__(self, threshold: float = 1.0, cell_size: float = None, metric: str = "euclid"):
        self.threshold = float(threshold)
        self.cell_size = float(cell_size) if cell_size else float(threshold / 2)  # 보통 threshold/2 ~ threshold/3 권장
        self.metric = metric
        self.r_cells = int(math.ceil(self.threshold / self.cell_size))
        self.offsets = build_offsets(self.r_cells, metric)

        self.grid = defaultdict(set)  # {(cx, cy): set[eid]}
        self.ent_cells = {}  # {eid: set[(cx,cy)]}
        self.dsu = DSU()

        self.stats = defaultdict(int)
        self.T = Timer()
        self.S = Stats()

        self.group_count = defaultdict(int)  # {root_id: count}
        self.eid_count = defaultdict(int)  # {eid: count}

        # hull cache
        self._group_hulls_cache = None  # 마지막 계산된 group_hulls 결과
        self._hulls_dirty = True  # update()/fit() 후 True

        # visibility cache
        self._visible_edges_cache = {}  # gid -> visible edge mask / segments

    def _clear_all(self):
        self.grid.clear()
        self.ent_cells.clear()
        self.dsu = DSU()

    def _cells_set_from_points(self, points_xy: np.ndarray, stride: int = 2):
        """(N,2/3) points -> (K,2) unique integer cells (cx,cy), vectorized."""
        self.T.tic('points_to_cells')

        cs = self.cell_size
        P = np.asarray(points_xy, float)
        if P.ndim != 2 or P.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64)
        if stride > 1:  # Downsample
            P = P[::stride]

        cells = np.floor(P[:, :2] / cs).astype(np.int64)
        cells = np.unique(cells, axis=0)  # (K, 2)
        self.T.toc()
        return set(map(tuple, cells))

    def _union_with_count(self, a_eid: int, b_eid: int):
        ra = self.dsu.root(a_eid)
        rb = self.dsu.root(b_eid)
        if ra == rb:
            return

        # union 수행
        self.dsu.union_eids(a_eid, b_eid)

        # 새 루트 취득 (a_eid 기준으로 다시 root)
        r_new = self.dsu.root(a_eid)
        # 이전 두 루트의 카운트를 합쳐서 새 루트로 이동
        ca = self.group_count.pop(ra, 0)
        cb = self.group_count.pop(rb, 0)
        self.group_count[r_new] += (ca + cb)

    def fit(self, entities):
        self._clear_all()

        eids = [e['id'][1] for e in entities if 'id' in e]
        for eid in eids:
            self.dsu.add(eid)

        for entity in entities:
            attrs = entity.get("_attrs", {})
            eid = entity['id'][1]

            pts = np.asarray(attrs['points'], float)
            cells = self._cells_set_from_points(pts)
            self.ent_cells[eid] = cells
            for c in cells:
                self.grid[c].add(eid)

        for eid, cells in self.ent_cells.items():
            seen = set()
            for (cx, cy) in cells:
                for (dx, dy) in self.offsets:
                    cc = (cx + dx, cy + dy)
                    for j in self.grid.get(cc, ()):
                        if j == eid or j in seen: # 동일 entity거나 이미 처리했거나
                            continue
                        if self.dsu.root(eid) == self.dsu.root(j): #
                            continue
                        self._union_with_count(eid, j)
                        self.stats['unions'] += 1
                        seen.add(j)

        self._hulls_dirty = True # 다음에 hull은 다시 계산해야 함
        return self

    def update(self, entities):
        for entity in entities:
            eid = entity['id'][1]
            attrs = entity.get("_attrs", {})
            if eid not in self.dsu.idx:
                self.dsu.add(eid)
                old = set()
            else:
                old = self.ent_cells.get(eid, set())

            new_cells = self._cells_set_from_points(np.asarray(attrs['points'], float))
            self.ent_cells[eid] = new_cells

            added = new_cells - old
            self.S.push('added_cells_per_update', len(added))
            removed = old - new_cells  # 줄어드는 경우 지원하려면 grid에서 제거
            self.S.push('kernel_size', len(self.offsets))

            # grid 갱신
            for c in added:
                self.grid[c].add(eid)
            for c in removed:
                s = self.grid.get(c)
                if s:
                    s.discard(eid)
                    if not s:
                        self.grid.pop(c, None)

            self.stats['added_cells'] += len(added)

            self.T.tic('neighbor_probe')
            cand = 0
            for (cx, cy) in added if added else new_cells:
                for (dx, dy) in self.offsets:
                    cand += len(self.grid.get((cx + dx, cy + dy), ()))
            self.T.toc()
            self.S.push("neighbor_candidates", cand)

            # 추가된 셀들만 팽창 후 교차 검사 → DSU union
            seen = set()
            for (cx, cy) in added if added else new_cells:
                for (dx, dy) in self.offsets:
                    cc = (cx + dx, cy + dy)
                    neigh = self.grid.get(cc)
                    if not neigh:
                        continue
                    for j in neigh:
                        if j == eid or j in seen:
                            continue
                        if self.dsu.root(eid) == self.dsu.root(j):
                            continue
                        self._union_with_count(eid, j)
                        self.stats['unions'] += 1
                        seen.add(j)
        self._hulls_dirty = True # 다음에 hull은 다시 계산해야 함

    def groups(self, gid=None) -> List[List[int]]:
        roots = defaultdict(list)
        for eid in self.dsu.ids:
            roots[self.dsu.root(eid)].append(eid)

        if gid is None:
            return list(roots.values())

        group_list = list(roots.values())
        if gid < 0 or gid >= len(group_list):
            return []
        return group_list[gid]

    def hull_from_cells(self, cells: Set[Tuple[int, int]]) -> np.ndarray:
        """
        cells: {(cx,cy), ...} in grid index
        return: hull in (x,y) world coords
        """
        if not cells:
            return np.empty((0, 2), float)
        A = np.asarray(list(cells), dtype=float)  # (N,2) [cx, cy]
        cs = self.cell_size
        # 각 셀의 4 모서리 (벡터화) → (4N,2) world 좌표
        corners = np.vstack((A, A + [1, 0], A + [0, 1], A + [1, 1])) * cs
        corners = np.unique(corners, axis=0)  # 중복 제거
        return _convex_hull_xy(corners)

    @staticmethod
    def boundary_cells(cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        b = set()
        for cx, cy in cells:
            for dx, dy in NEI8:
                if (cx + dx, cy + dy) not in cells:
                    b.add((cx, cy))
                    break
        return b

    def update_visibility(self, agent_pose, sg=None, max_range=None):
        print(f"<GridGrouper/update_visibility.1>")
        hulls = self.group_hulls()
        print(f"<GridGrouper/update_visibility.2>")
        # self._visible_edges_cache.clear()
        fov_rad = sg.fov_x

        print(f"<GridGrouper/update_visibility.3>")
        if len(hulls) > 0 and isinstance(agent_pose, dict):
            position = agent_pose['position']
            orientation = agent_pose['orientation']
            theta = theta_from_agent_pose(orientation)
            agent_pose = np.array([position[0], position[1], theta], dtype=np.float32)
        print(f"<GridGrouper/update_visibility.4>")

        if sg is not None and len(hulls) > 0:
            print(f"<GridGrouper/update_visibility.5>")
            agent_poses = []
            for (elevel, eid), entity in sg.G.nodes(data=True):
                if elevel == str(NodeLevel.KEYFRAME):
                    # print(f"{eid}")
                    attrs = entity.get("_attrs", {})
                    T = np.asarray(attrs["pose"], dtype=np.float32)
                    x, y = T[0, 3], T[1, 3]
                    theta = np.arctan2(T[1, 0], T[0, 0])
                    agent_poses.append(np.array([x, y, theta], dtype=np.float32))
            agent_poses.append(agent_pose)
        print(f"<GridGrouper/update_visibility.6>")

        for item in hulls:
            print(f"<GridGrouper/update_visibility.7>")
            gid = item['gid']
            hull = np.asarray(item['hull'], np.float32)
            M = len(hull)
            if hull is None or M < 2:
                item['edge_visible'] = np.zeros(0, dtype=bool)
                item['edge_segs'] = None
                continue

            new_sigs = [
                _edge_signature(hull[i], hull[(i + 1) % M])
                for i in range(M)
            ]

            old_sigs = item.get('edge_sigs', [])
            old_vis = item.get('edge_visible', np.zeros(0, bool))
            old_segs = item.get('edge_segs', np.empty((0, 2, 2), np.float32))

            old_sig_to_idx = {s: i for i, s in enumerate(old_sigs)}

            # --- 현재 프레임 visibility 계산 (전체 hull에 대해 한 번만) ---
            vis_all = np.zeros(M, dtype=bool)
            segs_all = []  # 디버깅/시각화용 (원하면)
            for eid, pose in enumerate(agent_poses):
                vis, segs, _ = visible_edges_from_pose(
                    hull,
                    pose,
                    fov_rad=fov_rad,
                    max_range=max_range,
                )
                vis_all |= vis
                if segs is not None:
                    segs_all.append(segs)
                # visualize_visibility(hull, pose, segs, fov_rad, max_range, save_path=f'/ws/external/vis/visibility_debug{eid}.jpg')

            # --- 새 edge_visible ---
            new_edge_visible = np.zeros(M, dtype=bool)
            new_edge_segs = []

            # ---- 1) 변하지 않은 edge는 그대로 복사 ----
            for i, sig in enumerate(new_sigs):
                cur_vis = vis_all[i]

                if sig in old_sig_to_idx:
                    j = old_sig_to_idx[sig]
                    new_edge_visible[i] = old_vis[j] or cur_vis # OR 누적
                else: # 새 edge
                    new_edge_visible[i] = cur_vis

                if new_edge_visible[i]:
                    p0, p1 = hull[i], hull[(i+1) % M]
                    new_edge_segs.append([p0, p1])

            # ---- cache 업데이트 ----
            # self._visible_edges_cache[gid] = {'vis': vis, 'segs': segs}
            item['edge_visible'] = new_edge_visible
            item['edge_segs'] = np.asarray(new_edge_segs, dtype=np.float32)
            item['edge_sigs'] = new_sigs
            print(f"<GridGrouper/update_visibility.8>")

    def group_hulls(self, use_boundary: bool = True):
        print(f"<GridGrouper/group_hulls.1>")
        if not self._hulls_dirty and self._group_hulls_cache is not None:
            print(f"<GridGrouper/group_hulls.1.2>")
            return self._group_hulls_cache

        print(f"<GridGrouper/group_hulls.2>")
        # ---- old cache snapshot (for carry-over) ----
        old_cache = self._group_hulls_cache or []
        old_by_gid = {it["gid"]: it for it in old_cache}
        print(f"<GridGrouper/group_hulls.3>")

        res = []
        for gid, eids in enumerate(self.groups()):
            print(f"<GridGrouper/group_hulls.3.0>")
            cells = set()
            for eid in eids:
                cells |= self.ent_cells.get(eid, set())
            if use_boundary:
                cells = self.boundary_cells(cells)
            hull_xy = self.hull_from_cells(cells)
            M = len(hull_xy)
            print(f"<GridGrouper/group_hulls.3.1>")

            # new signatures
            new_sigs = []
            for i in range(M):
                p0 = hull_xy[i]
                p1 = hull_xy[(i + 1) % M]
                new_sigs.append(_edge_signature(p0, p1))
            print(f"<GridGrouper/group_hulls.3.2>")

            # ---- default (fresh) ----
            new_edge_visible = np.zeros(M, dtype=bool)
            new_edge_segs = []

            # ---- carry over from old if possible ----
            print(f"<GridGrouper/group_hulls.3.3>")
            old_item = old_by_gid.get(gid)
            if old_item is not None:
                old_sigs = old_item.get("edge_sigs", [])
                old_vis = old_item.get("edge_visible", np.zeros(0, dtype=bool))

                # old visible edges as a set of signatures (so we can rebuild segs robustly)
                old_visible_sig_set = set()
                if len(old_sigs) == len(old_vis):
                    old_visible_sig_set = {s for s, v in zip(old_sigs, old_vis) if v}

                old_sig_to_idx = {s: i for i, s in enumerate(old_sigs)}

                # --- Ver 1 ---
                # for i, sig in enumerate(new_sigs):
                #     # visibility carry-over for unchanged edges
                #     j = old_sig_to_idx.get(sig)
                #     if j is not None and j < len(old_vis):
                #         new_edge_visible[i] = bool(old_vis[j])
                #
                #     # segs carry-over by visibility (do NOT rely on old_item['edge_segs'] geometry)
                #     if sig in old_visible_sig_set:
                #         p0, p1 = hull_xy[i], hull_xy[(i + 1) % M]
                #         new_edge_segs.append([p0, p1])
                # --- Ver 2 ---
                # 2) 이전 visible edge들의 vertex(끝점) sig 집합
                # old_visible_vertex_set = set()
                # for s in old_visible_sig_set:
                #     a, b = s  # each is point signature tuple
                #     old_visible_vertex_set.add(a)
                #     old_visible_vertex_set.add(b)
                # for i, sig in enumerate(new_sigs):
                #     # (A) 완전히 동일 edge면 기존 visibility 유지
                #     j = old_sig_to_idx.get(sig)
                #     if j is not None and j < len(old_vis):
                #         new_edge_visible[i] = bool(old_vis[j])
                #
                #     # (B) 동일 edge가 아닌데, "예전에 보였던 면이 변형된 것"이면 그냥 True 처리
                #     if j is None:
                #         p0_sig = _pt_sig(hull_xy[i])
                #         p1_sig = _pt_sig(hull_xy[(i + 1) % M])
                #         if (p0_sig in old_visible_vertex_set) or (p1_sig in old_visible_vertex_set):
                #             new_edge_visible[i] = True
                #
                #     if new_edge_visible[i]:
                #         p0, p1 = hull_xy[i], hull_xy[(i + 1) % M]
                #         new_edge_segs.append([p0, p1])
                # --- Ver 3: visible 구간(arc) 기반 carry-over ---
                old_arcs = []
                if len(old_sigs) == len(old_vis) and len(old_vis) > 0:
                    old_arcs = _visible_arcs_from_mask(old_vis)

                for i, sig in enumerate(new_sigs):
                    j = old_sig_to_idx.get(sig)

                    # (A) 동일 edge면 기존 visibility 유지
                    if j is not None and j < len(old_vis):
                        new_edge_visible[i] = bool(old_vis[j])
                    else:
                        # (B) edge가 변했으면: "예전에 보였던 구간 사이"면 그냥 True
                        for (a, b) in old_arcs:
                            if _idx_in_arc(i, a, b, M):
                                new_edge_visible[i] = True
                                break

                    if new_edge_visible[i]:
                        p0, p1 = hull_xy[i], hull_xy[(i + 1) % M]
                        new_edge_segs.append([p0, p1])
            print(f"<GridGrouper/group_hulls.3.4>")

            res.append({
                "gid": gid,
                "members": eids,
                "hull": hull_xy,
                "edge_visible": new_edge_visible,
                "edge_segs": np.asarray(new_edge_segs, dtype=np.float32) if new_edge_segs else None,
                "edge_sigs": new_sigs,
            })
            print(f"<GridGrouper/group_hulls.3.5>")
        print(f"<GridGrouper/group_hulls.4>")

        self._group_hulls_cache = res
        self._hulls_dirty = False
        print(f"<GridGrouper/group_hulls.5>")

        return res

    def mark_group_processed_by_eid(self, eid: int, inc: int = 1):
        r = self.dsu.root(eid)
        self.group_count[r] += inc

    def mark_group_processed_by_gid(self, gid: int, inc: int = 1):
        gs = self.groups()
        if gid < 0 or gid >= len(gs) or not gs[gid]:
            return
        any_member = gs[gid][0]
        self.mark_group_processed_by_eid(any_member, inc=inc)

    def mark_eids_processed(self, eids: int, inc: int = 1):
        """해당 eid가 처리되었다고 표시(inc만큼 증가)."""
        if not isinstance(eids, list):
            eids = [eids]
        for eid in eids:
            if eid in self.dsu.idx:  # 존재하는 eid만 기록하고 싶다면 체크
                self.eid_count[eid] += int(inc)

    def get_low_count_eids(self, max_count: int, only_present: bool = True):
        """
        max_count 이하인 eid만 리스트로 반환.
        only_present=True면 현재 DSU/ent_cells에 존재하는 eid만 반환.
        """
        if only_present:
            iterable = self.dsu.ids  # 또는 self.ent_cells.keys()
        else:
            # eid_count에 기록은 있으나 현재 없을 수도 있음
            iterable = set(self.dsu.ids) | set(self.eid_count.keys())

        return [eid for eid in iterable if self.eid_count.get(eid, 0) <= max_count]

    @classmethod
    def _draw_cell_rect_range(cls, overlay, cx0, cx1, cy, cs, meta, color):
        # (cx0..cx1, cy) 한 줄을 한 방에 사각형으로
        xmin, ymax = float(meta["xmin"]), float(meta["ymax"])
        scale = float(meta["scale"])
        pad = int(meta.get("pad", 0))
        H, W = overlay.shape[:2]

        # world → pixel
        x0, x1 = cx0 * cs, (cx1 + 1) * cs
        y0, y1 = cy * cs, (cy + 1) * cs
        r_top, c_left = _xy_to_rc(x0, y1, xmin, ymax, scale, pad)
        r_bot, c_right = _xy_to_rc(x1, y0, xmin, ymax, scale, pad)
        r0, r1 = sorted((r_top, r_bot))
        c0, c1 = sorted((c_left, c_right))
        if r1 < 0 or r0 >= H or c1 < 0 or c0 >= W:
            return
        r0, r1 = max(0, r0), min(H - 1, r1)
        c0, c1 = max(0, c0), min(W - 1, c1)
        cv2.rectangle(overlay, (c0, r0), (c1, r1), color, -1)

    @classmethod
    def draw_cells_runs(cls, overlay, cells, cs, meta, color):
        # (cx,cy) -> cy 기준 그룹
        rows = defaultdict(list)
        for (cx, cy) in cells:
            rows[cy].append(cx)
        for cy, xs in rows.items():
            xs = sorted(xs)
            # 연속 구간으로 묶기
            runs = []
            s, p = xs[0], xs[0]
            for x in xs[1:]:
                if x == p + 1:
                    p = x
                else:
                    runs.append((s, p))
                    s = p = x
            runs.append((s, p))
            # 각 run을 한 번에 그림
            for (x0, x1) in runs:
                cls._draw_cell_rect_range(overlay, x0, x1, cy, cs, meta, color)

    def visualize(self, image, meta, alpha=0.5, out_path="/ws/external/vis/grid_grouper.jpg"):
        cs = float(self.cell_size)
        paint = image.copy()
        colors = _color_palette(max(1, len(self.dsu.ids)))

        # 그룹별 셀 합집합 생성
        # groups()는 [[eid,...], ...] 리턴
        groups = self.groups()
        for gid, eids in enumerate(groups):
            color = colors[gid % len(colors)]
            # 그룹 내 모든 엔티티의 셀을 합집합
            union_cells = set()
            for eid in eids:
                union_cells |= self.ent_cells.get(eid, set())

            # 각 셀을 사각형으로 칠함
            self.T.tic("draw_cells_run")
            self.draw_cells_runs(paint, union_cells, cs, meta, color)
            self.T.toc()
            self.S.push("cell_drawn", len(union_cells))

        # 투명 합성
        vis = cv2.addWeighted(paint, alpha, image, 1.0 - alpha, 0)

        if out_path:
            cv2.imwrite(out_path, vis)
        return vis

    def visualize_grid(self, save_path=None):
        cell_dict = self.grid
        cell_size = 1.0
        id_to_points = defaultdict(list)

        for (x, y), id_set in cell_dict.items():
            for obj_id in id_set:
                id_to_points[obj_id].append((x, y))

        # ---- discrete colormap for object ids ----
        obj_ids = sorted(id_to_points.keys())
        cmap = cm.get_cmap("tab10", len(obj_ids))  # categorical
        id_to_color = {
            obj_id: cmap(i) for i, obj_id in enumerate(obj_ids)
        }

        fig, ax = plt.subplots(figsize=(6, 6))

        for obj_id, pts in id_to_points.items():
            color = id_to_color[obj_id]
            for (x, y) in pts:
                # grid cell
                rect = patches.Rectangle(
                    (x, y),
                    cell_size, cell_size,
                    facecolor=color, edgecolor='black',
                    linewidth=0.5, alpha=0.6
                )
                ax.add_patch(rect)

                # object id text
                ax.text(
                    x + cell_size / 2,
                    y + cell_size / 2,
                    str(obj_id),
                    ha="center", va="center",
                    fontsize=9, color='black',
                )

        # ---- grid 느낌을 살리는 설정 ----
        ax.set_aspect("equal")

        xs = [x for (x, y) in cell_dict.keys()]
        ys = [y for (x, y) in cell_dict.keys()]

        ax.set_xlim(min(xs) - 1, max(xs) + cell_size + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + cell_size + 1)

        ax.set_xticks(range(int(min(xs)), int(max(xs)) + 2))
        ax.set_yticks(range(int(min(ys)), int(max(ys)) + 2))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Grid-style visualization grouped by object id")

        legend_patches = [
            patches.Patch(color=id_to_color[obj_id], label=f"id={obj_id}")
            for obj_id in obj_ids
        ]
        ax.legend(handles=legend_patches, loc="upper right")

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()

    def visualize_group_hulls(self, save_path=None, padding_ratio=0.2):
        cluster_infos = self.group_hulls()

        # ---- gid → color ----
        gids = sorted({c["gid"] for c in cluster_infos})
        cmap = cm.get_cmap("tab10", len(gids))
        gid_to_color = {gid: cmap(i) for i, gid in enumerate(gids)}

        fig, ax = plt.subplots(figsize=(6, 6))

        # ---- 모든 hull 좌표 수집 ----
        all_pts = []

        for c in cluster_infos:
            hull = np.asarray(c["hull"], dtype=np.float32)
            all_pts.append(hull)

            poly = patches.Polygon(
                hull,
                closed=True,
                facecolor=gid_to_color[c["gid"]],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.6,
                label=f"gid={c['gid']}"
            )
            ax.add_patch(poly)

            # centroid label
            cx, cy = hull.mean(axis=0)
            ax.text(cx, cy, f"gid={c['gid']}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold")

        all_pts = np.vstack(all_pts)  # (N, 2)

        # ---- axis range with padding (핵심) ----
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)

        dx = xmax - xmin
        dy = ymax - ymin

        # hull이 거의 점인 경우 대비
        dx = max(dx, 1e-3)
        dy = max(dy, 1e-3)

        pad_x = dx * padding_ratio
        pad_y = dy * padding_ratio

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        # ---- styling ----
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Convex hull visualization (auto-scaled)")
        ax.grid(True, linestyle="--", linewidth=0.5)

        # ---- legend dedup ----
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()

    def visualize_grid_with_hulls(self, save_path=None):
        cell_dict = self.grid
        clusters = self.group_hulls()
        cell_size = self.cell_size

        fig, ax = plt.subplots(figsize=(7, 7))

        # 1) grid cells (배경)
        id_to_points = defaultdict(list)
        for (x, y), ids in cell_dict.items():
            for obj_id in ids:
                id_to_points[obj_id].append((x * cell_size, y * cell_size))

        obj_ids = sorted(id_to_points.keys())
        cmap_id = cm.get_cmap("tab10", max(len(obj_ids), 1))
        id_color = {oid: cmap_id(i) for i, oid in enumerate(obj_ids)}

        for oid, pts in id_to_points.items():
            for (x, y) in pts:
                ax.add_patch(
                    patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=id_color[oid],
                        edgecolor="black",
                        linewidth=0.3,
                        alpha=0.25,
                    )
                )

        # 2) hull polygons (전경)
        gids = sorted({c["gid"] for c in clusters})
        cmap_gid = cm.get_cmap("tab10", max(len(gids), 1))
        gid_color = {gid: cmap_gid(i) for i, gid in enumerate(gids)}

        for c in clusters:
            hull = np.asarray(c["hull"], dtype=np.float32)
            if hull.size == 0:
                continue
            ax.add_patch(
                patches.Polygon(
                    hull, closed=True,
                    facecolor=gid_color[c["gid"]],
                    edgecolor="black",
                    linewidth=2.0,
                    alpha=0.45,
                )
            )

        # 3) 축 자동 스케일 (grid+hull 포함), grid 느낌
        ax.set_aspect("equal")
        ax.relim()
        ax.autoscale_view()
        ax.grid(True, linestyle="--", linewidth=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Grid + Group Hulls (overlay)")

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()

