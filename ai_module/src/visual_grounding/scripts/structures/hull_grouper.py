import cv2
import numpy as np
import math
from typing import List, Tuple, Set
from collections import defaultdict
from ai_module.src.utils.timer import Timer, Stats
from ai_module.src.utils.visualizer import _color_palette
from ai_module.src.visual_grounding.scripts.structures.entity import Entities
from ai_module.src.visual_grounding.scripts.structures.dsu import DSU


NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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

    def fit(self, entities: Entities):
        self._clear_all()

        eids = [eid for eid in entities.ids if eid != -1]
        entities = entities.get(eids)
        for eid in entities.keys():
            self.dsu.add(eid)

        for eid, entity in entities.items():
            pts = np.asarray(entity.points, float)
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
                        if j == eid or j in seen:
                            continue
                        if self.dsu.root(eid) == self.dsu.root(j):
                            continue
                        self._union_with_count(eid, j)
                        self.stats['unions'] += 1
                        seen.add(j)

        return self

    def update(self, entities):
        eids = [eid for eid in entities.ids if eid != -1]
        entities = entities.get(eids)

        for eid, entity in entities.items():
            if eid not in self.dsu.idx:
                self.dsu.add(eid)
                old = set()
            else:
                old = self.ent_cells.get(eid, set())

            new_cells = self._cells_set_from_points(np.asarray(entity.points, float))
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

    def group_hulls(self, use_boundary: bool = True):
        res = []
        cs = float(self.cell_size)
        for gid, eids in enumerate(self.groups()):
            cells = set()
            for eid in eids:
                cells |= self.ent_cells.get(eid, set())
            if use_boundary:
                cells = self.boundary_cells(cells)
            hull_xy = self.hull_from_cells(cells)
            res.append({'gid': gid, 'members': eids, 'hull': hull_xy})
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
