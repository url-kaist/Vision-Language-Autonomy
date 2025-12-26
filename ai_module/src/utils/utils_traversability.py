import numpy as np
from collections import deque


def load_pcd_ascii_with_fields(path: str):
    fields = None
    counts = None
    data_start = None

    with open(path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        elif s.startswith("COUNT"):
            counts = list(map(int, s.split()[1:]))
        elif s.startswith("DATA"):
            # DATA ascii
            data_start = i + 1
            break

    if fields is None or counts is None or data_start is None:
        raise ValueError("PCD header parse failed (FIELDS/COUNT/DATA not found).")

    # Expand multi-count fields into per-component names
    col_names = []
    for name, c in zip(fields, counts):
        if c == 1:
            col_names.append(name)
        else:
            # orientation_xyzw 같은 경우 4개로 확장
            # 관례적으로 _0.. 혹은 x/y/z/w로 쪼갬
            if name.endswith("xyzw") and c == 4:
                base = name.replace("xyzw", "")
                col_names += [base + "x", base + "y", base + "z", base + "w"]
            else:
                col_names += [f"{name}_{k}" for k in range(c)]

    # Load numeric body
    data = np.loadtxt(lines[data_start:], dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != len(col_names):
        raise ValueError(f"Column mismatch: got {data.shape[1]} values/line, expected {len(col_names)}")

    return col_names, data


def filter_disconnected_traversable(
    cols, arr,
    *,
    res=0.10,
    collision_thr=0.7,
    confidence_thr=None,
    use_8n=True,
    seed_xy=(0.0, 0.0),          # 로봇/센서 기준점 (없으면 largest_component=True 권장)
    largest_component=False,     # True면 seed 대신 가장 큰 컴포넌트만 남김
):
    col2i = {c:i for i,c in enumerate(cols)}
    x = arr[:, col2i["x"]]
    y = arr[:, col2i["y"]]

    # 1) traversable 마스크 (원하는 조건으로 확장 가능)
    risk = arr[:, col2i["collision_risk"]]
    ok = np.isfinite(risk) & (risk < collision_thr)

    if confidence_thr is not None and "confidence" in col2i:
        conf = arr[:, col2i["confidence"]]
        ok &= np.isfinite(conf) & (conf >= confidence_thr)

    # traversable 포인트가 없으면 그대로 반환
    idx_ok = np.where(ok)[0]
    if idx_ok.size == 0:
        return arr[:0], np.zeros(arr.shape[0], dtype=bool)

    xt = x[idx_ok]
    yt = y[idx_ok]

    # 2) grid 인덱스
    ix = np.floor(xt / res).astype(np.int32)
    iy = np.floor(yt / res).astype(np.int32)

    # 셀 set + 셀->포인트 인덱스 목록
    cell2pts = {}
    for k, (cx, cy) in enumerate(zip(ix, iy)):
        key = (int(cx), int(cy))
        cell2pts.setdefault(key, []).append(idx_ok[k])

    cells = set(cell2pts.keys())

    # 3) connected components on grid
    if use_8n:
        nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    else:
        nbrs = [(1,0),(-1,0),(0,1),(0,-1)]

    visited = set()
    components = []  # list of list-of-cells

    for c in cells:
        if c in visited:
            continue
        q = deque([c])
        visited.add(c)
        comp = [c]
        while q:
            ux, uy = q.popleft()
            for dx, dy in nbrs:
                v = (ux + dx, uy + dy)
                if v in cells and v not in visited:
                    visited.add(v)
                    q.append(v)
                    comp.append(v)
        components.append(comp)

    # 4) keep component 선택
    if largest_component:
        keep_comp = max(components, key=len)
    else:
        sx, sy = seed_xy
        s_cell = (int(np.floor(sx / res)), int(np.floor(sy / res)))
        # seed가 traversable cell에 없으면 fallback: 가장 큰 컴포넌트
        comp_map = {cell: ci for ci, comp in enumerate(components) for cell in comp}
        if s_cell in comp_map:
            keep_comp = components[comp_map[s_cell]]
        else:
            keep_comp = max(components, key=len)

    keep_cells = set(keep_comp)

    # 5) keep_cells에 속한 포인트만 최종 keep
    keep_mask = np.zeros(arr.shape[0], dtype=bool)
    for cell in keep_cells:
        for pi in cell2pts[cell]:
            keep_mask[pi] = True

    # traversable 조건(ok) 중에서도 연결된 것만 남기기
    final_mask = ok & keep_mask
    return arr[final_mask], final_mask


def save_pcd_ascii_with_header(
    src_pcd_path: str,
    dst_pcd_path: str,
    data: np.ndarray,
):
    """
    src_pcd_path : 원본 pcd (header 재사용)
    dst_pcd_path : 저장할 pcd
    data         : (N, C) float array
    """

    # 원본 헤더 읽기 (DATA 이전까지)
    header_lines = []
    with open(src_pcd_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().startswith("DATA"):
                break

    # DATA ascii 강제 (안전)
    header_lines[-1] = "DATA ascii\n"

    with open(dst_pcd_path, "w") as f:
        # header
        for l in header_lines:
            f.write(l)

        # body
        # np.inf → "inf" 로 ascii 저장됨 (PCD 파서 호환)
        np.savetxt(
            f,
            data,
            fmt="%.6f",
        )
