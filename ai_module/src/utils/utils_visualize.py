import numpy as np


def _polyline_xy_to_ribbon_mesh3d(path_xy: np.ndarray, z: float, width: float, closed: bool = False):
    """
    path_xy: (N,2) float
    z: plane height
    width: ribbon width in world units
    closed: if True, connect last->first
    returns: (verts3, tris) or (None, None) if not enough points
    """
    pts2 = np.asarray(path_xy, dtype=np.float32).reshape(-1, 2)
    n = pts2.shape[0]
    if n < 2:
        return None, None

    p0 = pts2
    p1 = np.roll(pts2, -1, axis=0)
    m = n if closed else (n - 1)
    p0 = p0[:m]
    p1 = p1[:m]

    d = p1 - p0
    lens = np.linalg.norm(d, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-8)
    t = d / lens  # tangent
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)  # in-plane normal

    off = (width * 0.5) * nrm

    v0 = p0 + off
    v1 = p0 - off
    v2 = p1 - off
    v3 = p1 + off

    verts2 = np.stack([v0, v1, v2, v3], axis=1).reshape(-1, 2)  # (4m,2)
    verts3 = np.column_stack([verts2[:, 0], verts2[:, 1], np.full((verts2.shape[0],), z, np.float32)])  # (4m,3)

    base = (np.arange(m, dtype=np.int32) * 4)[:, None]
    tris = np.concatenate(
        [
            np.hstack([base + 0, base + 1, base + 2]),
            np.hstack([base + 0, base + 2, base + 3]),
        ],
        axis=0,
    ).astype(np.int32)  # (2m,3)

    return verts3, tris

def build_ribbon_mesh(path_xy: np.ndarray, z: float, width: float, rgb_u8):
    pts = path_xy.astype(np.float32)  # (N,2)
    N = pts.shape[0]

    # ====== 1) per-segment tangent/normal 계산 ======
    seg = pts[1:] - pts[:-1]                          # (N-1,2)
    seg_len = np.linalg.norm(seg, axis=1, keepdims=True)
    seg_len = np.maximum(seg_len, 1e-8)
    t = seg / seg_len                                  # (N-1,2) tangent
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)           # (N-1,2) in-plane normal (left)

    # ====== 2) per-vertex miter (join) 계산 ======
    # 각 vertex i에서 좌/우 세그먼트의 normal을 평균내서 miter 방향을 만들고,
    # miter 길이를 dot(miter, n_prev)로 보정 -> 코너 틈 방지
    miter = np.zeros((N, 2), np.float32)
    miter_len = np.ones((N, 1), np.float32)

    # endpoints: 그냥 인접 세그먼트 normal 사용
    miter[0] = n[0]
    miter[-1] = n[-1]

    # interior
    for i in range(1, N - 1):
        n0 = n[i - 1]
        n1 = n[i]
        m = n0 + n1
        m_norm = np.linalg.norm(m)
        if m_norm < 1e-6:
            # 180도에 가까운 꺾임(직선/반전) -> bevel처럼 처리
            miter[i] = n1
            miter_len[i] = 1.0
        else:
            m = m / m_norm
            miter[i] = m
            # miter 길이 스케일: (w/2) / dot(miter, n0)
            d = float(np.dot(m, n0))
            if abs(d) < 1e-6:
                miter_len[i] = 1.0
            else:
                miter_len[i] = 1.0 / d  # 폭 적용 전에 스케일만

    # ====== 3) miter limit (폭이 클 때 과도하게 튀어나오는 spike 방지) ======
    # radius가 클수록 코너에서 miter 길이가 커져 spike가 생길 수 있어서 제한합니다.
    MITER_LIMIT = 4.0  # 필요시 2~6 사이로 조정
    miter_len = np.clip(miter_len, -MITER_LIMIT, MITER_LIMIT).astype(np.float32)

    half = (width * 0.5)
    off = (miter * (miter_len * half)).astype(np.float32)  # (N,2)

    left2 = pts + off
    right2 = pts - off

    left3 = np.column_stack([left2[:, 0], left2[:, 1], np.full((N,), z, np.float32)])
    right3 = np.column_stack([right2[:, 0], right2[:, 1], np.full((N,), z, np.float32)])

    # interleave: [L0,R0,L1,R1,...]
    V = np.empty((2 * N, 3), np.float32)
    V[0::2] = left3
    V[1::2] = right3

    # triangles per segment i: (L_i,R_i,L_{i+1}) and (R_i,R_{i+1},L_{i+1})
    idx = np.arange(N - 1, dtype=np.int32)
    L0 = 2 * idx
    R0 = 2 * idx + 1
    L1 = 2 * (idx + 1)
    R1 = 2 * (idx + 1) + 1

    T = np.stack([
        np.stack([L0, R0, L1], axis=1),
        np.stack([R0, R1, L1], axis=1),
    ], axis=0).reshape(-1, 3).astype(np.int32)

    C = np.tile(np.array([rgb_u8], dtype=np.uint8), (V.shape[0], 1))
    return V, T, C
