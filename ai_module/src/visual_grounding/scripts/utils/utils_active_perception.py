import numpy as np
import matplotlib.pyplot as plt


def ensure_ccw(hull_xy: np.ndarray) -> np.ndarray:
    """hull이 CCW가 아니면 뒤집어서 CCW로 만듦."""
    P = np.asarray(hull_xy, dtype=np.float32)
    if P.shape[0] < 3:
        return P
    # signed area (shoelace); >0 이면 CCW
    x, y = P[:, 0], P[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if area2 < 0:
        P = P[::-1].copy()
    return P


def visible_edges_from_pose(hull_xy, agent_pose, fov_rad=None, max_range=8.0, eps=1e-8):
    # visible_edges_from_pose
    P = ensure_ccw(hull_xy)
    M = P.shape[0]
    if M < 2:
        return np.zeros((0,), bool), np.empty((0, 2, 2), np.float32), np.empty((0,), np.int32)

    v = np.array([agent_pose[0], agent_pose[1]], dtype=np.float32)

    # edges: e[i] = P[i+1] - P[i]
    Pn = np.roll(P, -1, axis=0)
    E = Pn - P  # (M, 2)

    # outward normal for CCW polygon: right normal (e_y, -e_x)
    N = np.stack([E[:, 1], -E[:, 0]], axis=1)  # (M, 2)

    # dot test
    V = (v[None, :] - P)  # (M, 2)
    dots = (N * V).sum(axis=1)
    vis = dots > eps

    theta = float(agent_pose[2])
    heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

    mid = 0.5 * (P + Pn)
    dir_vec = mid - v[None, :]
    dist = np.linalg.norm(dir_vec, axis=1) + eps
    dir_unit = dir_vec / dist[:, None]

    if fov_rad is not None:
        # cos(angle) = heading * dir_unit
        cosang = (dir_unit * heading[None, :]).sum(axis=1)
        # angle <= fov/2 <=> cosang >= cos(fov/2)
        vis &= (cosang >= np.cos(0.5 * fov_rad))

    if max_range is not None:
        vis &= (dist <= float(max_range))

    idx = np.nonzero(vis)[0].astype(np.int32)
    segs = np.stack([P[idx], Pn[idx]], axis=1).astype(np.float32)  # (K,2,2)

    return vis, segs, P


def visualize_visibility(P, agent_pose, segs, fov_rad, max_range, save_path='/ws/external/vis/visibility.jpg'):
    fig, ax = plt.subplots(figsize=(7, 7))

    # 1) hull polygon
    poly = np.vstack([P, P[:1]])
    ax.plot(poly[:, 0], poly[:, 1], linewidth=2, label='hull')

    # 2) agent position & heading
    x, y, theta = agent_pose
    ax.scatter([x], [y], s=50, label='agent')
    hx, hy = np.cos(theta), np.sin(theta)
    ax.arrow(x, y, 0.8 * hx, 0.8 * hy, head_width=0.2, length_includes_head=True)

    # 3) FoV wedge
    a1 = theta - 0.5 * fov_rad
    a2 = theta + 0.5 * fov_rad
    r = float(max_range)
    ax.plot([x, x + r * np.cos(a1)], [y, y + r * np.sin(a1)], linestyle='--', linewidth=1)
    ax.plot([x, x + r * np.cos(a2)], [y, y + r * np.sin(a2)], linestyle='--', linewidth=1)
    aa = np.linspace(a1, a2, 64)
    ax.plot(x + r * np.cos(aa), y + r * np.sin(aa), linestyle='--', linewidth=1, label="FoV")

    # 4) Visible edges
    for s in segs:
        ax.plot([s[0, 0], s[1, 0]], [s[0, 1], s[1, 1]], linewidth=4, label=None)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("")
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
