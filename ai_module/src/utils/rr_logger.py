import os
import subprocess
try:
    import rospy
except ImportError:
    from ai_module.src.utils.debug import rospy
import os.path as osp
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from typing import Dict
from skimage import color


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 -> quaternion (x,y,z,w)."""
    # 안정적인 변환 (Ken Shoemake 계열)
    m = R.astype(np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
        if i == 0:
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-12)  # Rerun 뷰어에서도 정규화되지만, 안전하게 선정규화 :contentReference[oaicite:2]{index=2}
    return q

def make_palette(K, n_candidates: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(0.1, 0.9, size=(n_candidates, 3))
    lab = color.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    selected = []
    idx = rng.integers(len(lab))
    selected.append(idx)

    dist = np.linalg.norm(lab - lab[idx], axis=1)

    for _ in range(1, K):
        idx = np.argmax(dist)
        selected.append(idx)
        dist = np.minimum(dist, np.linalg.norm(lab - lab[idx], axis=1))

    palette_rgb = (rgb[selected] * 255).astype(np.uint8)
    return palette_rgb

import time
import socket
def _wait_port(host: str, port: int, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False

class RRLogger:
    timeline = "ros_time"
    def __init__(self, output_path="/ws/external/log", name="debug", max_colors=100, save=False):
        print(f"*************** NAME: {name} ***************")
        self.palette = make_palette(K=max_colors)

        if not osp.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        full_output_path = osp.join(output_path, "test_logger.rrd")

        if save:
            rr.init(name)
            rr.save(full_output_path)
            self.rr = rr
            return

        host = "127.0.0.1"
        tcp_port = 9876
        # ws_port = 9877  # web viewer connects here
        use_terminal = True
        web_port = 9090

        if (not (_wait_port(host, tcp_port, timeout_s=0.2) and _wait_port(host, web_port, 0.2))) or (not use_terminal):
            # 2) 실패 로그를 버리지 말고 파일로 남김
            log_path = osp.join(output_path, "rerun_server.stderr.log")
            err_f = open(log_path, "a", buffering=1)

            subprocess.Popen(
                ["rerun", "--serve", "--bind", "0.0.0.0",
                 "--web-viewer-port", str(web_port),
                 "--port", str(tcp_port)],
                stdout=subprocess.DEVNULL,
                stderr=err_f,
            )

            # 3) 서버 준비 대기
            if not _wait_port(host, tcp_port, timeout_s=3.0):
                raise RuntimeError(
                    f"Rerun server did not open {host}:{tcp_port}. "
                    f"Check: {log_path}"
                )
            # if not _wait_port(host, ws_port, timeout_s=3.0):
            #     raise RuntimeError(
            #         f"Rerun server did not open {host}:{ws_port}. "
            #         f"Check: {log_path}"
            #     )
            if not _wait_port(host, web_port, timeout_s=3.0):
                raise RuntimeError(
                    f"Rerun server did not open {host}:{web_port}. "
                    f"Check: {log_path}"
                )

        rr.init(name, spawn=False)

        # 4) connect 재시도 (간단 backoff)
        last_err = None
        for _ in range(10):
            try:
                rr.connect(f"{host}:{tcp_port}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.1)

        if last_err is not None:
            raise RuntimeError(f"rr.connect failed: {last_err}")
        rr.log("debug/hello", rr.TextLog("connected"))

        prefix_sg = "SG"
        log_root = "VG/"
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    # (1) 3D: 오브젝트 박스 + 키프레임 Transform(카메라 frustum 포함)
                    rrb.Spatial3DView(
                        name="SceneGraph 3D",
                        origin="/",
                        contents=[f"SG/**"],
                    ),

                    # (2) 우측 패널: 2D 이미지 + 선택 패널
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            name="Observation",
                            origin="/",
                            contents=[f"obs/**"],
                        ),
                        rrb.Spatial2DView(
                            name="Detection",
                            origin=f"/",
                            contents=[f"det/**"],
                        ),
                        rrb.SelectionPanel(),
                    ),
                ),
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TextLogView(
                            name="VG/summary/task",
                            origin="/",
                            contents=[f"{log_root}/summary/task/**"],
                        ),
                        rrb.TextLogView(
                            name="VG/summary/status",
                            origin="/",
                            contents=[f"{log_root}/summary/status/**"],
                        ),
                        row_shares=[1, 1],
                    ),
                    rrb.TextLogView(
                        name="VG/details",
                        origin="/",
                        contents=[f"{log_root}/details/**"],
                    ),
                    column_shares=[1, 1],
                ),
            ),
            # row_shares=[1, 1],

            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextLogView(
                        name="VG/default",
                        origin="/",
                        contents=[f"{log_root}/default/**"],
                    ),
                    rrb.TextLogView(
                        name="VG/main",
                        origin="/",
                        contents=[f"{log_root}/main/**"],
                    ),
                    column_shares=[1, 1],
                ),
                rrb.Horizontal(
                    rrb.TextLogView(
                        name="VG/inference",
                        origin="/",
                        contents=[f"{log_root}/inference/**"],
                    ),
                    rrb.TextLogView(
                        name="VG/nav",
                        origin="/",
                        contents=[f"{log_root}/nav/**"],
                    ),
                    column_shares=[1, 1],
                ),
                row_shares=[1, 1],
            ),
        )
        rr.send_blueprint(blueprint, make_active=True)

        self.rr = rr

    def set_time(self, t_sec=None):
        if t_sec is None:
            t_sec = rospy.Time.now().to_sec()
        self.rr.set_time_seconds(self.timeline, t_sec)
        return t_sec

    def log(self, data: Dict, t_sec=None, **kwargs):
        t_sec = self.set_time(t_sec)
        for entity_path, entity in data.items():
            self.log_single(entity_path, entity, t_sec, **kwargs)

    def log_single(self, entity_path, entity=None, t_sec=None, level=None, **kwargs):
        t_sec = self.set_time(t_sec)
        if isinstance(entity, str):
            entity = rr.TextLog(entity, level=level)
        elif isinstance(entity, (int, float)):
            entity = rr.Scalar(entity)
        # else:
        #     print(f"[WARN] Invalid entity type: {type(entity)}")
        self.rr.log(entity_path, entity, **kwargs)




if __name__ == "__main__":
    rr_logger = RRLogger(name="rerun_example")
