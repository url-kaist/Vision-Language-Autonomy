import os

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


class RRLogger:
    timeline = "ros_time"
    def __init__(self, output_path="/ws/external/log", name="rerun_example", max_colors=100, save=False):
        self.palette = make_palette(K=max_colors)

        if not osp.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        full_output_path = osp.join(output_path, "test_logger.rrd")

        if save:
            rr.init(name, spawn=True)
        else:
            rr.init(name)
            rr.save(full_output_path)

        self.rr = rr

    def set_time(self, t_sec=None):
        if t_sec is None:
            t_sec = rospy.Time.now().to_sec()
        self.rr.set_time_seconds(self.timeline, t_sec)
        return t_sec

    def log(self, data: Dict, t_sec=None, **kwargs):
        t_sec = self.set_time(t_sec)
        for entity_path, entity in data.items():
            self.log_single(entity_path, entity, t_sec)

    def log_single(self, entity_path, entity=None, t_sec=None, **kwargs):
        t_sec = self.set_time(t_sec)
        if isinstance(entity, str):
            entity = rr.TextLog(entity)
        elif isinstance(entity, (int, float)):
            entity = rr.Scalar(entity)
        else:
            print(f"[WARN] Invalid entity type: {type(entity)}")
        self.rr.log(entity_path, entity, **kwargs)


if __name__ == "__main__":
    rr_logger = RRLogger(name="rerun_example")