import os
import sys
import time
sys.path.append('/ws/external')
import numpy as np
import cv2
import networkx as nx
import json
import shutil
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Optional, Any, ClassVar, Dict, List
from ai_module.src.visual_grounding.scripts.structures.place import Places
from ai_module.src.visual_grounding.scripts.structures.entity import Entities
from ai_module.src.visual_grounding.scripts.structures.keyframe import Keyframes


@dataclass(frozen=True)
class AttrSpec:
    required: bool = False
    ignored: bool = False
    default: Any = None


class NodeLevel(IntEnum):
    BUILDING = 0
    PLACE = 1
    OBJECT = 2
    KEYFRAME = 3
    NOT_DEFINED = 4

class Node:
    level = str(NodeLevel.NOT_DEFINED)
    schema : ClassVar[Dict[str, AttrSpec]] = {}
    def __init__(self,
                 id: int,
                 level: Optional[NodeLevel] = None,
                 **attrs: Any):
        self.id = (self.level, id)
        if level is not None:
            self.level = level

        out = {}
        for k, spec in self.schema.items():
            if spec.ignored:
                attrs.pop(k)
            elif k in attrs:
                out[k] = attrs.pop(k)
            else:
                if spec.required and spec.default is None:
                    raise TypeError(f"{self.__class__.__name__} missing required attribute: '{k}'")
                out[k] = spec.default
        if attrs:
            unknown = ", ".join(sorted(attrs.keys()))
            raise TypeError(f"{self.__class__.__name__} got unknown attributes: {unknown}")

        self._attrs = out

    def __repr__(self):
        return f"Node[{str(self.id[0])}, {self.id[1]}]()"

class BuildingNode(Node):
    level = str(NodeLevel.BUILDING)
    schema = {
        'name':         AttrSpec(),             # {str}
        'centroid':     AttrSpec(),             # {list: 3} [x,y,z] # TODO: Add (position->centroid)
        'position':     AttrSpec(ignored=True), # {list: 3} [x,y,z] # TODO: Remove
    }

class PlaceNode(Node):
    level = str(NodeLevel.PLACE)
    schema = {
        'centroid':     AttrSpec(),             # {list: 3} [x,y,z] # TODO: Add (position->centroid)
        'position':     AttrSpec(ignored=True), # {list: 3} [x,y,z] # TODO: Remove
        'image_path':   AttrSpec(ignored=True), # {str}     # TODO: Remove
        'pose':         AttrSpec(ignored=True), # {list: 4} # TODO: Remove
        'detections':   AttrSpec(ignored=True), # {list: N  {dict: 6} } # TODO: Remove
        'correct':      AttrSpec(ignored=True), # {bool} # TODO: Remove
    }

class ObjectNode(Node):
    level = str(NodeLevel.OBJECT)
    schema = {
        # Required
        'name':         AttrSpec(required=True, default='unknown'), # {str} # TODO: Add
        'points':       AttrSpec(required=True),                    # {list: N [ {list:3} ]}
        'centroid':     AttrSpec(required=True),                    # {list: 3} [x,y,z] # TODO: Add (position->centroid)
        'extent':       AttrSpec(required=True),                    # {list: 3} [x,y,z] # TODO: Add
        'R':            AttrSpec(required=True),                    # {list: 3 [{list:3}]} # TODO: Add
        # Ignored
        'type': AttrSpec(ignored=True),  # {int}                    # TODO: Remove
        'instance_id':  AttrSpec(ignored=True),                     # {int} # TODO: Remove
        'class_name': AttrSpec(required=True), # {str}              # TODO: Add
        'position': AttrSpec(ignored=True),  # {list: 3}            # TODO: Remove
        'has_close_place':AttrSpec(ignored=True),                   # {bool} # TODO: Remove
        'closest_temp_dist': AttrSpec(ignored=True),                # {float} # TODO: Remove
        'closest_temp_place': AttrSpec(ignored=True),               # {float} # TODO: Remove
        'yolo_confs': AttrSpec(ignored=True),                       # {float} # TODO: Remove
        'bbox_by_kf': AttrSpec(ignored=True),                       # {Dict[kf_id(int): {list:4}]} # TODO: Remove
    }

class KeyframeNode(Node):
    level = str(NodeLevel.KEYFRAME)
    schema = {
        # Required
        'image_path':   AttrSpec(required=True),    # {str}
        'pose':         AttrSpec(required=True),    # {list: 4 {list: 4} }
        'detections':   AttrSpec(required=True),    # {list: Detections}, Detections(id{int}, bbox{list: 4})
        # Ignored
        'position':     AttrSpec(ignored=True),     # {list: 3} [x,y,z]
        'type':         AttrSpec(ignored=True),
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attrs = getattr(self, '_attrs', {})
        if not 'image_path' in attrs:
            raise ValueError(f"KeyframeNode must have image_path")
        image_path = attrs['image_path']
        attrs['fname'] = image_path.split("/")[-1]
        attrs['image'] = cv2.imread(image_path)
        if attrs['image'] is None:
            keyframe_dir = os.environ.get('KEYFRAMES_DIR2', '/ws/external/test_data/vla_js_chair_2025-12-17-12-17-43/keyframes')
            image_path = os.path.join(keyframe_dir, attrs['fname'])
            attrs['image'] = cv2.imread(image_path)
            attrs['image_path'] = image_path

_NODE_LEVEL_TO_CLS = {
    NodeLevel.BUILDING: BuildingNode,
    NodeLevel.PLACE: PlaceNode,
    NodeLevel.OBJECT: ObjectNode,
    NodeLevel.KEYFRAME: KeyframeNode,
    str(NodeLevel.BUILDING): BuildingNode,
    str(NodeLevel.PLACE): PlaceNode,
    str(NodeLevel.OBJECT): ObjectNode,
    str(NodeLevel.KEYFRAME): KeyframeNode,
}


class SceneGraph:
    def __init__(self, candidate_names=[], reference_names=[], save_dir='/ws/external/log/sg', *args, **kwargs):
        self._lock = threading.Lock()
        self.G = nx.DiGraph()
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.candidate_names = candidate_names
        self.reference_names = reference_names
        self.related_names = list(set(candidate_names + reference_names))

        self.depth_K = depth_K = np.array(
            [
                [383.73699951171875, 0.0, 324.5614929199219],
                [0.0, 383.73699951171875, 238.5606689453125],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.fx_d, self.fy_d = float(depth_K[0, 0]), float(depth_K[1, 1])
        self.cx_d, self.cy_d = float(depth_K[0, 2]), float(depth_K[1, 2])

        self.rgb_K = rgb_K = np.array(
            [
                [379.15924072265625, 0.0, 320.7159423828125],
                [0.0, 378.71624755859375, 237.7329559326172],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.fx_rgb, self.fy_rgb = float(rgb_K[0, 0]), float(rgb_K[1, 1])
        self.cx_rgb, self.cy_rgb = float(rgb_K[0, 2]), float(rgb_K[1, 2])

        self.cam_to_body_R = np.array(
            [
                [0.0, 0.0, 1.0],  # x_b ← z_cam
                [-1.0, 0.0, 0.0],  # y_b ← -x_cam
                [0.0, -1.0, 0.0],  # z_b ← -y_cam
            ],
            dtype=np.float32,
        )
        self.cam_to_body_t = np.zeros(3, dtype=np.float32)

        self.image_height, self.image_width = 480, 640
        self.z_const = -0.5
        self.max_range = 8.0
        self.ground_offset = np.array([0.0, 0.0, 0.8])

    @property
    def fov_x(self):
        return 2 * np.arctan(self.image_width / (2 * self.fx_rgb))

    @property
    def fov_y(self):
        return 2 * np.arctan(self.image_height / (2 * self.fy_rgb))

    def save_path(self, etype, fname='', suffix=''):
        save_path = os.path.join(self.save_dir, etype)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        return os.path.join(save_path, fname)

    def __repr__(self) -> str:
        repr = (f"SceneGraph")
        return repr

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def update(self, scene_graph, objects, **kwargs) -> None:
        with (self._lock):
            for data in scene_graph.get('nodes', []):
                level = data.pop('level')
                id = data.pop('id')
                if id < 0:
                    continue

                NodeCls = _NODE_LEVEL_TO_CLS[level]
                try:
                    node = NodeCls(id=id, **data)
                except Exception as e:
                    print(f">>>> Error at SceneGraph.update: {level}, {id}, {data}")
                    continue

                if level == str(NodeLevel.KEYFRAME) and \
                    (self.image_width is None or self.image_height is None):
                    image = cv2.imread(data['image_path'])
                    self.image_height, self.image_width = image.shape[:-1]

                self.G.add_node(node.id, **node.__dict__)
                # print(f"Add node: {node}")

            for data in scene_graph.get('edges', []):
                source = (data['source']['level'], data['source']['id'])
                target = (data['target']['level'], data['target']['id'])

                # TODO: Remove
                if target[0] == str(NodeLevel.PLACE):
                    target = (str(NodeLevel.KEYFRAME), target[1])
                if source[0] == str(NodeLevel.PLACE):
                    source = (str(NodeLevel.KEYFRAME), source[1])
                if target[0] == str(NodeLevel.OBJECT) and target[1] < 0:
                    continue
                if source[0] == str(NodeLevel.OBJECT) and source[1] < 0:
                    continue
                if target[0] == source[0]:
                    continue

                # filter the obj and kf edges
                # s_is_obj = (source[0] == str(NodeLevel.OBJECT))
                # t_is_obj = (target[0] == str(NodeLevel.OBJECT))
                # s_is_kf = (source[0] == str(NodeLevel.KEYFRAME))
                # t_is_kf = (target[0] == str(NodeLevel.KEYFRAME))

                # if (s_is_obj and t_is_kf):
                #     if not self._ok_object_keyframe_edge(source, target, min_point_ratio=0.1, min_bbox_area_ratio=0.002):
                #         continue
                # elif (s_is_kf and t_is_obj):
                #     if not self._ok_object_keyframe_edge(target, source, min_point_ratio=0.1, min_bbox_area_ratio=0.002):
                #         continue

                self.G.add_edge(source, target)
                self.G.add_edge(target, target)
                # print(f"Add edge: {source} <-> {target}")

            # self.update_projection_links(
            #     min_point_ratio=0.1,
            #     min_bbox_area_ratio=0.002,
            # )
        print(f"=> Graph: {self.G}")

    def _projection_stats(self, pts_world: np.ndarray, pose: np.ndarray, image_size):
        """
        pts_world: (N,3)
        pose: (4,4) body->world (현재 코드의 convention 그대로 사용)
        return:
          proj_ratio: 투영 성공한 point 비율
          bbox_area_ratio: bbox 면적 / 이미지 면적
          bbox: (u_min,v_min,u_max,v_max) (없으면 (0,0,0,0))
        """
        H, W = image_size
        xs, ys = self.project_pts(pts_world, pose, image_size=(H, W))

        n_total = int(pts_world.shape[0])
        n_proj = int(len(xs)) if xs is not None else 0
        proj_ratio = (n_proj / max(n_total, 1))

        if xs is None or len(xs) == 0:
            return proj_ratio, 0.0, (0, 0, 0, 0)

        u_min, v_min, u_max, v_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        bw = max(0, u_max - u_min)
        bh = max(0, v_max - v_min)
        bbox_area = float(bw * bh)
        bbox_area_ratio = bbox_area / float(max(H * W, 1))

        return proj_ratio, bbox_area_ratio, (u_min, v_min, u_max, v_max)

    def update_projection_links(self, min_point_ratio: float = 0.05, min_bbox_area_ratio: float = 0.002):
        """
        각 OBJECT(eid)가 KEYFRAME(pid)들에 충분히 투영되면 OBJECT<->KEYFRAME edge를 추가한다.
        - min_point_ratio: object point 중 이미지 안으로 투영되는 점 비율 최소값
        - min_bbox_area_ratio: bbox 면적/이미지 면적 최소값 (너무 작으면 멀리 있거나 거의 안보임)
        """
        # 노드 목록 수집
        object_nodes = []
        keyframe_nodes = []
        for (level, nid), data in self.G.nodes(data=True):
            if level == str(NodeLevel.OBJECT):
                object_nodes.append(((level, nid), data))
            elif level == str(NodeLevel.KEYFRAME):
                keyframe_nodes.append(((level, nid), data))

        for (obj_key, obj_data) in object_nodes:
            e_attrs = obj_data.get("_attrs", {})
            pts = np.asarray(e_attrs.get("points", []), dtype=np.float32)
            if pts.size == 0:
                continue

            good_pids = []
            for (kf_key, kf_data) in keyframe_nodes:
                kf_attrs = kf_data.get("_attrs", {})
                pose = np.asarray(kf_attrs.get("pose", None), dtype=np.float32)
                img = kf_attrs.get("image", None)
                if pose is None or img is None:
                    continue

                H, W = img.shape[:2]
                proj_ratio, area_ratio, bbox = self._projection_stats(pts, pose, (H, W))

                ok = (proj_ratio >= min_point_ratio) and (area_ratio >= min_bbox_area_ratio)
                if ok:
                    good_pids.append(kf_key)

            for kf_key in good_pids:
                self.G.add_edge(obj_key, kf_key)
                self.G.add_edge(kf_key, obj_key)

    def _ok_object_keyframe_edge(
            self,
            obj_key,
            kf_key,
            min_point_ratio: float,
            min_bbox_area_ratio: float,
    ) -> bool:
        """
        obj_key: (str(NodeLevel.OBJECT), eid)
        kf_key : (str(NodeLevel.KEYFRAME), pid)
        """
        if (obj_key not in self.G.nodes) or (kf_key not in self.G.nodes):
            return False

        obj_data = self.G.nodes[obj_key]
        kf_data = self.G.nodes[kf_key]

        e_attrs = obj_data.get("_attrs", {})
        kf_attrs = kf_data.get("_attrs", {})

        pts = np.asarray(e_attrs.get("points", []), dtype=np.float32)
        if pts.size == 0:
            return False

        pose = np.asarray(kf_attrs.get("pose", None), dtype=np.float32)
        img = kf_attrs.get("image", None)

        # KeyframeNode에서 image를 읽어오지만, None일 수도 있어서 안전 처리 :contentReference[oaicite:2]{index=2}
        if img is None:
            image_path = kf_attrs.get("image_path", None)
            if image_path:
                img = cv2.imread(image_path)
                kf_attrs["image"] = img

        if pose is None or img is None:
            return False

        H, W = img.shape[:2]
        proj_ratio, area_ratio, _ = self._projection_stats(pts, pose, (H, W))  # :contentReference[oaicite:3]{index=3}
        return (proj_ratio >= min_point_ratio) and (area_ratio >= min_bbox_area_ratio)

    def get_entity_names(self, names, *args, **kwargs) -> Entities:
        output_entities = []
        for (level, id), data in self.G.nodes(data=True):
            if level == str(NodeLevel.OBJECT):
                name = data.get("_attrs", {}).get("name")
                if name in names:
                    output_entities.append(data)
        return output_entities

    def get_candidate_entities(self, *args, **kwargs) -> List:
        output_entities = []
        for (level, id), data in self.G.nodes(data=True):
            if level == str(NodeLevel.OBJECT):
                name = data.get("_attrs", {}).get("name")
                if name in self.candidate_names:
                    output_entities.append(id)
        return list(set(output_entities))

    def get_reference_entities(self, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        return self.get_entity_names(self.reference_names, etype=etype)

    def get_related_entities(self, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        return self.get_entity_names(self.related_names, etype=etype)

    @property
    def keyframes(self):
        output = []
        for (level, id), data in self.G.nodes(data=True):
            if level == str(NodeLevel.KEYFRAME):
                output.append(data)
        return output

    @property
    def eid2pids(self):
        G = self.G
        eid2pids = {}
        for u, v in G.edges():
            ulev, uid = G.nodes[u].get('id', (None, None))
            vlev, vid = G.nodes[v].get('id', (None, None))
            if not uid or not vid:
                continue
            if ulev == str(NodeLevel.OBJECT) and vlev == str(NodeLevel.KEYFRAME):
                if not uid in eid2pids:
                    eid2pids[uid] = []
                eid2pids[uid] = list(set(eid2pids[uid] + [vid]))
            elif ulev == str(NodeLevel.KEYFRAME) and vlev == str(NodeLevel.OBJECT):
                if not vid in eid2pids:
                    eid2pids[vid] = []
                eid2pids[vid] = list(set(eid2pids[vid] + [uid]))
        return eid2pids

    @property
    def pid2eids(self):
        G = self.G
        pid2eids = {}
        for u, v in G.edges():
            ulev, uid = G.nodes[u].get('id', (None, None))
            vlev, vid = G.nodes[v].get('id', (None, None))
            if not uid or not vid:
                continue
            if ulev == str(NodeLevel.OBJECT) and vlev == str(NodeLevel.KEYFRAME):
                if not vid in pid2eids:
                    pid2eids[vid] = []
                pid2eids[vid] = list(set(pid2eids[vid] + [uid]))
            elif ulev == str(NodeLevel.KEYFRAME) and vlev == str(NodeLevel.OBJECT):
                if not uid in pid2eids:
                    pid2eids[uid] = []
                pid2eids[uid] = list(set(pid2eids[uid] + [vid]))
        return pid2eids

    def project_pts(self, pts_world, pose, image_size):
        R_b2w = pose[:3, :3]
        t_b2w = pose[:3, 3]

        pts_body = (pts_world - t_b2w) @ R_b2w
        pts_cam = (pts_body - self.cam_to_body_t) @ self.cam_to_body_R

        X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        valid = Z > 0
        X = X[valid]
        Y = Y[valid]
        Z = Z[valid]

        fx_rgb, fy_rgb = float(self.rgb_K[0, 0]), float(self.rgb_K[1, 1])
        cx_rgb, cy_rgb = float(self.rgb_K[0, 2]), float(self.rgb_K[1, 2])
        xs = fx_rgb * (X / Z) + cx_rgb
        ys = fy_rgb * (Y / Z) + cy_rgb

        image_height, image_width = image_size
        in_img = (
                (xs >= 0) & (xs < image_width) &
                (ys >= 0) & (ys < image_height)
        )
        xs = xs[in_img]
        ys = ys[in_img]
        return xs, ys

    def project_entity_bbox(self, entity, kf):
        e_attrs = entity.get("_attrs", {})
        pts_world = np.asarray(e_attrs['points'], dtype=np.float32)

        kf_attrs = kf.get("_attrs", {})
        pose = np.array(kf_attrs['pose'], dtype=np.float32)

        image_height, image_width, _ = kf_attrs['image'].shape
        xs, ys = self.project_pts(pts_world, pose, image_size=(image_height, image_width))
        if (xs is None) or (len(xs) == 0):
            return (0, 0, 0, 0)
        u_min, v_min, u_max, v_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        return (u_min, v_min, u_max, v_max)


if __name__ == "__main__":
    DATA_DIR = "/ws/external/bags/20251224/6count_2025-12-24-21-01-39/offline_map"
    DATA_DIR = "/ws/data/VLA/offline_map"
    dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    dir_sorted = sorted(dirs, key=os.path.getmtime)
    styles = {
        'reference': {'show': True, 'color': 'green'},
        'candidate': {'show': True, 'color': 'blue'},
    }

    sg = SceneGraph(candidate_names=['fire extinguisher'], reference_names=['TV monitor'])
    for dir in dir_sorted:
        with open(os.path.join(dir, 'scene_graph.json'), 'r', encoding='utf-8') as f:
            scene_graph = json.load(f)
        with open(os.path.join(dir, 'objects.json'), 'r', encoding='utf-8') as f:
            objects = json.load(f)
        sg.update(scene_graph, objects)

        print('--------------')
        pid2eids = sg.pid2eids

        # eid -> object name 캐시
        eid2name = {}
        for (level, nid), data in sg.G.nodes(data=True):
            if level == str(NodeLevel.OBJECT):
                eid2name[nid] = data.get("_attrs", {}).get("name", "unknown")

        # pid별로 (eid, name) 형태로 출력
        for pid, eids in sorted(pid2eids.items(), key=lambda x: x[0]):
            pairs = [(eid, eid2name.get(eid, "unknown")) for eid in sorted(eids)]
            print(f"pid={pid}: {pairs}")

        # for kf_id, kf in sg.keyframes.items():
        #     kf.annotate(styles, node_name='test', suffix='_annotated_global')

        time.sleep(0.1)

    print("Done")
