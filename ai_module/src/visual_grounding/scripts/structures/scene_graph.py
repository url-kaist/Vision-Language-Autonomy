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
        attrs['image'] = cv2.imread(image_path)
        attrs['fname'] = image_path.split("/")[-1]

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
                [389.8971252441406, 0.0, 325.1298828125],
                [0.0, 389.8971252441406, 236.91766357421875],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.fx_d, self.fy_d = float(depth_K[0, 0]), float(depth_K[1, 1])
        self.cx_d, self.cy_d = float(depth_K[0, 2]), float(depth_K[1, 2])

        self.rgb_K = rgb_K = np.array(
            [
                [606.040283203125, 0.0, 328.3797912597656],
                [0.0, 606.2955932617188, 245.35792541503906],
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
        # try:
        #     # ROS version
        #     for data in scene_graph.get('nodes', []):
        #         level = data.pop('level')
        #         id = data.pop('id')
        #         if id < 0:
        #             continue
        #         NodeCls = _NODE_LEVEL_TO_CLS[level]
        #         node = NodeCls(id=id, **data)
        #         self.G.add_node(node.id, **node.__dict__)
        #         print(f"Add node: {node}")
        #
        # except:
        #     # Python version
        with self._lock:
            for data in scene_graph.get('nodes', []):
                level = data.pop('level')
                id = data.pop('id')
                if id < 0:
                    continue

                NodeCls = _NODE_LEVEL_TO_CLS[level]
                node = NodeCls(id=id, **data)
                self.G.add_node(node.id, **node.__dict__)
                print(f"Add node: {node}")

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

                self.G.add_edge(source, target)
                self.G.add_edge(target, target)
                print(f"Add edge: {source} <-> {target}")
        print(f"=> Graph: {self.G}")


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
        
        if xs is None or ys is None or len(xs) == 0 or len(ys) == 0:
            return None
        
        u_min, v_min, u_max, v_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        return (u_min, v_min, u_max, v_max)

    def render_object_minimal(
        self,
        scope: Literal["all", "candidate", "related"] = "related",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return JSON-able dict:
        { "<eid>": {"level": <str>, "id": <int>, "name": <str|None>, "centroid": <list|None>} }
        Only OBJECT nodes are included.
        """

        def _passes_scope(obj_attrs: dict) -> bool:
            n = obj_attrs.get("name", None)
            cn = obj_attrs.get("class_name", None)

            if scope == "all":
                return True
            if scope == "candidate":
                return (n in self.candidate_names) or (cn in self.candidate_names)
            if scope == "related":
                return (n in self.related_names) or (cn in self.related_names)
            raise ValueError(f"Unknown scope: {scope}")

        out: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            for (level, eid), data in self.G.nodes(data=True):
                # only object nodes
                if level != str(NodeLevel.OBJECT):
                    continue

                attrs = data.get("_attrs", {}) or {}
                if not _passes_scope(attrs):
                    continue

                name = attrs.get("name", None)
                centroid = attrs.get("centroid", None)

                # ensure JSON-serializable
                if hasattr(centroid, "tolist"):
                    centroid = centroid.tolist()

                out[str(int(eid))] = {
                    "level": level,          # e.g. "NodeLevel. OBJECT"
                    "id": int(eid),          # e.g. 17
                    "name": name,            # e.g. "chair"
                    "centroid": centroid,    # e.g. [x,y,z]
                }

        return out
        
        
if __name__ == "__main__":
    DATA_DIR = "/ws/external/offline_map"
    dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    dir_sorted = sorted(dirs, key=os.path.getmtime)
    styles = {
        'reference': {'show': True, 'color': 'green'},
        'candidate': {'show': True, 'color': 'blue'},
    }

    sg = SceneGraph(candidate_names=['fire extinguisher'], reference_names=['TV'])
    for dir in dir_sorted:
        with open(os.path.join(dir, 'scene_graph.json'), 'r', encoding='utf-8') as f:
            scene_graph = json.load(f)
        with open(os.path.join(dir, 'objects.json'), 'r', encoding='utf-8') as f:
            objects = json.load(f)
        sg.update(scene_graph, objects)

        # for kf_id, kf in sg.keyframes.items():
        #     kf.annotate(styles, node_name='test', suffix='_annotated_global')
        from ai_module.src.utils.visualizer import Visualizer

        visualizer = Visualizer()

        for etype in ['object']:
            for (level, id), kf in sg.G.nodes(data=True):
                if level == str(NodeLevel.KEYFRAME):
                    attrs = kf.get('_attrs', {})

                    image = attrs['image'].copy()
                    H, W = image.shape[:2]
                    fname = attrs['fname']
                    detections = attrs['detections']
                    if etype == 'detection':
                        for det in detections:
                            is_object = det['id'] > 0
                            if not is_object:
                                continue

                            is_candidate = (det['name'] in sg.candidate_names)
                            is_reference = (det['name'] in sg.reference_names)
                            if is_candidate:
                                style = styles.get('candidate', {'show': False})
                                if not style['show']:
                                    continue
                                color = style.get('color', 'green')
                            elif is_reference:
                                style = styles.get('reference', {'show': False})
                                if not style['show']:
                                    continue
                                color = style.get('color', 'blue')
                            else:
                                continue

                            bbox = det['bbox']
                            color = visualizer._parse_color(color)
                            u_min, v_min, u_max, v_max = bbox
                            u_min, v_min, u_max, v_max = int(u_min), int(v_min), int(u_max), int(v_max)
                            top_left, bottom_right = (u_min, v_min), (u_max, v_max)

                            cv2.rectangle(image, top_left, bottom_right, color=color, thickness=2)
                            cv2.putText(
                                image, f"{det['id']}", (u_min, v_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                            )
                    elif etype == 'object':
                        # eids = self.sg.pid2eids.get(id, [])
                        for (elevel, eid), entity in sg.G.nodes(data=True):
                            if elevel == str(NodeLevel.OBJECT):
                                # if eid in eids:
                                bbox = sg.project_entity_bbox(entity, kf)

                                if bbox is None:
                                    continue

                                e_attrs = entity.get("_attrs", {})

                                is_candidate = (e_attrs['name'] in sg.candidate_names)
                                is_reference = (e_attrs['name'] in sg.reference_names)

                                if is_candidate:
                                    style = styles.get('candidate', {'show': False})
                                    if not style['show']:
                                        continue
                                    color = style.get('color', 'green')
                                elif is_reference:
                                    style = styles.get('reference', {'show': False})
                                    if not style['show']:
                                        continue
                                    color = style.get('color', 'blue')
                                else:
                                    continue

                                color = visualizer._parse_color(color)
                                u_min, v_min, u_max, v_max = bbox
                                u_min, v_min, u_max, v_max = int(u_min), int(v_min), int(u_max), int(v_max)
                                top_left, bottom_right = (u_min, v_min), (u_max, v_max)

                                if u_max < 0 or v_max < 0 or u_min >= W or v_min >= H:
                                    continue

                                cv2.rectangle(image, top_left, bottom_right, color=color, thickness=2)
                                cv2.putText(
                                    image, f"{eid}", (u_min, v_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                                )
                    else:
                        raise NotImplementedError("No implementation for other etypes")
                    save_path = sg.save_path(etype, fname)
                    success = cv2.imwrite(save_path, image)

        time.sleep(0.1)

    print("Done")
