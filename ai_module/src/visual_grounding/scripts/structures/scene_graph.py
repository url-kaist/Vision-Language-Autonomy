import os
import sys
import time
sys.path.append('/ws/external')
import numpy as np
import copy
import networkx as nx
import json
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Optional, Any, ClassVar, Dict
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
    level = NodeLevel.NOT_DEFINED
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
    level = NodeLevel.BUILDING
    schema = {
        'name':         AttrSpec(),             # {str}
        'centroid':     AttrSpec(),             # {list: 3} [x,y,z] # TODO: Add (position->centroid)
        'position':     AttrSpec(ignored=True), # {list: 3} [x,y,z] # TODO: Remove
    }

class PlaceNode(Node):
    level = NodeLevel.PLACE
    schema = {
        'centroid':     AttrSpec(),             # {list: 3} [x,y,z] # TODO: Add (position->centroid)
        'position':     AttrSpec(ignored=True), # {list: 3} [x,y,z] # TODO: Remove
        'image_path':   AttrSpec(ignored=True), # {str}     # TODO: Remove
        'pose':         AttrSpec(ignored=True), # {list: 4} # TODO: Remove
        'detections':   AttrSpec(ignored=True), # {list: N  {dict: 6} } # TODO: Remove
        'correct':      AttrSpec(ignored=True), # {bool} # TODO: Remove
    }

class ObjectNode(Node):
    level = NodeLevel.OBJECT
    schema = {
        'name':         AttrSpec(required=True, default='unknown'), # {str} # TODO: Add
        'points':       AttrSpec(),                                 # {list: N [ {list:3} ]}
        'centroid':     AttrSpec(default=[0.0, 0.0, 0.0]),          # {list: 3} [x,y,z] # TODO: Add (position->centroid)
        'extent':       AttrSpec(default=[0.0, 0.0, 0.0]),          # {list: 3} [x,y,z] # TODO: Add
        'R':            AttrSpec(default=np.eye(3)),                # {list: 3 [{list:3}]} # TODO: Add
        'position':     AttrSpec(ignored=True),                     # {list: 3} # TODO: Remove
        'instance_id':  AttrSpec(ignored=True),                     # {int} # TODO: Remove
        'has_close_place':AttrSpec(ignored=True),                   # {bool} # TODO: Remove
        'closest_temp_dist': AttrSpec(ignored=True),                # {float} # TODO: Remove
        'closest_temp_place': AttrSpec(ignored=True),               # {float} # TODO: Remove
        'yolo_confs': AttrSpec(ignored=True),                       # {float} # TODO: Remove
        'bbox_by_kf': AttrSpec(ignored=True),                       # {Dict[kf_id(int): {list:4}]} # TODO: Remove
    }

class KeyframeNode(Node):
    level = NodeLevel.KEYFRAME
    schema = {
        'image_path':   AttrSpec(required=True, default='vis/000000.jpg'),  # {str} # TODO: Add
        'pose':         AttrSpec(required=True, default=np.eye(4)),         # {list: 4 {list: 4} } # TODO: Add
        'detections':   AttrSpec(required=True, default=[]),                # {list: Detections}, Detections(id{int}, bbox{list: 4}) # TODO: Add
        'position':     AttrSpec(ignored=True),                             # {list: 3} [x,y,z]
    }

_NODE_LEVEL_TO_CLS = {
    NodeLevel.BUILDING: BuildingNode,
    NodeLevel.PLACE: PlaceNode,
    NodeLevel.OBJECT: ObjectNode,
    NodeLevel.KEYFRAME: KeyframeNode,
}


class SceneGraph:
    def __init__(self, places=None, entities=None, candidate_names=None, reference_names=None, *args, **kwargs):
        self._lock = threading.Lock()
        self.G = nx.DiGraph()

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
        try:
            # ROS version
            for id, data in scene_graph.nodes.items():
                break # TODO:
                if 'place' in id:
                    if not 'image_path' in data.keys():
                        logging.warning(f"image_path is not in data: {data}")
                    if not "id" in data:
                        data['id'] = parse_id(id)
                    self[parse_id(id)] = Place(data)
        except:
            # Python version
            with self._lock:
                for data in scene_graph.get('nodes', []):
                    level = str(data.pop('type')).lower() # TODO: 'type' -> 'level'

                    # TODO: Remove
                    if level in str(NodeLevel.BUILDING).lower():   # Building
                        level = NodeLevel.BUILDING
                    elif level in str(NodeLevel.PLACE).lower():    # Place
                        level = NodeLevel.PLACE
                    elif level in str(NodeLevel.OBJECT).lower():   # Object
                        level = NodeLevel.OBJECT
                    elif level in str(NodeLevel.KEYFRAME).lower(): # Keyframe
                        level = NodeLevel.KEYFRAME
                    else:
                        raise TypeError(f"Node level must be in ")

                    # TODO: Remove
                    if 'id' in data:
                        id = data.pop('id')
                    else:
                        id = max((node_id for (lvl, node_id) in self.G.nodes if lvl == level), default=-1) + 1

                    # TODO: Remove
                    if level == NodeLevel.OBJECT:
                        data['name'] = data.pop('class_name')
                        if isinstance(id, str):
                            id = id.replace("object_", "")
                        data['centroid'] = objects[id]['center']
                        data['points'] = objects[id]['points']

                    NodeCls = _NODE_LEVEL_TO_CLS[level]
                    node = NodeCls(id=id, **data)
                    self.G.add_node(node.id, **node.__dict__)
                    # print(f"Add node: {node}")

                # for data in scene_graph.get('links', []):
                #     source = (data['source']['level'], data['source']['id']) # TODO: Apply
                #     target = (data['target']['level'], data['target']['id']) # TODO: Apply
                #     self.G.add_edge(source, target)
                #     self.G.add_edge(target, target)
                #     # print(f"Add edge: {source} <-> {target}")
            # print(f"=> Graph: {self.G}")


    def get_entity_names(self, names, etype: Literal['object', 'detection', 'all'] = 'object', *args, **kwargs) -> Entities:
        if not isinstance(names, list):
            names = [names]
        return Entities({id: data for id, data in self.entities(etype).items() if data.name in names}, *args, **kwargs)

    def get_candidate_entities(self, etype: Literal['object', 'detection', 'all'] = 'object', *args, **kwargs) -> Entities:
        return self.get_entity_names(self.candidate_names, etype=etype, *args, **kwargs)

    def get_reference_entities(self, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        return self.get_entity_names(self.reference_names, etype=etype)

    def get_related_entities(self, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        return self.get_entity_names(self.related_names, etype=etype)


if __name__ == "__main__":
    DATA_DIR = "/ws/external/test_data/offline_map"
    dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    dir_sorted = sorted(dirs, key=os.path.getmtime)
    styles = {
        'reference': {'show': True, 'color': 'green'},
        'candidate': {'show': True, 'color': 'blue'},
    }

    sg = SceneGraph(candidate_names='pillow', reference_names=['sofa'])
    for dir in dir_sorted:
        with open(os.path.join(dir, 'scene_graph.json'), 'r', encoding='utf-8') as f:
            scene_graph = json.load(f)
        with open(os.path.join(dir, 'objects.json'), 'r', encoding='utf-8') as f:
            objects = json.load(f)
        sg.update(scene_graph, objects)

        for kf_id, kf in sg.keyframes.items():
            kf.annotate(styles, node_name='test', suffix='_annotated_global')

        time.sleep(0.1)

    print("Done")
