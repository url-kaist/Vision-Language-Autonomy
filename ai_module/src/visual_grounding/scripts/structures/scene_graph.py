import os
import sys
import time
sys.path.append('/ws/external')
import json
import threading
from typing import Literal
from ai_module.src.visual_grounding.scripts.structures.place import Places
from ai_module.src.visual_grounding.scripts.structures.entity import Entities
from ai_module.src.visual_grounding.scripts.structures.keyframe import Keyframes


class SceneGraph:
    def __init__(self, places=None, entities=None, candidate_names=None, reference_names=None, *args, **kwargs):
        self._lock = threading.Lock()

        candidate_names = [candidate_names] if isinstance(candidate_names, str) else candidate_names
        reference_names = [reference_names] if isinstance(reference_names, str) else reference_names
        assert not (set(candidate_names or []) & set(reference_names or [])), \
            "Candidate and reference names must not overlap"
        self.candidate_names = list(candidate_names or [])
        self.reference_names = list(reference_names or [])
        self.related_names = list(candidate_names or []) + list(reference_names or [])

        self.places = places if places else Places()
        self.entities = entities if entities else Entities()
        self.keyframes = Keyframes(candidate_names=self.candidate_names, reference_names=self.reference_names, *args, **kwargs)
        # self.history_keyframes = Keyframes(candidate_names=self.candidate_names, reference_names=self.reference_names, *args, **kwargs)

    def __repr__(self) -> str:
        repr = (f"SceneGraph("
                f"#places={len(self.places)}, "
                f"#objects={len(self.entities('object'))}, "
                f"#detections={len(self.entities('detection'))})")
        return repr

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def has_candidate(self):
        return len(self.keyframes.get_entity_names(self.candidate_names)) > 0

    def update(self, scene_graph, objects, **kwargs) -> None:
        with self._lock:
            self.places.update(scene_graph, **kwargs)
            self.entities.update(scene_graph, objects)
            self.keyframes.update(scene_graph, self.places, self.entities)
            
            # Update history keyframes
            # for kf_id, kf in self.keyframes.items():
            #     if kf_id >= 10000:
            #         self.history_keyframes[kf_id] = kf

    def get_entity_names(self, names, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        if not isinstance(names, list):
            names = [names]
        return Entities({id: data for id, data in self.entities(etype).items() if data.name in names})

    def get_candidate_entities(self, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        return self.get_entity_names(self.candidate_names, etype=etype)

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
