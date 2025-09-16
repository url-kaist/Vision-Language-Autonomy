import os
import logging
from .data import Data, Datas
from .utils import parse_id


class Place(Data):
    _rename_map = {'room_label': 'name'}
    _equal_keys = ['id', 'position', 'room_label', 'image_path', 'detections']
    _default_image_dir = "/ws/external/keyframes"
    _repr_keys = ['id', 'position', 'room_label', 'image_path', 'detections']

    def __init__(self, data, *args, **kwargs):
        self._init_vars()
        super().__init__(data, *args, **kwargs)

        if self.image_path:
            self.directory, self.filename = os.path.split(self.image_path)
            self.ext = os.path.splitext(self.image_path)[1]
        else:
            logging.warning(f"Place({self.id}) has no image_path")

    def _init_vars(self):
        self.id = None
        self.image_path = None
        self.pose = None
        self.directory = None
        self.filename = None
        self.ext = None


class Places(Datas):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        repr = f"Places(#{len(self)}):\n"
        for id, place in self.items():
            repr += f"  > [{id}] {place}\n"
        return repr

    def update(self, scene_graph, dir=None):
        try:
            # ROS version
            for id, data in scene_graph.nodes.items():
                if 'place' in id:
                    if not 'image_path' in data.keys():
                        logging.warning(f"image_path is not in data: {data}")
                    if not "id" in data:
                        data['id'] = parse_id(id)
                    self[parse_id(id)] = Place(data)
        except:
            # Python version
            for data in scene_graph.get('nodes', []):
                id = data.get('id')
                if 'place' in id:
                    if not 'image_path' in data.keys():
                        logging.warning(f"image_path is not in data: {data}") # TODO: BUG
                        continue
                    if not "id" in data:
                        data['id'] = parse_id(id)
                    if dir: # For debug
                        data['image_path'] = data['image_path'].replace("/ws/external/keyframes", dir)
                    self[parse_id(id)] = Place(data)


if __name__ == "__main__":
    pass
