from ai_module.src.visual_grounding.scripts.utils.utils_message import object_to_marker
from ai_module.src.visual_grounding.scripts.structures.keyframe import Keyframe, Keyframes
try:
    from std_msgs.msg import String, Int32
except:
    pass


class Answer:
    def __init__(self, data=None, **kwargs):
        self.count = None
        self.object = None
        pids, eids = [], []
        if data:
            keyframes = data['keyframes']
            if isinstance(keyframes, Keyframe):
                pids += [keyframes.id]
                eids += keyframes.entities['candidate'].ids
            else:  # Keyframes
                for pid, kf in keyframes.items():
                    pids += [pid]
                    eids += kf.entities['candidate'].ids
        self.pids = tuple(pids)
        self.eids = tuple(eids)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        if self.count is not None:
            args = self.count
        elif self.object is not None:
            args = self.object.id
        else:
            args = "?"
        return f"{self.__class__.__name__}({args})"

    def __str__(self):
        if self.count is not None:
            return str(self.count)
        elif self.object is not None:
            return str(self.object.id)
        else:
            return "?"

    def __eq__(self, other):
        return isinstance(other, Answer) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def get_answer_msg(self, action):
        if action in ['find']:
            return object_to_marker(self.object, self.object.id, color=(0.0, 0.0, 1.0, 1.0), style='cube')
        elif action in ['count']:
            return Int32(self.count)
        else:
            return String("")

    def get_answer(self, action, **kwargs):
        if action in ['find']:
            return self.object.id
        elif action in ['count']:
            return str(self.count)
        else:
            object_ids = self.objects.ids
            return object_ids

    def get_answer_vis_msg(self, action, color=(0.0, 0.0, 1.0, 1.0), style='box'):
        if action in ['find']:
            marker = object_to_marker(self.object, self.object.id, color=color, style=style)
            return [marker]
        elif action in ['count']:
            return []
        else:
            markers = []
            for object_id, object in self.objects.items():
                marker = object_to_marker(object, object_id, color=color, style=style)
                markers.append(marker)
            return markers
