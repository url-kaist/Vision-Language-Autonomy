class Entity:
    def __init__(self, target_name):
        self.target_name = target_name


class Subtask:
    def __init__(self, action, target_name):
        self.action = action
        self.entity = Entity(target_name)
