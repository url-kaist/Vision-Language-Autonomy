# Complete the create_subtasks_by_LLM function

# relation examples: on, near, between, in, under, above, below, next to, in front of, behind, close to, far from, inside, etc.

# Extract all entities mentioned in the question. Follow these rules strictly:
# 1. Each occurrence of an entity should be treated as a separate instance.  
#    - Example: If "cabinet" appears twice, create `cabinet_1` and `cabinet_2`.
# 2. Each entity should have a unique ID and preserve the attributes mentioned (type, color, shape, etc.).  
#    - Example: "red table" and "table" should be treated as separate entities if both appear.
# 3. Define spatial relationships properly.  
#    - Example: `"folder on the cabinet"` should be encoded as `folder_1` has a relation `"on"` with `cabinet_2`.

from actions import goto <obj>, count <obj>, find <obj>, avoid <obj>

class Entity:
    def __init__(self, id: int, type: str, shape: str = "", color: str = ""):
        self.id = id          # unique id
        self.type = type      # example: window, fridge, table, trash can, path, etc.
        self.shape = shape    # example: big, small, rectangle, circle, square, etc.
        self.color = color    # example: red, blue, green, etc.
        self.relations = []

    def add_relation(self, relation: str, entities: list["Entity"]):
        self.relations.append([relation, entities]) # example: ("near", ["fridge"]), ("between", ["table", "picture"])

class SubTask:
    def __init__(self, id: int, action: str, target):
        self.id = id            # unique id
        self.action = action    # example: goto, avoid
        self.target = target    # entity

class PositiveSubTask(SubTask):
    def __init__(self, id: int, action: str, target):
        super().__init__(id, action, target)
        # action: goto, count, find
        pass

class NegativeSubTask(SubTask):
    def __init__(self, id: int, action: str, target):
        super().__init__(id, action, target)
        # action: avoid
        pass

def goto_example1():
    question = "Take the path between the TV and the coffee table and go to the kettle on the dining table, then go to the white potted plant between the curtain and the TV"
 
    # create entities in order, treating each occurrence as a separate instance with unique ID and attributes. 
    entities = [
        Entity(id=0, type="path"),
        Entity(id=1, type="tv"), # tv_1
        Entity(id=2, type="coffee table"),
        Entity(id=3, type="kettle"),
        Entity(id=4, type="dining table"),
        Entity(id=5, type="potted plant", color="white"),
        Entity(id=6, type="curtain"),
        Entity(id=7, type="tv") # tv_2
    ]
    
    # add relations
    entities[0].add_relation("between", [entities[1], entities[2]])
    entities[3].add_relation("on", [entities[4]])
    entities[5].add_relation("between", [entities[6], entities[7]])

    # create subtasks
    positive_subtasks = [
        PositiveSubTask(id=i, action="goto", target=entities[t])
        for i, t in enumerate([0, 3, 5])
    ]
    negative_subtasks = []

    return positive_subtasks, negative_subtasks, entities

def goto_example2():
    question = "First, go near the tea table with the elephant figurine on it, then stop at the red tea table with the horse figurine on it, avoiding between the big chair and the folding screen."

    # create entities in order, treating each occurrence as a separate instance with unique ID and attributes. 
    entities = [
        Entity(type="tea table"),
        Entity(type="elephant figurine"),
        Entity(type="tea table", color="red"),
        Entity(type="horse figurine"),
        Entity(type="path"), # implicit path between big chair and folding screen
        Entity(type="chair", shape="big"),
        Entity(type="folding screen")
    ]

    # add relations
    entities[1].add_relation("on", [entities[0]])
    entities[3].add_relation("on", [entities[2]])
    entities[4].add_relation("between", [entities[5], entities[6]])

    # create subtasks
    positive_subtasks = [PositiveSubTask(action="goto", target=entities[t]) for t in [0, 2]]
    negative_subtasks = [NegativeSubTask(action="avoid", target=entities[4])]

    return positive_subtasks, negative_subtasks, entities

def count_example():
    question = "How many blue chairs are between the table and the wall?"

    # create entities in order, treating each occurrence as a separate instance with unique ID and attributes. 
    entities = [
        Entity(type="chair", color="blue"),
        Entity(type="table"),
        Entity(type="wall")
    ]

    # add relations
    entities[0].add_relation("between", [entities[1], entities[2]])

    # create subtasks
    positive_subtasks = [PositiveSubTask(action="count", target=entities[0])]
    negative_subtasks = []

    return positive_subtasks, negative_subtasks, entities

def find_example():
    question = "Find the potted plant on the kitchen island that is closest to the fridge."

    # Create entities in order, treating each occurrence as a separate instance with unique ID and attributes. 
    entities = [
        Entity(type="potted plant"),
        Entity(type="kitchen island"),
        Entity(type="fridge")
    ]

    # add relations
    entities[0].add_relation("on", [entities[1]])
    entities[0].add_relation("closest to", [entities[2]])

    # create subtasks
    positive_subtasks = [PositiveSubTask(action="find", target=entities[0])]
    negative_subtasks = []

    return positive_subtasks, negative_subtasks, entities


def create_subtasks_by_LLM():
    question = "Go near the magazine on the ottoman, then go to the potted plant on the dressing table."
    quesiton = "Go between the bench and the bed and stop at the lamp closest to the fireplace."
    question = "The lantern between the vase and the stone decoration that is closest to the vase."
    question = "First, go near the lamp closest to the black chair, then take the path between the sofa and the round tables, and stop at the cabinet with a picture above it."
    question = "First, go to the trash can near the cabinet, then go to the folder on the cabinet closest to the whiteboard, and finally, to the door near the exit sign."
    question = "First, go to the vase closest to the easel, then, take the path between the couch and the table and stop at the window closest to the couch."
    quesiton = "First, go near the bedside table closest to the bench, then take the path between the TV and the bed to the picture closest to the TV."
        
    # create entities in order, treating each occurrence as a separate instance with unique ID and attributes. 

    # add relations

    # create subtasks

    # return positive_subtasks, negative_subtasks, entities

if __name__ == "__main__":
    create_subtasks_by_LLM()