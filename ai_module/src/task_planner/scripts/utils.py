from __future__ import annotations

import os
import sys
sys.path.append('/ws/external')
import json
import rospy

from task_planner.msg import Plan as PlanMsg
from task_planner.msg import Task as TaskMsg
from task_planner.msg import Entity as EntityMsg
from task_planner.msg import Graph as GraphMsg
from task_planner.msg import Node as NodeMsg
from task_planner.msg import Edge as EdgeMsg
from task_planner.msg import KeyValue


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Node:
    def __init__(self, id: int, name: str, is_target: bool = False, attr: dict = None, exclude_attr: dict = None):
        self.id = id                            # unique id
        self.name = name                        # example: 'bed', 'picture'
        self.is_target = is_target              # True or False
        self.attr = attr or {}                  # example: {'color': ['red'], 'shape': ['round'], 'subject': ['butterfly']}
        self.exclude_attr = exclude_attr or {}

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'is_target': self.is_target,
            'attr': self.attr,
            'exclude_attr': self.exclude_attr,
        }

    @staticmethod
    def dict_to_kv_list(d):
        if not isinstance(d, dict):
            rospy.logerr(f"[Node.dict_to_kv_list] Expected dict, got: {type(d)}")
            raise TypeError("Attributes must be a dict of str to list of str")
        kv_list = []
        for k, v in d.items():
            if not isinstance(k, str) or not isinstance(v, list):
                rospy.logerr(f"[Node.dict_to_kv_list] Invalid key/value in attr: {k} -> {v}")
                raise TypeError("Each key must be str and value must be list of str")
            kv_list.append(KeyValue(key=k, value=v))
        return kv_list

    def to_msg(self):
        if not isinstance(self.id, int):
            rospy.logerr(f"[Node.to_msg] id is not int: {self.id}")
            raise TypeError("Node.id must be an int")
        if not isinstance(self.name, str):
            rospy.logerr(f"[Node.to_msg] name is not str: {self.name}")
            raise TypeError("Node.name must be a str")
        if not isinstance(self.is_target, bool):
            rospy.logerr(f"[Node.to_msg] is_target is not bool: {self.is_target}")
            raise TypeError("Node.is_target must be a bool")

        return NodeMsg(
            id=self.id, name=self.name, is_target=self.is_target,
            attr=self.dict_to_kv_list(self.attr),
            exclude_attr=self.dict_to_kv_list(self.exclude_attr))


class Edge:
    def __init__(self, name: str, source_id: int, target_ids: list[int]):
        self.name = name                # 'under'
        self.source_id = source_id      # unique id of source node
        self.target_ids = target_ids    # list of target ids

    def to_dict(self):
        return {'name': self.name, 'source_id': self.source_id, 'target_ids': self.target_ids}

    def to_msg(self):
        if not isinstance(self.name, str):
            rospy.logerr(f"[Edge.to_msg] name is not str: {self.name}")
            raise TypeError("Edge.name must be a str")
        if not isinstance(self.source_id, int):
            rospy.logerr(f"[Edge.to_msg] source_id is not int: {self.source_id}")
            raise TypeError("Edge.source_id must be an int")
        if not isinstance(self.target_ids, list) or not all(isinstance(i, int) for i in self.target_ids):
            rospy.logerr(f"[Edge.to_msg] target_ids is not list[int]: {self.target_ids}")
            raise TypeError("Edge.target_ids must be a list of int")

        return EdgeMsg(name=self.name, source_id=self.source_id, target_ids=self.target_ids)


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def to_msg(self):
        try:
            node_msgs = [node.to_msg() for node in self.nodes]
            edge_msgs = [edge.to_msg() for edge in self.edges]
        except Exception as e:
            rospy.logerr(f"[Graph.to_msg] Error converting nodes/edges: {e}")
            raise
        return GraphMsg(nodes=node_msgs, edges=edge_msgs)


class Entity:
    def __init__(self, target_name: str):
        self.target_name = target_name  # example: 'the bed under the butterfly picture'
        self.relation_graph = Graph()

    def add_node(self, node: Node):
        self.relation_graph.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.relation_graph.edges.append(edge)

    def __repr__(self):
        return self.target_name

    def to_dict(self):
        return {
            'target_name': self.target_name,
            'relation_graph': {
                'nodes': [node.to_dict() for node in self.relation_graph.nodes],
                'edges': [edge.to_dict() for edge in self.relation_graph.edges]
            }
        }

    def to_msg(self):
        if not isinstance(self.target_name, str):
            rospy.logerr(f"[Entity.to_msg] target_name is not str: {self.target_name}")
            raise TypeError("Entity.target_name must be a str")
        return EntityMsg(target_name=self.target_name, relation_graph=self.relation_graph.to_msg())


class Task:
    def __init__(self, action: str, entity: Entity):
        assert action in ['find', 'count', 'goto', 'avoid', 'stop']
        self.action = action    # action
        self.entity = entity    # entity

    def __repr__(self):
        return f"action: {self.action}, entity: {self.entity}"

    def to_dict(self):
        return {'action': self.action, 'entity': self.entity.to_dict()}

    def to_msg(self):
        if not isinstance(self.action, str):
            rospy.logerr(f"[Task.to_msg] action is not str: {self.action}")
            raise TypeError("Task.action must be a str")
        if not isinstance(self.entity, Entity):
            rospy.logerr(f"[Task.to_msg] entity is not Entity: {self.entity}")
            raise TypeError("Task.entity must be an Entity")

        return TaskMsg(action=self.action, entity=self.entity.to_msg())


class Plan:
    def __init__(self):
        self.constraints = []    # always-valid constraints (e.g., avoid zones)
        self.steps = []         # ordered list of executable tasks (goto, find, count, etc.)

    def __repr__(self):
        out = "\n"
        out += f"--- constraints ({len(self.constraints)}) ---\n"
        for constraint in self.constraints:
            out += f"   {constraint}\n"
        out += f"--- steps ({len(self.steps)}) ---\n"
        for step in self.steps:
            out += f"   {step}\n"
        return out

    def to_msg(self):
        try:
            constraints_msg = [c.to_msg() for c in self.constraints]
            steps_msg = [s.to_msg() for s in self.steps]
        except Exception as e:
            rospy.logerr(f"[Plan.to_msg] Error converting constraints or steps: {e}")
            raise

        return PlanMsg(constraints=constraints_msg, steps=steps_msg)


# to make prompt
def get_prompt(task):
    prompt = ""
    # read the prompt format from the file
    with open(os.path.join(CURRENT_DIR, "prompt.txt"), "r") as f:
        prompt += f.read()

    # add task information to the prompt
    prompt += f"\nTODO: complete the following function\n"
    prompt += f"def create_subtasks_by_LLM():\n"
    prompt += f"\tquestion = \"{task}\"\n\n"
    prompt += f"\t# Extract entities from the question\n"
    prompt += f"\t# - Use full names for compound nouns (e.g., 'coffee table')\n"
    prompt += f"\t# - Assign unique node IDs and set attributes or exclude_attr if needed\n\n"
    prompt += f"\t# Define spatial relations as edges and add nodes/edges to each entity\n\n"
    prompt += f"\t# For each entity, create a Task and add it to plan.steps or plan.constraints\n\n"
    prompt += f"\treturn plan\n"
    return prompt


def save_json(data, filename):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_data_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data
