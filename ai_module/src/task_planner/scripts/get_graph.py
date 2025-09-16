import os
import json
import copy
import gzip
import pickle
import math
import torch
import numpy as np
from collections import deque
try:
    import clip
except:
    os.system('pip install --no-deps git+https://github.com/openai/CLIP.git')
    import clip
from visualize_output import visualize_graph
from graph_retriever import G_Retriever

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


class Geometry:
    def __init__(self, bbox_extent=None, bbox_center=None, bbox_volume=None, **kwargs):
        self.extent = bbox_extent
        self.center = bbox_center
        self.volume = bbox_volume

    def __repr__(self):
        return f"geometry(extent={self.extent}, center={self.center}, volume={self.volume})"

class Node:
    def __init__(self, id, name=None, img=None, geometry=None,
                 textual_feature=None, visual_feature=None, **kwargs):
        self._id = id
        self._visual_feature = visual_feature
        self._textual_feature = textual_feature
        self.edges = []

        self.name = name
        self.img = img
        self.geometry = geometry

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            assert isinstance(name, str), "name should be a string."

            text = clip.tokenize([name]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy()
            self._textual_feature = np.squeeze(text_features)
        self._name = name

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).to(device)

            assert isinstance(img, torch.Tensor), 'img should be a torch.Tensor or a np.ndarray.'
            img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(img)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()
            self._visual_features = image_features
        self._img = img

    @property
    def visual_feature(self):
        return self._visual_feature

    @visual_feature.setter
    def visual_feature(self, visual_feature):
        if isinstance(visual_feature, torch.Tensor):
            visual_feature = visual_feature.cpu().detach().numpy()
        assert isinstance(visual_feature, np.ndarray), 'visual_feature should be a np.ndarray.'
        self._visual_feature = np.squeeze(visual_feature)

    @property
    def textual_feature(self):
        return self._textual_feature

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        if geometry is not None:
            if isinstance(geometry, Geometry):
                self._geometry = geometry
            elif isinstance(geometry, dict):
                self._geometry = Geometry(**geometry)
            else:
                raise TypeError('geometry should be dict or Geometry.')
        else:
            self._geometry = geometry

    def add_edge(self, edge):
        self.edges.append(edge)

    def __repr__(self):
        output = f"Node ("
        output_list = []
        for k, v in self.__dict__.items():
            k = k.lstrip("_")
            if v is None:
                continue
            elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                output_list.append(f"{k}={v.shape}")
            elif isinstance(v, list):
                output_list.append(f"#{k}={len(v)}")
            else:
                output_list.append(f"{k}={v}")
        for _i, o in enumerate(output_list):
            output += o
            if _i < len(output_list) - 1:
                output += ", "
        output += ")"
        return output

class Edge:
    def __init__(self, from_id, to_id, name=None, weight=None):
        self.from_id = from_id
        self.to_id = to_id
        self.name = name
        self.weight = weight

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            assert isinstance(name, str), "name should be a string."

            text = clip.tokenize([name]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy()
            self._textual_feature = np.squeeze(text_features)
        self._name = name

    def __repr__(self):
        output = f"Edge ("
        output_list = []
        for k, v in self.__dict__.items():
            k = k.lstrip("_")
            if v is None:
                continue
            elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                output_list.append(f"{k}={v.shape}")
            elif isinstance(v, list):
                output_list.append(f"#{k}={len(v)}")
            else:
                output_list.append(f"{k}={v}")
        for _i, o in enumerate(output_list):
            output += o
            if _i < len(output_list) - 1:
                output += ", "
        output += ")"
        return output

class Graph:
    node_attributes = ['name', 'img', 'visual_feature', 'textual_feature', 'geometry']
    def __init__(self, directed=False):
        self.nodes = {}
        self.edges = []
        self.directed = directed

    def add_vertex(self, new_node):
        is_success = True
        unexpected_ids = []

        if new_node.id not in self.nodes.keys():
            self.nodes[new_node.id] = new_node
        else:
            existing_node = self.nodes[new_node.id]
            for attr in self.node_attributes:
                existing_node_data = getattr(existing_node, attr)
                new_node_data = getattr(new_node, attr)
                if existing_node_data is not None:
                    if new_node_data is not None:
                        # Both data should be same
                        if isinstance(existing_node_data, torch.Tensor):
                            if not torch.equal(existing_node_data, new_node_data):
                                is_success = False
                                unexpected_ids.append(new_node.id)
                        elif isinstance(existing_node_data, np.ndarray):
                            if not np.array_equal(existing_node_data, new_node_data):
                                is_success = False
                                unexpected_ids.append(new_node.id)
                        elif existing_node_data != new_node_data:
                            is_success = False
                            unexpected_ids.append(new_node.id)
                elif new_node_data is not None:
                    setattr(self.nodes[new_node.id], attr, new_node_data)
        if not is_success:
            assert TypeError
        return is_success, unexpected_ids

    def add_edge(self, from_node, to_node, **kwargs):
        self.add_vertex(from_node)
        self.add_vertex(to_node)
        edge = Edge(from_node.id, to_node.id, **kwargs)
        self.edges.append(edge)
        self.nodes[from_node.id].add_edge(edge)
        if not self.directed:
            self.nodes[to_node.id].add_edge(edge)

    def merge(self, other_subgraph):
        self.nodes.update(other_subgraph.nodes)
        self.edges.extend(other_subgraph.edges)

    def display(self):
        self.__repr__()
        node_keys = []
        for edge in self.edges:
            edge_display = edge.name if edge.name is not None else edge.weight
            print(f"{self.nodes[edge.from_id]} \033[32m--{edge_display}-->\033[0m {self.nodes[edge.to_id]}")
            node_keys.append(edge.from_id)
            node_keys.append(edge.to_id)
        for node_id, node in self.nodes.items():
            if not node_id in node_keys:
                print(f"{node}")

    def remove_edge(self, from_node_id, to_node_id):
        # Find and remove the edge
        self.edges = [edge for edge in self.edges if not (edge.from_id == from_node_id and edge.to_id == to_node_id)]

        # Remove the edge from the nodes if they exist
        from_node = self.nodes.get(from_node_id, None)
        if from_node is not None:
            from_node.edges = [edge for edge in from_node.edges if edge.to_id != to_node_id]

        to_node = self.nodes.get(to_node_id, None)
        if (not self.directed) and (to_node is not None):
            to_node.edges = [edge for edge in to_node.edges if edge.from_id != from_node_id]

    def get_k_hop_subgraph(self, start_node_id, k):
        if start_node_id not in self.nodes:
            raise ValueError(f"Node {start_node_id} not found in the graph.")

        # BFS to find k-hop neighbors
        visited = set()
        queue = deque([(start_node_id, 0)])  # (node_id, current_hop)
        subgraph_nodes = set()
        subgraph_edges = []

        while queue:
            node_id, hop = queue.popleft()
            if node_id in visited or hop > k:
                continue
            visited.add(node_id)
            subgraph_nodes.add(node_id)

            for edge in self.nodes[node_id].edges:
                neighbor_id = edge.to_id if edge.from_id == node_id else edge.from_id
                if neighbor_id not in visited and hop < k:
                    queue.append((neighbor_id, hop + 1))
                if neighbor_id in subgraph_nodes:
                    subgraph_edges.append(edge)

        # Create a new subgraph
        subgraph = Graph(directed=self.directed)
        for node_id in subgraph_nodes:
            subgraph.add_vertex(self.nodes[node_id])
        for edge in subgraph_edges:
            subgraph.add_edge(self.nodes[edge.from_id], self.nodes[edge.to_id], name=edge.name, weight=edge.weight)

        return subgraph

    def to_json(self, save_clip_feature=False):
        graph_dict = {
            "directed": self.directed,
            "nodes": [],
            "edges": []
        }

        # Add node
        for node_id, node in self.nodes.items():
            node_info = {
                "id": node.id,
                "name": node.name,
                "geometry": {
                    "extent": node.geometry.extent,
                    "center": node.geometry.center,
                    "volume": node.geometry.volume
                } if node.geometry else None
            }

            if save_clip_feature:
                node_info["visual_feature"] = node.visual_feature.tolist() if node.visual_feature is not None else None
                node_info["textual_feature"] = node.textual_feature.tolist() if node.textual_feature is not None else None

            graph_dict["nodes"].append(node_info)

        # Add edge
        for edge in self.edges:
            edge_info = {
                "from_id": edge.from_id,
                "to_id": edge.to_id,
                "name": edge.name,
                "weight": edge.weight
            }
            graph_dict["edges"].append(edge_info)

        return json.dumps(graph_dict, indent=2)

    def save_to_json_file(self, file_path):
        with open(file_path, "w") as f:
            f.write(self.to_json())

    def __repr__(self):
        return f"Graph (#nodes={len(self.nodes.keys())}, #edges={len(self.edges)}, directed={self.directed})"


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


if __name__ == '__main__':
    """ Subplans -> Query Graph """
    # Choose file index
    FILE_INDEX = 0
    DATA_DIR = '/ws/external/ai_module/src/task_planner/output'

    # filenames = [filename for filename in os.listdir(DATA_DIR) if filename.endswith('.json')]
    # file_path = os.path.join(DATA_DIR, filenames[FILE_INDEX])
    file_path = os.path.join(DATA_DIR, 'saved_answers.json')
    with open(file_path, 'r') as file:
        # Load .json question
        data = json.load(file)
        if isinstance(data, list):
            data = data[FILE_INDEX]

        # Get question
        question = data['question']
        positive_subtasks = data['positive_subtasks']
        negative_subtasks = data['negative_subtasks']
        print(f"Question: {question}")

        # Construct graph from .json question
        graph_query = Graph(directed=True)
        for entity in data['entities']:
            entity['name'] = f"{entity['color']} {entity['type']}" if entity['color'] else entity['type']
            graph_query.add_vertex(Node(**entity))
            for relation in entity['relations']:
                for entity_id in relation['entity_id']:
                    graph_query.add_edge(Node(**entity), Node(entity_id), name=relation['relation'])

        # Visualize
        graph_query.display()

    """ 3D-SG -> Observation Graph """
    # Choose file index
    OBS_DATA_DIR = '/ws/data/Replica/room0/exps/r_mapping_stride10'
    MAKE_GEO_EDGE = True
    DISTANCE_THRESHOLD = 0.6

    graph_obs = Graph(directed=True)
    config_params, config_params_detections = None, None

    obs_filenames = [filename for filename in os.listdir(OBS_DATA_DIR)]
    for filename in obs_filenames:
        file_path = os.path.join(OBS_DATA_DIR, filename)
        if filename.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
        elif filename.endswith('.pkl.gz'):
            with gzip.open(file_path, 'rb') as file:
                data = pickle.load(file)
        else:
            raise NotImplemenetedError

        if filename.startswith('obj_json'):
            # Objects to Node
            data_obj = copy.deepcopy(data)
            print(f"Object: #objs={len(data_obj)}")
            for k, obj in data_obj.items():
                obj['name'] = f"{obj['object_caption']} {obj['object_tag']}" \
                    if obj['object_caption'] is not None else obj['object_tag']
                obj['geometry'] = Geometry(**obj)
                _ = graph_obs.add_vertex(Node(**obj))
        elif filename.startswith('edge_json'):
            # Edges to Edge
            print(f"Edges: #objs={len(data)}")
        elif filename.startswith('pcd'):
            data_pcd = copy.deepcopy(data)
            print(f"Pcd:")
            print(f" - #objs={len(data_pcd['objects'])}")
            for _obj_data, _obj_obs in zip(data_obj.values(), data_pcd['objects']):
                node_args = {
                    'id': _obj_data['id'],
                    'name': _obj_obs['class_name'],
                    'visual_feature': _obj_obs['clip_ft'],
                }
                _ = graph_obs.add_vertex(Node(**node_args))
        elif filename.startswith('config_params_detections'):
            print(f"Detection Configuration Parameters")
            config_params_detections = copy.deepcopy(data)
        elif filename.startswith('config_params'):
            print(f"Configuration Parameters")
            config_params = copy.deepcopy(data)

    """ Geometry Edges """
    for from_node_id, from_node in graph_obs.nodes.items():
        for to_node_id, to_node in graph_obs.nodes.items():
            if ((from_node_id == to_node_id) or
                    (from_node.geometry is None) or
                    (to_node.geometry is None)):
                continue
            distance = euclidean_distance(from_node.geometry.center, to_node.geometry.center)
            if distance < DISTANCE_THRESHOLD:
                graph_obs.add_edge(from_node, to_node, weight=distance)
            else:
                graph_obs.remove_edge(from_node_id, to_node_id)
    graph_obs.display()

    # create folder
    vis_dir = '/ws/external/vis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Directory {vis_dir} created.")

    visualize_graph(graph_query, save_path=os.path.join(vis_dir, 'graph_query.png'))
    visualize_graph(graph_obs, save_path=os.path.join(vis_dir, 'graph_obs.png'))

    """ Similarity """
    TOP_K = 5

    features_query = np.stack([node.textual_feature for node in graph_query.nodes.values()], axis=0)
    features_key = np.stack([node.visual_feature for node in graph_obs.nodes.values()], axis=0)
    features_query, features_key = torch.Tensor(features_query), torch.Tensor(features_key)
    similarities = torch.matmul(features_query, features_key.t())
    topk_indices = torch.topk(similarities, k=TOP_K, dim=1)

    """ K-hop """
    HOP_K = 2
    candidates = {}
    for i, indices in enumerate(topk_indices.indices):
        id_query = graph_query.nodes[i].id
        candidates[id_query] = []
        topk = 0
        for j in indices.cpu().detach().numpy():
            id_key = [node.id for target_j, node in enumerate(graph_obs.nodes.values()) if target_j == j][0]
            subgraph = graph_obs.get_k_hop_subgraph(id_key, HOP_K)
            visualize_graph(subgraph, title=graph_query.nodes[i].name, save_path=os.path.join(vis_dir, f'subgraph{id_query}-top{topk}-{id_key}.png'))
            candidates[id_query].append(subgraph)
            topk += 1

    """ Query to LLM """
    # Merge all candidate subgraphs into a unified graph
    unified_subgraph = Graph(directed=True)
    for _, graphs in candidates.items():
        for graph in graphs:
            unified_subgraph.merge(graph)
    visualize_graph(unified_subgraph, save_path=os.path.join(vis_dir, 'unified_subgraph.png'))

    # Create graph retriever
    g_retriever = G_Retriever(unified_subgraph)

    # Query the G_Retriever
    subtask = positive_subtasks[0] # TODO: for loop
    ans = g_retriever.retrieve(subtask['action'], subtask['entity_id'], graph_query)

