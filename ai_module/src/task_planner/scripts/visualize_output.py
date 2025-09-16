import networkx as nx
import matplotlib.pyplot as plt
import os
import textwrap

from utils import *

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

def visualize_relations(data, save_path):
    G = nx.DiGraph()
    labels = {}
    colors = []
    node_sizes = []
    subtask_labels = {}
    added_nodes = set()
    edge_labels = {}
    edge_offset = {}
    multi_edges = set()
    
    # Extract question
    question = data["question"]
    
    # Extract entities
    for entity in data["entities"]:
        node_id = entity["id"]
        
        # Format node label
        if entity["shape"] == "":
            entity["shape"] = "None"
        if entity["color"] == "":
            entity["color"] = "None"
            
        node_label = f"{node_id}: {entity['type']}\n({entity['shape']}, {entity['color']})"
        labels[node_id] = node_label
        
        # Default node color
        node_color = "lightgray"
        
        # Check if entity is in subtasks
        for i, subtask in enumerate(data["positive_subtasks"], start=1):
            if subtask["entity_id"] == node_id:
                node_color = "#87CEFA"  # Light Blue
                subtask_labels[node_id] = str(i)  # Label sequence for positive subtasks
        
        for subtask in data["negative_subtasks"]:
            if subtask["entity_id"] == node_id:
                node_color = "#FF6347"  # Tomato Red
        
        if node_id not in added_nodes:
            G.add_node(node_id)
            colors.append(node_color)
            node_sizes.append(2200)  # Increased node size
            added_nodes.add(node_id)
    
    # Add relations with offset for bidirectional edges
    for entity in data["entities"]:
        node_id = entity["id"]
        for relation in entity["relations"]:
            for target_id in relation["entity_id"]:
                edge_pair = (node_id, target_id)
                reverse_edge_pair = (target_id, node_id)
                
                if reverse_edge_pair in edge_labels:
                    offset = 0.15  # Offset bidirectional edges to prevent overlap
                else:
                    offset = 0.0
                
                G.add_edge(node_id, target_id)
                edge_labels[edge_pair] = (f"{node_id} → {target_id}: {relation['relation']}", offset)
    
    # Draw graph
    plt.figure(figsize=(12, 8))
    plt.title(f"Question: {question}", fontsize=12, fontweight="bold", pad=20, wrap=True)
    pos = nx.spring_layout(G, seed=42, k=1.2)
    
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, edge_color="black", font_size=9, font_weight="bold", alpha=0.9)
    
    # Draw edge labels with offset for bidirectional edges
    for (source, target), (relation_text, offset) in edge_labels.items():
        edge_x, edge_y = (pos[source][0] + pos[target][0]) / 2, (pos[source][1] + pos[target][1]) / 2 + offset
        plt.text(edge_x, edge_y, relation_text, fontsize=10, fontweight="bold", color="black", ha="center", va="center", bbox=dict(facecolor="white", edgecolor="black"))
    
    # Add positive subtask labels
    for node, label in subtask_labels.items():
        x, y = pos[node]
        plt.text(x, y + 0.1, label, fontsize=12, fontweight="bold", color="white", ha="center", va="center", bbox=dict(facecolor="#87CEFA", edgecolor="black", boxstyle="circle"))
    
    plt.savefig(save_path, bbox_inches="tight")  # Ensures long text does not get cut off
    plt.close()
    print(f"Visualization saved at {save_path}")


def visualize_graph(graph, save_path="/ws/external/vis/temp_graph.png", node_color="lightgray", title=""):
    G = nx.DiGraph()
    labels = {} # key=node.id; value={str}description

    colors = []
    node_sizes = []
    subtask_labels = {}
    added_nodes = set()
    edge_labels = {}
    edge_offset = {}
    multi_edges = set()

    for node_id, node in graph.nodes.items():
        labels[node_id] = textwrap.fill(f"{node.name}({node.id})", width=15)
        if node_id not in added_nodes:
            G.add_node(node_id)
            colors.append(node_color)
            node_sizes.append(2200)  # Increased node size
            added_nodes.add(node_id)

    for edge in graph.edges:
        edge_pair = (edge.from_id, edge.to_id)
        reverse_edge_pair = (edge.to_id, edge.from_id)
        if reverse_edge_pair in edge_labels:
            offset = 0.15  # Offset bidirectional edges to prevent overlap
        else:
            offset = 0.0

        G.add_edge(edge.from_id, edge.to_id)
        relation = edge.name if edge.name is not None else f"{edge.weight:.2f}"
        edge_labels[edge_pair] = (f"{edge.from_id} → {edge.to_id}: {relation}", offset)

    # Draw graph
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=12, fontweight="bold", pad=20, wrap=True)
    pos = nx.spring_layout(G, seed=42, k=1.2)

    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, node_size=node_sizes, edge_color="black",
            font_size=9, font_weight="bold", alpha=0.9)

    # Draw edge labels with offset for bidirectional edges
    for (source, target), (relation_text, offset) in edge_labels.items():
        edge_x, edge_y = (pos[source][0] + pos[target][0]) / 2, (pos[source][1] + pos[target][1]) / 2 + offset
        plt.text(edge_x, edge_y, relation_text, fontsize=10, fontweight="bold", color="black", ha="center", va="center",
                 bbox=dict(facecolor="white", edgecolor="black"))

    # Add positive subtask labels
    for node, label in subtask_labels.items():
        x, y = pos[node]
        plt.text(x, y + 0.1, label, fontsize=12, fontweight="bold", color="white", ha="center", va="center",
                 bbox=dict(facecolor="#87CEFA", edgecolor="black", boxstyle="circle"))

    plt.savefig(save_path, bbox_inches="tight")  # Ensures long text does not get cut off
    plt.close()
    print(f"Visualization saved at {save_path}")



if __name__ == "__main__":
    input_folder = os.path.join(PARENT_DIR, "output")
    output_folder = os.path.join(PARENT_DIR, "visualization")
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_folder, file_name)
            save_path = os.path.join(output_folder, f"{file_name.replace('.json', '.png')}")
            
            data = load_data_from_json(file_path)
            visualize_relations(data, save_path)
