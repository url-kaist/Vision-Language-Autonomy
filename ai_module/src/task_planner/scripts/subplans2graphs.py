import json
try:
    import networkx as nx
except:
    import os
    os.system("sudo apt-get install python3-pip")
    os.system("pip install networkx")
    import networkx as nx
import matplotlib.pyplot as plt
import textwrap


def build_graph(data, default_color='gray'):
    G = nx.DiGraph()

    # Add nodes
    for obj in data['objects']:
        color = obj.get('color', default_color) or default_color
        G.add_node(obj['id'], label=obj['type'], color=color, label_id=f"{obj['type']}({obj['id']})")

    # Add edges
    for obj in data['objects']:
        for relation in obj['relations']:
            for target_id in relation['object_id']:
                G.add_edge(obj['id'], target_id, label=relation['relation'])

    return G


def extract_subgraph(graph, object_id):
    subgraph_nodes = set()

    # BFS to collect all connected nodes to object_id
    queue = [object_id]
    while queue:
        current = queue.pop(0)
        if current not in subgraph_nodes:
            subgraph_nodes.add(current)
            queue.extend(graph.successors(current))
            queue.extend(graph.predecessors(current))

    subgraph = graph.subgraph(subgraph_nodes).copy()
    return subgraph


def wrap_title(title, width=40, char_per_inch=10):
    max_chars = max(int(width * char_per_inch), 20)
    return '\n'.join(textwrap.wrap(title, max_chars))


def draw_graph(graph, title='', filename='graph', default_color='gray'):
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label_id')
    colors = [graph.nodes[n].get('color', default_color) for n in graph.nodes]

    fig = plt.figure(figsize=(8, 6))
    fig_width, _ = fig.get_size_inches()
    plt.title(wrap_title(title, fig_width), fontsize=14)
    nx.draw(graph, pos, with_labels=True, labels=labels, edge_color='black', node_color=colors, node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    # Save the image
    plt.tight_layout()
    plt.savefig(f'/ws/external/vis/rag/subplans2graphs/{filename}.png')
    plt.clf()


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' is not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An expected error occured: {e}")


def parse_plans(file_path, vis=False, name=''):
    data = read_json(file_path)
    print(f"instruction: {data['question']}")
    chunks = data['question'].replace(',', '//').replace('and', '//').split('//')

    graph = build_graph(data)
    if vis:
        draw_graph(graph, title=data['question'], filename=f'{name}graph')

    subtask_list = []
    for i, subtask in enumerate(data.get('positive_subtasks', [])):
        object_id = subtask['object_id']
        subgraph = extract_subgraph(graph, object_id)
        if vis:
            title = chunks[i] #  if len(chunks) == len(data['positive_subtasks']) else ''
            draw_graph(subgraph, title=title, filename=f"{name}subgraph{i}-{subtask['action']}-{object_id}")

        subtask['subgraph'] = subgraph
        subtask_list.append(subtask)

    return subtask_list


if __name__ == '__main__':
    subtasks1 = parse_plans("../output/task1.json", vis=True, name='task1-')
    subtasks2 = parse_plans("../output/task2.json", vis=True, name='task2-')
    subtasks3 = parse_plans("../output/task3.json", vis=True, name='task3-')
