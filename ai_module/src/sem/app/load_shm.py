import json
from multiprocessing import shared_memory
from networkx.readwrite import json_graph
import networkx as nx

def load_graph_from_shared_memory(shm_name = "scene_graph_shm"):
    try:
        shm = shared_memory.SharedMemory(shm_name)
        raw_data = bytes(shm.buf[:])
        graph_data = json.loads(raw_data.decode("utf-8"))
        G = graph_data
        # print(f"graph_data: {graph_data}")
        # G = node_link_graph(graph_data, link='edges')
        # print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

        return G, shm
    
    except FileNotFoundError:
        print(f"[Error] Shared memory '{shm_name}' not found.")
        return None
    except Exception as e:
        print(f"[Error] Failed to load from shared memory: {e}")
        return None
    
    # finally:
    #     try:
    #         shm.close()
    #         shm.unlink()
    #         print("Shared memory cleaned up.")
    #     except:
    #         pass

from itertools import chain, count


def _to_tuple(x):
    """Converts lists to tuples, including nested lists.

    All other non-list inputs are passed through unmodified. This function is
    intended to be used to convert potentially nested lists from json files
    into valid nodes.

    Examples
    --------
    >>> _to_tuple([1, 2, [3, 4]])
    (1, 2, (3, 4))
    """
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(map(_to_tuple, x))
def node_key(n: dict) -> str:
    return f"{n['level']}:{n['id']}"   # 예: "NodeLevel.PLACE:0"

def node_link_graph(
    data,
    directed=False,
    multigraph=True,
    attrs=None,
    *,
    source="source",
    target="target",
    name="id",
    key="key",
    link="links",
):
    if attrs is not None:
        import warnings

        msg = (
            "\n\nThe `attrs` keyword argument of node_link_graph is deprecated\n"
            "and will be removed in networkx 3.2. It is replaced with explicit\n"
            "keyword arguments: `source`, `target`, `name`, `key` and `link`.\n"
            "To make this warning go away, and ensure usage is forward\n"
            "compatible, replace `attrs` with the keywords. "
            "For example:\n\n"
            "   >>> node_link_graph(data, attrs={'target': 'foo', 'name': 'bar'})\n\n"
            "should instead be written as\n\n"
            "   >>> node_link_graph(data, target='foo', name='bar')\n\n"
            "in networkx 3.2.\n"
            "The default values of the keywords will not change.\n"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        source = attrs.get("source", "source")
        target = attrs.get("target", "target")
        name = attrs.get("name", "name")
        key = attrs.get("key", "key")
        link = attrs.get("link", "links")
    # -------------------------------------------------- #
    multigraph = data.get("multigraph", multigraph)
    directed = data.get("directed", directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()

    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else key
    graph.graph = data.get("graph", {})
    c = count()
    for d in data["nodes"]:
        node = _to_tuple(d.get(name, next(c)))
        nodedata = {str(k): v for k, v in d.items() if k != name}
        graph.add_node(node, **nodedata)
    for d in data[link]:
        src = tuple(d[source]) if isinstance(d[source], list) else d[source]
        tgt = tuple(d[target]) if isinstance(d[target], list) else d[target]
        src = node_key(src)
        tgt = node_key(tgt)
        if not multigraph:
            edgedata = {str(k): v for k, v in d.items() if k != source and k != target}
            graph.add_edge(src, tgt, **edgedata)
        else:
            ky = d.get(key, None)
            edgedata = {
                str(k): v
                for k, v in d.items()
                if k != source and k != target and k != key
            }
            graph.add_edge(src, tgt, ky, **edgedata)
    return graph


        
def load_object_data_from_shared_memory(shm_name="object_shm"):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        
        raw_data = bytes(shm.buf[:]) 
        json_str = raw_data.decode("utf-8") 
        object_data = json.loads(json_str)
        objects = {int(k): v for k, v in object_data.items()}
        # for instance_id, obj in objects.items():
            # print(f"ID {instance_id}, Class ID: {obj['class_id']}, Center: {obj['center']}, Type: {obj['class_name']}")
        # print(f"Loaded {len(objects)} objects.")

        return objects, shm

    except FileNotFoundError:
        print(f"[Error] Shared memory '{shm_name}' not found.")
        return None
    except Exception as e:
        print(f"[Error] Failed to load from shared memory: {e}")
        return None
    # finally:
    #     try:
    #         shm.close()
    #         shm.unlink()
    #         print("Shared memory cleaned up.")
    #     except:
    #         pass

def load_point_cloud_map_from_shared_memory(shm_name="point_cloud_map_shm"):
    import pickle
    from multiprocessing import shared_memory

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        # Read the bytes from shared memory
        data_bytes = bytes(shm.buf)
        # Unpickle to get the numpy array
        point_cloud_map = pickle.loads(data_bytes)
        print(f"Loaded point_cloud_map from shared memory '{shm_name}' with shape {point_cloud_map.shape}")
        return point_cloud_map, shm
    except FileNotFoundError:
        print(f"[Error] Shared memory '{shm_name}' not found.")
        return None
    except Exception as e:
        print(f"[Error] Failed to load point_cloud_map from shared memory: {e}")
        return None

        
if __name__ == "__main__":
    shm_name = "scene_graph_shm"
    obj_shm_name = "object_shm"

    graph_res = load_graph_from_shared_memory(shm_name)
    # if graph_res is not None:
    #     graph, graph_shm = graph_res
    #     try:
    #         graph_shm.close()
    #         graph_shm.unlink()
    #         print("Shared memory cleaned up.")
    #     except:
    #         pass

    obj_res = load_object_data_from_shared_memory(obj_shm_name)
    # if graph_res is not None:
    #     objects, obj_shm = obj_res
    #     try:
    #         obj_shm.close()
    #         obj_shm.unlink()
    #         print("Shared memory cleaned up.")
    #     except:
    #         pass

    # print(f"graph_res:")
    # print(graph_res)
    # print("")
    # print(f"obj_res:")
    # print(obj_res)

    print(f"=====")
    print(f"graph_res.keys: {graph_res[0].nodes}")
    print(f"obj_res.keys: {obj_res[0].keys()}")
    print(f"=====")
    valid_places = []
    place2object_ids = {}
    for u, v, attr in graph_res[0].edges(data=True):
        if 'place' in u and 'object' in v:
            if not u in place2object_ids.keys():
                place2object_ids[u] = []
            place2object_ids[u].append(v)

    object_id2place = {}
    for k, v in place2object_ids.items():
        for vv in v:
            object_id2place[vv] = k

    print(f"place_object_ids_map:")
    for k, v in place2object_ids.items():
        print(f"{k}: {v}")

    print("")
    print(f"object_id_place_map:")
    for k, v in object_id2place.items():
        print(f"{k}: {v}")

    print(f"++++++++++++++++")
    object_name2places = {}
    for object_id, attr in obj_res[0].items():
        key = f"object_{object_id}"
        if key in object_id2place.keys():
            object_name = attr['class_name']
            if not object_name in object_name2places.keys():
                object_name2places[object_name] = []
            object_name2places[object_name].append(object_id2place[key])

    for k, v in object_name2places.items():
        print(f"{k}: {v}")
    print(f"++++++++++++++++")

    place_list = v
    for place in place_list:
        print(f"place: {place}")
        node = graph_res[0].nodes[place]
        print(node)
    print(f"++++++++++++++++")

    id = lambda s: int(s.split("_")[-1])
    place_data, object_data = {}, {}
    for place, object, attr in graph_res[0].edges(data=True):
        if 'place' in place and 'object' in object:
            place_data[id(place)] = graph_res[0].nodes[place]
            object_data[id(object)] = obj_res[0][id(object)]
    print(f"[place]")
    for k, v in place_data.items():
        print(f"{k}: ")
        for kk, vv in v.items():
            print(f"  > {kk}: {vv}")
    print(f"[object]")
    for k, v in object_data.items():
        print(f"{k}: ")
        for kk, vv in v.items():
            if kk == 'point_hash_key':
                continue
            print(f"  > {kk}: {vv}")
    print(f"++++++++++++++++")