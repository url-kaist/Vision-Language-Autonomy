import json
from multiprocessing import shared_memory
from networkx.readwrite import json_graph
import networkx as nx

def load_graph_from_shared_memory(shm_name = "scene_graph_shm"):
    try:
        shm = shared_memory.SharedMemory(shm_name)
        raw_data = bytes(shm.buf[:])
        graph_data = json.loads(raw_data.decode("utf-8"))
        G = json_graph.node_link_graph(graph_data)
        # print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

        return G, shm
    
    except FileNotFoundError:
        # print(f"[Error] Shared memory '{shm_name}' not found.")
        return None
    except Exception as e:
        # print(f"[Error] Failed to load from shared memory: {e}")
        return None
    
    # finally:
    #     try:
    #         shm.close()
    #         shm.unlink()
    #         print("Shared memory cleaned up.")
    #     except:
    #         pass

        
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