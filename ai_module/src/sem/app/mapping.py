import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from collections import Counter

from typing import List, Optional

from .slam_classes import DetectionList, MapObjectList


def compute_spatial_similarities(
    # spatial_sim_type: str,
    detection_list: DetectionList,
    objects: MapObjectList,
    # downsample_voxel_size,
) -> torch.Tensor:
    det_bboxes = detection_list.get_stacked_values_torch("bbox")
    obj_bboxes = objects.get_stacked_values_torch("bbox")

    # spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    # Compute min and max for each box
    bbox1_min, _ = det_bboxes.min(dim=1)  # Shape: (M, 3)
    bbox1_max, _ = det_bboxes.max(dim=1)  # Shape: (M, 3)
    bbox2_min, _ = obj_bboxes.min(dim=1)  # Shape: (N, 3)
    bbox2_max, _ = obj_bboxes.max(dim=1)  # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(
        torch.clamp(inter_max - inter_min, min=0), dim=2
    )  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    spatial_sim = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return spatial_sim


def compute_visual_similarities(
    detection_list: DetectionList, objects: MapObjectList
) -> torch.Tensor:
    """
    Compute the visual similarities between the detections and the objects

    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    """
    det_fts = detection_list.get_stacked_values_torch("clip_ft")  # (M, D)
    obj_fts = objects.get_stacked_values_torch("clip_ft")  # (N, D)

    det_fts = det_fts.unsqueeze(-1)  # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0)  # (1, D, N)

    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1)  # (M, N)

    return visual_sim


def aggregate_similarities(
    match_method: str,
    phys_bias: float,
    spatial_sim: torch.Tensor,
    visual_sim: torch.Tensor,
) -> torch.Tensor:
    """
    Aggregate spatial and visual similarities into a single similarity score

    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    """
    if match_method == "sim_sum":
        sims = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
    else:
        raise ValueError(f"Unknown matching method: {match_method}")

    return sims


def match_detections_to_objects(
    agg_sim: torch.Tensor, detection_threshold: float = float("-inf")
) -> List[Optional[int]]:
    """
    Matches detections to objects based on similarity, returning match indices or None for unmatched.

    Args:
        agg_sim: Similarity matrix (detections vs. objects).
        detection_threshold: Threshold for a valid match (default: -inf).

    Returns:
        List of matching object indices (or None if unmatched) for each detection.
    """
    match_indices = []
    for detected_obj_idx in range(agg_sim.shape[0]):
        max_sim_value = agg_sim[detected_obj_idx].max()
        if max_sim_value <= detection_threshold:
            match_indices.append(None)
        else:
            match_indices.append(agg_sim[detected_obj_idx].argmax().item())

    return match_indices


def merge_obj_matches(
    detection_list: DetectionList,
    objects: MapObjectList,
    match_indices: List[Optional[int]],
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
) -> MapObjectList:
    """
    Merges detected objects into existing objects based on a list of match indices.

    Args:
        detection_list (DetectionList): List of detected objects.
        objects (MapObjectList): List of existing objects.
        match_indices (List[Optional[int]]): Indices of existing objects each detected object matches with.
        downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device:
            Parameters for merging and similarity computation.

    Returns:
        MapObjectList: Updated list of existing objects with detected objects merged as appropriate.
    """
    global tracker
    temp_curr_object_count = tracker.curr_object_count
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        if existing_obj_match_idx is None:
            # track the new object detection
            tracker.object_dict.update(
                {
                    "id": detection_list[detected_obj_idx]["id"],
                    "first_discovered": tracker.curr_frame_idx,
                }
            )

            objects.append(detection_list[detected_obj_idx])
        else:

            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False,
            )
            objects[existing_obj_match_idx] = merged_obj
    return objects


def process_pcd(
    pcd,
    downsample_voxel_size,
    dbscan_remove_noise,
    dbscan_eps,
    dbscan_min_points,
    run_dbscan=True,
):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    if dbscan_remove_noise and run_dbscan:
        pass
        pcd = pcd_denoise_dbscan(pcd, eps=dbscan_eps, min_points=dbscan_min_points)

    return pcd


def pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10
) -> o3d.geometry.PointCloud:
    # Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def get_bounding_box(spatial_sim_type, pcd):
    # if ("accurate" in spatial_sim_type or "overlap" in spatial_sim_type) and len(
    #     pcd.points
    # ) >= 4:
    #     try:
    #         return pcd.get_oriented_bounding_box(robust=True)
    #     except RuntimeError as e:
    #         print(f"Met {e}, use axis aligned bounding box instead")
    #         return pcd.get_axis_aligned_bounding_box()
    # else:
    return pcd.get_axis_aligned_bounding_box()


def merge_obj2_into_obj1(
    obj1,
    obj2,
    spatial_sim_type,
    device,
    downsample_voxel_size=0.025,
    dbscan_remove_noise=True,
    dbscan_eps=0.05,
    dbscan_min_points=10,
    run_dbscan=True,
):
    """
    Merges obj2 into obj1 with structured attribute handling, including explicit checks for unhandled keys.

    Parameters:
    - obj1, obj2: Objects to merge.
    - downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type: Parameters for point cloud processing.
    - device: Computation device.
    - run_dbscan: Whether to run DBSCAN for noise removal.

    Returns:
    - obj1: Updated object after merging.
    """
    # Attributes to be explicitly handled
    extend_attributes = [
        "image_idx",
        "mask_idx",
        "color_path",
        "class_id",
        "mask",
        "xyxy",
        "conf",
        "contain_number",
        "captions",
    ]
    add_attributes = ["num_detections", "num_obj_in_class"]
    skip_attributes = [
        "id",
        "class_name",
        "is_background",
        "new_counter",
        "curr_obj_num",
        "inst_color",
    ]  # 'inst_color' just keeps obj1's
    custom_handled = ["pcd", "bbox", "clip_ft", "text_ft", "n_points"]

    # Check for unhandled keys and throw an error if there are
    all_handled_keys = set(
        extend_attributes + add_attributes + skip_attributes + custom_handled
    )
    unhandled_keys = set(obj2.keys()) - all_handled_keys
    if unhandled_keys:
        raise ValueError(
            f"Unhandled keys detected in obj2: {unhandled_keys}. Please update the merge function to handle these attributes."
        )

    # Custom handling for 'pcd', 'bbox', 'clip_ft', and 'text_ft'
    n_obj1_det = obj1["num_detections"]
    n_obj2_det = obj2["num_detections"]

    # Process extend and add attributes
    for attr in extend_attributes:
        if attr in obj1 and attr in obj2:
            obj1[attr].extend(obj2[attr])

    for attr in add_attributes:
        if attr in obj1 and attr in obj2:
            obj1[attr] += obj2[attr]

    # Handling 'caption'
    if "caption" in obj1 and "caption" in obj2:
        # n_obj1_det = obj1['num_detections']
        for key, value in obj2["caption"].items():
            obj1["caption"][key + n_obj1_det] = value

    # merge pcd and bbox
    obj1["pcd"] += obj2["pcd"]
    obj1["pcd"] = process_pcd(
        obj1["pcd"],
        downsample_voxel_size,
        dbscan_remove_noise,
        dbscan_eps,
        dbscan_min_points,
        run_dbscan,
    )
    # update n_points
    obj1["n_points"] = len(np.asarray(obj1["pcd"].points))

    # Update 'bbox'
    obj1["bbox"] = get_bounding_box(spatial_sim_type, obj1["pcd"])
    obj1["bbox"].color = [0, 1, 0]

    # Merge and normalize 'clip_ft'
    obj1["clip_ft"] = (obj1["clip_ft"] * n_obj1_det + obj2["clip_ft"] * n_obj2_det) / (
        n_obj1_det + n_obj2_det
    )
    obj1["clip_ft"] = F.normalize(obj1["clip_ft"], dim=0)

    # merge text_ft
    # obj2['text_ft'] = to_tensor(obj2['text_ft'], device)
    # obj1['text_ft'] = to_tensor(obj1['text_ft'], device)
    # obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
    #                    obj2['text_ft'] * n_obj2_det) / (
    #                    n_obj1_det + n_obj2_det)
    # obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)

    return obj1
