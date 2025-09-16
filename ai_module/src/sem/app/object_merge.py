import numpy as np
from typing import List, Dict
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class ObjectMerger:
    def __init__(self, voxel_downsample_func, color_getter):
        """
        voxel_downsample_func: np.ndarray → np.ndarray
        color_getter: function(int) → (r, g, b)
        """
        self.voxel_downsample = voxel_downsample_func
        self.get_color = color_getter
        self.debug_marker_pub = rospy.Publisher("/merge_debug_markers", MarkerArray, queue_size=1)

    def merge(
        self,
        detections,
        objects,
        spatial_edges,
        merge_threshold1=0.1,
        merge_threshold2=0.3,
        merge_threshold3=0.1,
    ):
        """
        detections: 현재 프레임에서의 객체 리스트
        objects: 기존 저장된 객체 리스트
        spatial_edges: (num_detections, num_objects) similarity matrix
        """
        debug_markers = MarkerArray()

        # 이전 마커 제거
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "merge_debug_text"
        delete_marker.action = Marker.DELETEALL
        debug_markers.markers.append(delete_marker)
        # next_id = max([obj["id"] for obj in objects], default=-1) + 1
        # merged_indices = set()    # 삭제할 object 인덱스
        # matched_indices = set()   # 이미 매칭된 object 인덱스
        # unmatched = 0

        # num_detections, num_objects = spatial_edges.shape

        # for i in range(num_detections):
        #     matched = False
        #     det = detections[i]
        #     candidate_objects = []

        #     for j in range(num_objects):
        #         if j >= len(objects):
        #             continue
        #         if j in matched_indices:
        #             continue

        #         obj = objects[j]
        #         sim = spatial_edges[i, j]
        #         dist = np.linalg.norm(det["center"] - obj["center"])

        #         def is_fully_contained(bbox1, bbox2):
        #             return (
        #                 np.all(bbox1[0] >= bbox2[0]) and np.all(bbox1[1] <= bbox2[1])
        #                 or
        #                 np.all(bbox2[0] >= bbox1[0]) and np.all(bbox2[1] <= bbox1[1])
        #             )

        #         if (det["class_id"] == obj["class_id"] and sim > merge_threshold1) or (
        #             det["class_id"] != obj["class_id"] and sim > merge_threshold2) or (
        #             det["class_id"] == obj["class_id"] and dist < merge_threshold3) or (
        #             is_fully_contained(det["bbox"], obj["bbox"])
        #         ):
        #             candidate_objects.append((j, obj, sim, dist))

        #     selected_objects = {}
        #     for j, obj, sim, dist in candidate_objects:
        #         image_idx = tuple(obj.get("image_idx", []))
        #         for idx in image_idx:
        #             if idx not in selected_objects:
        #                 selected_objects[idx] = (j, obj, sim, dist)
        #             else:
        #                 prev_j, prev_obj, prev_sim, prev_dist = selected_objects[idx]
        #                 prev_bbox = prev_obj["bbox"]
        #                 curr_bbox = obj["bbox"]
        #                 bbox_contained = is_fully_contained(prev_bbox, curr_bbox)

        #                 if obj["class_id"] == det["class_id"] and prev_obj["class_id"] != det["class_id"]:
        #                     selected_objects[idx] = (j, obj, sim, dist)
        #                 elif obj["class_id"] == det["class_id"] and prev_obj["class_id"] == det["class_id"]:
        #                     if sim > prev_sim:
        #                         selected_objects[idx] = (j, obj, sim, dist)
        #                     elif sim == prev_sim and dist < prev_dist:
        #                         selected_objects[idx] = (j, obj, sim, dist)
        #                     elif bbox_contained:
        #                         selected_objects[idx] = (j, obj, sim, dist)
        #                 elif obj["class_id"] != det["class_id"] and prev_obj["class_id"] != det["class_id"]:
        #                     if sim > prev_sim:
        #                         selected_objects[idx] = (j, obj, sim, dist)

        #     matched_objects = []
        #     used_image_idx = set()

        #     sorted_candidates = sorted(
        #         selected_objects.values(),
        #         key=lambda x: (-int(x[1]["class_id"] == det["class_id"]), -x[2])
        #     )

        #     for j, obj, sim, dist in sorted_candidates:
        #         obj_image_idx = obj.get("image_idx", [])
        #         if not isinstance(obj_image_idx, list):
        #             obj_image_idx = [obj_image_idx]

        #         if not any(idx in used_image_idx for idx in obj_image_idx):
        #             matched_objects.append(obj)
        #             used_image_idx.update(obj_image_idx)

        #     matched_j_indices = [objects.index(obj) for obj in matched_objects]

        #     if matched_objects:
        #         merged_points = det["points"]
        #         merged_image_idx = det["image_idx"].copy()
        #         for obj in matched_objects:
        #             merged_points = np.concatenate([merged_points, obj["points"]], axis=0)
        #         merged_points = np.unique(merged_points, axis=0)
        #         merged_image_idx = list(sorted(set(merged_image_idx)))

        #         center = np.mean(merged_points, axis=0)
        #         min_pt = np.min(merged_points, axis=0)
        #         max_pt = np.max(merged_points, axis=0)
        #         bbox = np.stack([min_pt, max_pt], axis=0)

        #         obj = matched_objects[0]
        #         if det["class_id"] != obj["class_id"] and det["conf"] > obj["conf"]:
        #             obj["class_id"] = det["class_id"]
        #         obj["conf"] = max(obj["conf"], det["conf"])

        #         obj["points"] = merged_points
        #         obj["n_points"] = merged_points.shape[0]
        #         obj["center"] = center
        #         obj["bbox"] = bbox
        #         obj["min_bbox"] = min_pt
        #         obj["max_bbox"] = max_pt
        #         obj["num_detections"] += len(matched_objects)
        #         matched = True

        #         for merged_obj in matched_objects[1:]:
        #             merged_indices.add(objects.index(merged_obj))

        #         matched_indices.update(matched_j_indices)

        #     if not matched:
        #         new_obj = det.copy()
        #         new_obj["id"] = next_id
        #         new_obj.pop("box", None)
        #         new_obj.pop("xyxy", None)
        #         objects.append(new_obj)
        #         next_id += 1
        #         unmatched += 1

        # objects = [obj for idx, obj in enumerate(objects) if idx not in merged_indices]

        next_id = max([obj["id"] for obj in objects], default=-1) + 1
        merged_indices = set()
        unmatched = 0

        num_detections, num_objects = spatial_edges.shape

        for i in range(num_detections):
            matched = False
            det = detections[i]
            matched_objects = []

            for j in range(num_objects):
                if j >= len(objects):
                    continue
                obj = objects[j]

                included = (
                    np.all(det["bbox"][0] >= obj["bbox"][0]) and np.all(det["bbox"][1] <= obj["bbox"][1])
                ) or (
                    np.all(obj["bbox"][0] >= det["bbox"][0]) and np.all(obj["bbox"][1] <= det["bbox"][1])
                )

                sim = spatial_edges[i, j]
                dist = np.linalg.norm(det["center"] - obj["center"])
                if dist < 0.8:
                    matched_this = (
                        (det["class_id"] == obj["class_id"] and sim > merge_threshold1) or
                        (det["class_id"] != obj["class_id"] and sim > merge_threshold2) or
                        (det["class_id"] == obj["class_id"] and dist < merge_threshold3) or
                        included
                    )

                    idx = i * 1000 + j  # 고유 ID
                    debug_markers.markers.extend(self.create_debug_markers(
                        det, obj, sim, dist, included, matched_this, idx
                    ))

                    if matched_this:
                        matched_objects.append(obj)

            if matched_objects:
                merged_points = det["points"]
                for obj in matched_objects:
                    merged_points = np.concatenate(
                        [merged_points, obj["points"]], axis=0
                    )
                merged_points = np.unique(merged_points, axis=0)
                obj["points"] = self.voxel_downsample(merged_points, voxel_size=0.1)

                center = np.mean(merged_points, axis=0)
                min_pt = np.min(merged_points, axis=0)
                max_pt = np.max(merged_points, axis=0)
                bbox = np.stack([min_pt, max_pt], axis=0)

                obj = matched_objects[0]
                if det["class_id"] != obj["class_id"] and det["conf"] > obj["conf"]:
                    obj["class_id"] = det["class_id"]
                obj["conf"] = max(obj["conf"], det["conf"])

                # total_detections = det.get("num_detections", 1) + sum(
                #     obj.get("num_detections", 1)
                #     for obj in matched_objects
                #     if obj is not merged_object
                # )

                obj["points"] = merged_points
                obj["n_points"] = merged_points.shape[0]
                obj["center"] = center
                obj["bbox"] = bbox
                obj["min_bbox"] = min_pt
                obj["max_bbox"] = max_pt
                obj["num_detections"] += len(matched_objects)
                matched = True

                for merged_obj in matched_objects[1:]:
                    merged_indices.add(objects.index(merged_obj))

            if not matched:
                new_obj = det.copy()
                new_obj["id"] = next_id
                new_obj.pop("box", None)
                new_obj.pop("xyxy", None)

                # print(f"🆕 New object ID: {next_id}")
                objects.append(new_obj)
                next_id += 1
                unmatched += 1

        objects = [obj for idx, obj in enumerate(objects) if idx not in merged_indices]

        print(
            f"\033[94m✅ detected: {len(detections)}, all: {len(objects)}, new objs: {unmatched}, matched: {len(detections) - unmatched}\033[0m"
        )
        self.debug_marker_pub.publish(debug_markers)


        return objects

    def create_debug_markers(self, det, obj, sim, dist, included, matched, idx):
        center1 = np.array(det["center"])
        center2 = np.array(obj["center"])
        mid = (center1 + center2) / 2.0

        reason = "O" if matched else "X"
        text = f"iou={sim:.2f}, d={dist:.2f}, inc={'O' if included else 'X'} {reason}"

        color = (0.0, 1.0, 0.0) if matched else (0.7, 0.7, 0.7)

        # 텍스트 마커
        marker_text = Marker()
        marker_text.header.frame_id = "map"
        marker_text.header.stamp = rospy.Time.now()
        marker_text.ns = "merge_debug_text"
        marker_text.id = idx * 2
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD
        marker_text.pose.position.x, marker_text.pose.position.y, marker_text.pose.position.z = (
            mid[0], mid[1], mid[2] + 0.3
        )
        marker_text.scale.z = 0.2
        marker_text.color.r, marker_text.color.g, marker_text.color.b = color
        marker_text.color.a = 1.0
        marker_text.text = text

        # 라인 마커
        marker_line = Marker()
        marker_line.header.frame_id = "map"
        marker_line.header.stamp = rospy.Time.now()
        marker_line.ns = "merge_debug_line"
        marker_line.id = idx * 2 + 1
        marker_line.type = Marker.LINE_LIST
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.01
        marker_line.color.r, marker_line.color.g, marker_line.color.b = color
        marker_line.color.a = 0.8
        marker_line.points = [Point(*center1), Point(*center2)]

        return [marker_text, marker_line]


class ObjectMerger2:
    def __init__(self, voxel_downsample_func, color_getter):
        """
        voxel_downsample_func: np.ndarray → np.ndarray
        color_getter: function(int) → (r, g, b)
        """
        self.voxel_downsample = voxel_downsample_func
        self.get_color = color_getter
        self.debug_marker_pub = rospy.Publisher("/merge_debug_markers", MarkerArray, queue_size=1)

    def merge(
        self,
        detections,
        objects,
        spatial_edges,
        merge_threshold1=0.1,
        merge_threshold2=0.3,
        merge_threshold3=0.1,
    ):
        """
        detections: 현재 프레임에서의 객체 리스트
        objects: 기존 저장된 객체 리스트
        spatial_edges: (num_detections, num_objects) similarity matrix
        """
        debug_markers = MarkerArray()

        # 이전 마커 제거
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "merge_debug_text"
        delete_marker.action = Marker.DELETEALL
        debug_markers.markers.append(delete_marker)
        # next_id = max([obj["id"] for obj in objects], default=-1) + 1
        # merged_indices = set()    # 삭제할 object 인덱스
        # matched_indices = set()   # 이미 매칭된 object 인덱스
        # unmatched = 0

        # num_detections, num_objects = spatial_edges.shape

        # for i in range(num_detections):
        #     matched = False
        #     det = detections[i]
        #     candidate_objects = []

        #     for j in range(num_objects):
        #         if j >= len(objects):
        #             continue
        #         if j in matched_indices:
        #             continue

        #         obj = objects[j]
        #         sim = spatial_edges[i, j]
        #         dist = np.linalg.norm(det["center"] - obj["center"])

        #         def is_fully_contained(bbox1, bbox2):
        #             return (
        #                 np.all(bbox1[0] >= bbox2[0]) and np.all(bbox1[1] <= bbox2[1])
        #                 or
        #                 np.all(bbox2[0] >= bbox1[0]) and np.all(bbox2[1] <= bbox1[1])
        #             )

        #         if (det["class_id"] == obj["class_id"] and sim > merge_threshold1) or (
        #             det["class_id"] != obj["class_id"] and sim > merge_threshold2) or (
        #             det["class_id"] == obj["class_id"] and dist < merge_threshold3) or (
        #             is_fully_contained(det["bbox"], obj["bbox"])
        #         ):
        #             candidate_objects.append((j, obj, sim, dist))

        #     selected_objects = {}
        #     for j, obj, sim, dist in candidate_objects:
        #         image_idx = tuple(obj.get("image_idx", []))
        #         for idx in image_idx:
        #             if idx not in selected_objects:
        #                 selected_objects[idx] = (j, obj, sim, dist)
        #             else:
        #                 prev_j, prev_obj, prev_sim, prev_dist = selected_objects[idx]
        #                 prev_bbox = prev_obj["bbox"]
        #                 curr_bbox = obj["bbox"]
        #                 bbox_contained = is_fully_contained(prev_bbox, curr_bbox)

        #                 if obj["class_id"] == det["class_id"] and prev_obj["class_id"] != det["class_id"]:
        #                     selected_objects[idx] = (j, obj, sim, dist)
        #                 elif obj["class_id"] == det["class_id"] and prev_obj["class_id"] == det["class_id"]:
        #                     if sim > prev_sim:
        #                         selected_objects[idx] = (j, obj, sim, dist)
        #                     elif sim == prev_sim and dist < prev_dist:
        #                         selected_objects[idx] = (j, obj, sim, dist)
        #                     elif bbox_contained:
        #                         selected_objects[idx] = (j, obj, sim, dist)
        #                 elif obj["class_id"] != det["class_id"] and prev_obj["class_id"] != det["class_id"]:
        #                     if sim > prev_sim:
        #                         selected_objects[idx] = (j, obj, sim, dist)

        #     matched_objects = []
        #     used_image_idx = set()

        #     sorted_candidates = sorted(
        #         selected_objects.values(),
        #         key=lambda x: (-int(x[1]["class_id"] == det["class_id"]), -x[2])
        #     )

        #     for j, obj, sim, dist in sorted_candidates:
        #         obj_image_idx = obj.get("image_idx", [])
        #         if not isinstance(obj_image_idx, list):
        #             obj_image_idx = [obj_image_idx]

        #         if not any(idx in used_image_idx for idx in obj_image_idx):
        #             matched_objects.append(obj)
        #             used_image_idx.update(obj_image_idx)

        #     matched_j_indices = [objects.index(obj) for obj in matched_objects]

        #     if matched_objects:
        #         merged_points = det["points"]
        #         merged_image_idx = det["image_idx"].copy()
        #         for obj in matched_objects:
        #             merged_points = np.concatenate([merged_points, obj["points"]], axis=0)
        #         merged_points = np.unique(merged_points, axis=0)
        #         merged_image_idx = list(sorted(set(merged_image_idx)))

        #         center = np.mean(merged_points, axis=0)
        #         min_pt = np.min(merged_points, axis=0)
        #         max_pt = np.max(merged_points, axis=0)
        #         bbox = np.stack([min_pt, max_pt], axis=0)

        #         obj = matched_objects[0]
        #         if det["class_id"] != obj["class_id"] and det["conf"] > obj["conf"]:
        #             obj["class_id"] = det["class_id"]
        #         obj["conf"] = max(obj["conf"], det["conf"])

        #         obj["points"] = merged_points
        #         obj["n_points"] = merged_points.shape[0]
        #         obj["center"] = center
        #         obj["bbox"] = bbox
        #         obj["min_bbox"] = min_pt
        #         obj["max_bbox"] = max_pt
        #         obj["num_detections"] += len(matched_objects)
        #         matched = True

        #         for merged_obj in matched_objects[1:]:
        #             merged_indices.add(objects.index(merged_obj))

        #         matched_indices.update(matched_j_indices)

        #     if not matched:
        #         new_obj = det.copy()
        #         new_obj["id"] = next_id
        #         new_obj.pop("box", None)
        #         new_obj.pop("xyxy", None)
        #         objects.append(new_obj)
        #         next_id += 1
        #         unmatched += 1

        # objects = [obj for idx, obj in enumerate(objects) if idx not in merged_indices]

        next_id = max([obj["id"] for obj in objects], default=-1) + 1
        merged_indices = set()
        unmatched = 0

        num_detections, num_objects = spatial_edges.shape

        map_obj_ids = []
        for i in range(num_detections):
            matched = False
            det = detections[i]
            matched_objects = []

            for j in range(num_objects):
                if j >= len(objects):
                    continue
                obj = objects[j]

                included = (
                    np.all(det["bbox"][0] >= obj["bbox"][0]) and np.all(det["bbox"][1] <= obj["bbox"][1])
                ) or (
                    np.all(obj["bbox"][0] >= det["bbox"][0]) and np.all(obj["bbox"][1] <= det["bbox"][1])
                )

                sim = spatial_edges[i, j]
                dist = np.linalg.norm(det["center"] - obj["center"])
                if dist < 0.8:
                    matched_this = (
                        (det["class_id"] == obj["class_id"] and sim > merge_threshold1) or
                        (det["class_id"] != obj["class_id"] and sim > merge_threshold2) or
                        (det["class_id"] == obj["class_id"] and dist < merge_threshold3) or
                        included
                    )

                    idx = i * 1000 + j  # 고유 ID
                    debug_markers.markers.extend(self.create_debug_markers(
                        det, obj, sim, dist, included, matched_this, idx
                    ))

                    if matched_this:
                        matched_objects.append(obj)

            if matched_objects:
                merged_points = det["points"]
                for obj in matched_objects:
                    merged_points = np.concatenate(
                        [merged_points, obj["points"]], axis=0
                    )
                merged_points = np.unique(merged_points, axis=0)
                obj["points"] = self.voxel_downsample(merged_points, voxel_size=0.1)

                center = np.mean(merged_points, axis=0)
                min_pt = np.min(merged_points, axis=0)
                max_pt = np.max(merged_points, axis=0)
                bbox = np.stack([min_pt, max_pt], axis=0)

                obj = matched_objects[0]
                if det["class_id"] != obj["class_id"] and det["conf"] > obj["conf"]:
                    obj["class_id"] = det["class_id"]
                obj["conf"] = max(obj["conf"], det["conf"])

                # total_detections = det.get("num_detections", 1) + sum(
                #     obj.get("num_detections", 1)
                #     for obj in matched_objects
                #     if obj is not merged_object
                # )

                obj["points"] = merged_points
                obj["n_points"] = merged_points.shape[0]
                obj["center"] = center
                obj["bbox"] = bbox
                obj["min_bbox"] = min_pt
                obj["max_bbox"] = max_pt
                obj["num_detections"] += len(matched_objects)
                matched = True

                for merged_obj in matched_objects[1:]:
                    merged_indices.add(objects.index(merged_obj))
                    
                det["id"] = obj["id"]
                map_obj_ids.append(obj['id'])

            if not matched:
                new_obj = det.copy()
                new_obj["id"] = next_id
                new_obj.pop("box", None)
                new_obj.pop("xyxy", None)

                # print(f"🆕 New object ID: {next_id}")
                objects.append(new_obj)
                next_id += 1
                unmatched += 1
                
                det["id"] = new_obj["id"]
                map_obj_ids.append(new_obj["id"])

        objects = [obj for idx, obj in enumerate(objects) if idx not in merged_indices]

        print(
            f"\033[94m✅ detected: {len(detections)}, all: {len(objects)}, new objs: {unmatched}, matched: {len(detections) - unmatched}\033[0m"
        )
        self.debug_marker_pub.publish(debug_markers)


        return objects, map_obj_ids

    def create_debug_markers(self, det, obj, sim, dist, included, matched, idx):
        center1 = np.array(det["center"])
        center2 = np.array(obj["center"])
        mid = (center1 + center2) / 2.0

        reason = "O" if matched else "X"
        text = f"iou={sim:.2f}, d={dist:.2f}, inc={'O' if included else 'X'} {reason}"

        color = (0.0, 1.0, 0.0) if matched else (0.7, 0.7, 0.7)

        # 텍스트 마커
        marker_text = Marker()
        marker_text.header.frame_id = "map"
        marker_text.header.stamp = rospy.Time.now()
        marker_text.ns = "merge_debug_text"
        marker_text.id = idx * 2
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD
        marker_text.pose.position.x, marker_text.pose.position.y, marker_text.pose.position.z = (
            mid[0], mid[1], mid[2] + 0.3
        )
        marker_text.scale.z = 0.2
        marker_text.color.r, marker_text.color.g, marker_text.color.b = color
        marker_text.color.a = 1.0
        marker_text.text = text

        # 라인 마커
        marker_line = Marker()
        marker_line.header.frame_id = "map"
        marker_line.header.stamp = rospy.Time.now()
        marker_line.ns = "merge_debug_line"
        marker_line.id = idx * 2 + 1
        marker_line.type = Marker.LINE_LIST
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.01
        marker_line.color.r, marker_line.color.g, marker_line.color.b = color
        marker_line.color.a = 0.8
        marker_line.points = [Point(*center1), Point(*center2)]

        return [marker_text, marker_line]


