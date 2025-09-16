import numpy as np
import cv2
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import OccupancyGrid
import sem.msg
from cv_bridge import CvBridge
import time


class ROSVisualizer:
    def __init__(
        self, get_color_func=None
    ):  # , grid_size=0.1, map_width=200, map_height=200, map_origin=(-12.5, -5.0)
        self.image_pub = rospy.Publisher("/annotated_image", Image, queue_size=1)
        self.object_nodes_pub = rospy.Publisher(
            "/object_nodes", sem.msg.ObjectNodes, queue_size=10
        )
        self.door_nodes_pub = rospy.Publisher(
            "/door_nodes", sem.msg.ObjectNodes, queue_size=10
        )
        self.door_markers_pub = rospy.Publisher(
            "/door_markers", MarkerArray, queue_size=10
        )
        self.obj_marker_pub = rospy.Publisher(
            "/all_objects_markers", MarkerArray, queue_size=10
        )
        self.cloud_color_pub = rospy.Publisher(
            "/segmented_cloud_color", pc2.PointCloud2, queue_size=1
        )
        self.object_cloud_pub = rospy.Publisher(
            "/object_cloud", pc2.PointCloud2, queue_size=1
        )
        self.floor_cloud_pub = rospy.Publisher(
            "/floor_cloud", pc2.PointCloud2, queue_size=1
        )
        self.wall_cloud_pub = rospy.Publisher(
            "/wall_cloud", pc2.PointCloud2, queue_size=1
        )
        self.debug_marker_pub = rospy.Publisher(
            "/debug_markers", MarkerArray, queue_size=10
        )
        self.map_cloud_pub = rospy.Publisher(
            "/map_cloud", pc2.PointCloud2, queue_size=1
        )
        self.room_marker_pub = rospy.Publisher(
            "/room_nodes", MarkerArray, queue_size=10
        )

        self.pcl_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.pcl_header = std_msgs.msg.Header()
        self.pcl_header.frame_id = "map"

        self.bridge = CvBridge()
        self.get_color = (
            get_color_func if get_color_func is not None else self.default_color
        )

        # occupancy grid map
        # self.occupancy_grid_pub = rospy.Publisher("/bg_occupancy_map", OccupancyGrid, queue_size=1)
        # self.occupancy_grid_msg = self.init_occupancy_grid(grid_size, map_width, map_height, map_origin)

    def default_color(self, class_id):
        return (255, 255, 255)

    # def init_occupancy_grid(self, grid_size, map_width, map_height, map_origin):
    #     grid = OccupancyGrid()
    #     grid.header.frame_id = "map"
    #     grid.info.resolution = grid_size
    #     grid.info.width = map_width
    #     grid.info.height = map_height

    #     origin = Pose()
    #     origin.position.x = map_origin[0]
    #     origin.position.y = map_origin[1]
    #     origin.position.z = 0.0
    #     grid.info.origin = origin

    #     return grid

    # def publish_occupancy_grid(self, occupancy_data):
    #     self.occupancy_grid_msg.header.stamp = rospy.Time.now()
    #     self.occupancy_grid_msg.data = occupancy_data
    #     self.occupancy_grid_pub.publish(self.occupancy_grid_msg)

    def draw_annotated_image(self, image, masks, boxes, confs, class_ids, class_names):
        image = image.copy()
        overlay = np.zeros_like(image, dtype=np.uint8)

        for i, cls_id in enumerate(class_ids):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]
            label = (
                f"{class_names[cls_id]} {conf:.2f}"
                if cls_id < len(class_names)
                else f"id {cls_id}"
            )
            color = self.get_color(cls_id)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if masks is not None and i < len(masks):
                mask = masks[i].astype(np.uint8)
                colored_mask = np.zeros_like(image, dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = mask * color[c]
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.8, 0)

        return cv2.addWeighted(image, 1.0, overlay, 0.8, 0)

    def publish_object_nodes(self, objects):
        object_nodes = sem.msg.ObjectNodes()
        object_nodes.header.stamp = rospy.Time.now()
        object_nodes.header.frame_id = "map"
        # object_nodes.num_objects = len(objects)
        object_nodes.objects = []
        # for obj in objects:
        for i, (object_id, obj) in enumerate(objects.items()):
            # if obj["num_detections"] < 5:
                # continue
            object_info = sem.msg.ObjectNode()
            object_info.id = obj["id"]
            # object_info.points = (
            # obj["points"].flatten().tolist()
            # )  # points는 1D 리스트로 변환
            object_info.points = []
            object_info.center = obj["center"].tolist()
            object_info.min_pt = obj["min_bbox"].tolist()
            object_info.max_pt = obj["max_bbox"].tolist()
            object_info.class_id = obj["class_id"]
            object_info.class_name = obj["class_name"]
            object_info.conf = obj["conf"]
            object_nodes.objects.append(object_info)
        object_nodes.num_objects = len(object_nodes.objects)
        self.object_nodes_pub.publish(object_nodes)

    def publish_door_nodes(self, objects):
        door_nodes = sem.msg.ObjectNodes()
        door_nodes.header.stamp = rospy.Time.now()
        door_nodes.header.frame_id = "map"
        door_nodes.objects = []
        # for obj in objects:
        for i, (object_id, obj) in enumerate(objects.items()):
            if obj["class_name"] == "door":
                door_info = sem.msg.ObjectNode()
                door_info.id = obj["id"]
                door_info.points = obj["points"].flatten().tolist()
                door_info.points = []
                door_info.center = obj["center"].tolist()
                door_info.min_pt = obj["min_bbox"].tolist()
                door_info.max_pt = obj["max_bbox"].tolist()
                door_info.class_id = obj["class_id"]
                door_info.class_name = obj["class_name"]
                door_info.conf = obj["conf"]
                door_nodes.objects.append(door_info)
        door_nodes.num_objects = len(door_nodes.objects)
        self.door_nodes_pub.publish(door_nodes)

    def create_door_markers(self, objects):
        marker_array = MarkerArray()
        # for i, obj in enumerate(objects):
        for i, (object_id, obj) in enumerate(objects.items()):
            if obj["class_name"] == "door":
                min_pt, max_pt = obj["min_bbox"], obj["max_bbox"]
                center = obj["center"]
                class_id = obj["class_id"]
                label = obj["class_name"]
                object_id = obj["id"]
                color = self.get_color(class_id)
                # marker = self.get_bbox_marker(min_pt, max_pt, object_id, color)
                scale = np.array(max_pt) - np.array(min_pt)
                center = (np.array(min_pt) + np.array(max_pt)) / 2.0
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "bbox"
                marker.id = object_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                (
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z,
                ) = center.tolist()
                (
                    marker.pose.orientation.x,
                    marker.pose.orientation.y,
                    marker.pose.orientation.z,
                    marker.pose.orientation.w,
                ) = [0.0, 0.0, 0.0, 1.0]
                marker.scale.x, marker.scale.y, marker.scale.z = scale.tolist()
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)
        return marker_array

    def create_markers(self, objects):
        marker_array = MarkerArray()
        # for i, obj in enumerate(objects):
        # if obj["num_detections"] < 5:
        # continue
        for i, (object_id, obj) in enumerate(objects.items()):
            # if obj.get("num_detections", 0) < 5:
            # continue
            min_pt, max_pt = obj["min_bbox"], obj["max_bbox"]
            center = obj["center"]
            class_id = obj["class_id"]
            label = obj["class_name"]
            object_id = obj["id"]
            color = self.get_color(class_id)
            num_detections = obj["num_detections"]
            conf = obj["conf"]

            marker_array.markers.append(self.get_bbox_marker(min_pt, max_pt, i, color))
            bubble, bubble_pos = self.get_bubble_marker(min_pt, max_pt, i, color)
            marker_array.markers.append(bubble)
            marker_array.markers.append(
                self.get_connection_line_marker(min_pt, max_pt, i, color)
            )
            marker_array.markers.append(
                self.get_label_marker(label, bubble_pos, object_id, conf, i, color)
            )

            room_center = obj.get("room_center", None)
            if room_center is not None:
                room_pos = [room_center[0], room_center[1], 10]
                marker_array.markers.append(
                    self.get_room_line_marker(bubble_pos, room_pos, i, color))

        return marker_array

    def get_bbox_marker(self, min_pt, max_pt, obj_id, color, frame_id="map"):
        scale = np.array(max_pt) - np.array(min_pt)
        center = (np.array(min_pt) + np.array(max_pt)) / 2.0
        center[2] += 4
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bbox"
        marker.id = obj_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            center.tolist()
        )
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = [0.0, 0.0, 0.0, 1.0]
        marker.scale.x, marker.scale.y, marker.scale.z = scale.tolist()
        marker.color.r, marker.color.g, marker.color.b = [c / 255.0 for c in color]
        marker.color.a = 0.3
        return marker

    def get_label_marker(
        self,
        text,
        pos,
        object_id,
        num_detections,
        obj_id,
        color,
        frame_id="map",
        id_offset=1000,
        scale=0.2,
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "labels"
        marker.id = obj_id + id_offset
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            pos[0],
            pos[1],
            pos[2] + 0.3,
        )
        marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b = [c / 255.0 for c in color]
        marker.color.a = 1.0
        marker.text = str(object_id) + " " + text # + f" ({num_detections:.2f})"
        return marker

    def get_bubble_marker(self, min_pt, max_pt, obj_id, color, frame_id="map"):
        top_center = (np.array(min_pt) + np.array(max_pt)) / 2.0
        top_center[2] = max_pt[2] +4
        sphere_pos = top_center + np.array([0.0, 0.0, 3.0])
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bubbles"
        marker.id = obj_id + 2000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            sphere_pos.tolist()
        )
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.r, marker.color.g, marker.color.b = [c / 255.0 for c in color]
        marker.color.a = 1.0
        return marker, sphere_pos

    def get_connection_line_marker(self, min_pt, max_pt, obj_id, color, frame_id="map"):
        top_center = (np.array(min_pt) + np.array(max_pt)) / 2.0
        top_center[2] = max_pt[2] + 4
        sphere_pos = top_center + np.array([0.0, 0.0, 3.0])
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lines"
        marker.id = obj_id + 4000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r, marker.color.g, marker.color.b = [c / 255.0 for c in color]
        marker.color.a = 0.8
        marker.points = [Point(*top_center), Point(*sphere_pos)]
        return marker

    def publish_room_nodes(self, room_centers):
        marker_array = MarkerArray()
        for room_id, center in room_centers.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "room_nodes"
            marker.id = room_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
                center[0],
                center[1],
                10.0,
            )
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r, marker.color.g, marker.color.b = [1.0, 0.0, 0.0]
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        self.room_marker_pub.publish(marker_array)

    def get_room_line_marker(self, bubble_pos, room_pos, obj_id, color, frame_id="map"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "room_lines"
        marker.id = obj_id + 6000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r, marker.color.g, marker.color.b = [1.0, 0.0, 0.0]
        marker.color.a = 0.8
        marker.points = [Point(*bubble_pos), Point(*room_pos)]
        return marker

    def publish_colored_point_cloud(self, obj_cloud_world_list, colors_list):
        all_points = []
        all_colors = []

        # Iterate over each object cloud and color
        for obj_cloud_world, color in zip(obj_cloud_world_list, colors_list):
            # Ensure the color array matches the number of points in obj_cloud_world
            all_points.append(obj_cloud_world)
            all_colors.append(color)

        # Stack all points and colors
        all_points = np.vstack(all_points)  # Shape: (total_points, 3)
        all_colors = np.vstack(all_colors)  # Shape: (total_points, 3)

        rgb_uint32 = (
            (all_colors[:, 2].astype(np.uint32) << 16)
            | (all_colors[:, 1].astype(np.uint32) << 8)
            | (all_colors[:, 0].astype(np.uint32))
        )
        rgb_float32 = rgb_uint32.view(np.float32).reshape(-1, 1)

        # (x, y, z, rgb) 형태로 합치기
        colored_cloud = np.hstack([all_points, rgb_float32])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcl_msg = pc2.create_cloud(header, self.pcl_fields, colored_cloud)

        self.cloud_color_pub.publish(pcl_msg)
    # def publish_colored_point_cloud(self, cloud_world, colors):
    #     rgb_uint32 = (
    #         (colors[:, 2].astype(np.uint32) << 16)
    #         | (colors[:, 1].astype(np.uint32) << 8)
    #         | (colors[:, 0].astype(np.uint32))
    #     )
    #     rgb_float32 = rgb_uint32.view(np.float32).reshape(-1, 1)

    #     colored_cloud = np.hstack([cloud_world, rgb_float32])

    #     header = std_msgs.msg.Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = "map"
    #     pcl_msg = pc2.create_cloud(header, self.pcl_fields, colored_cloud)

    #     self.cloud_color_pub.publish(pcl_msg)

    def publish_object_cloud(self, obj_clouds, labels):
        merged = []
        for cloud, label in zip(obj_clouds, labels):
            color = self.get_color(label)
            color_np = np.tile(np.array(color, dtype=np.uint8), (cloud.shape[0], 1))
            rgb = (
                (
                    np.left_shift(color_np[:, 0].astype(np.uint32), 16)
                    + np.left_shift(color_np[:, 1].astype(np.uint32), 8)
                    + color_np[:, 2].astype(np.uint32)
                )
                .astype(np.float32)
                .view(np.float32)
            )
            colored = np.concatenate([cloud[:, :3], rgb[:, None]], axis=1)
            merged.append(colored)

        if merged:
            cloud = np.vstack(merged).astype(np.float32)
            self.pcl_header.stamp = rospy.Time.now()
            t200 = time.time()
            msg = pc2.create_cloud(
                self.pcl_header, self.pcl_fields, cloud
            )  # take long time
            t201 = time.time()
            self.object_cloud_pub.publish(msg)

            # print(f"pc2.create_cloud: {t201 - t200:.4f}")

    def publish_image(self, image, pub):
        pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))

    def generate_bev_image(
        self,
        objects,
        robot_pos,
        cloud,
        image_size=(750, 500),
        scale=50,
        offset=(-4.0, -2.0),
    ):
        img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        origin = (image_size[0] // 2, image_size[1] // 2)

        def world_to_pixel(x, y):
            return int(origin[0] + (x + offset[0]) * scale), int(
                origin[1] - (y + offset[1]) * scale
            )

        for pt in cloud:
            px, py = world_to_pixel(pt[0], pt[1])
            if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                img[py, px] = (255, 255, 255)

        rx, ry = world_to_pixel(robot_pos[0], robot_pos[1])
        cv2.circle(img, (rx, ry), 5, (255, 0, 0), -1)

        for det in objects:
            cx, cy = det["center"][:2]
            px, py = world_to_pixel(cx, cy)
            min_pt, max_pt = det["min_bbox"], det["max_bbox"]
            w = abs(max_pt[0] - min_pt[0])
            h = abs(max_pt[1] - min_pt[1])
            box_w = int(w * scale / 2)
            box_h = int(h * scale / 2)
            color = self.get_color(det["class_id"])
            cv2.rectangle(
                img, (px - box_w, py - box_h), (px + box_w, py + box_h), color, 2
            )
            cv2.putText(
                img,
                det["class_name"],
                (px + box_w + 3, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return img

    def publish_bg_cloud(self, floor_cloud, wall_cloud):
        if floor_cloud.shape[0] == 0 and wall_cloud.shape[0] == 0:
            return

        if floor_cloud.shape[0] > 0:
            N1 = floor_cloud.shape[0]
            rgb1 = np.full((N1, 1), (160 << 16 | 160 << 8 | 160), dtype=np.uint32).view(
                np.float32
            )
            cloud1 = np.concatenate([floor_cloud, rgb1], axis=1)

        if wall_cloud.shape[0] > 0:
            N2 = wall_cloud.shape[0]
            rgb2 = np.full((N2, 1), (80 << 16 | 80 << 8 | 80), dtype=np.uint32).view(
                np.float32
            )
            cloud2 = np.concatenate([wall_cloud, rgb2], axis=1)

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        msg = pc2.create_cloud(header, self.pcl_fields, cloud1)

        self.floor_cloud_pub.publish(msg)

        msg = pc2.create_cloud(header, self.pcl_fields, cloud2)
        self.wall_cloud_pub.publish(msg)

    def create_merge_debug_markers(self, det1, det2, iou, center_dist, reason):
        center1 = (np.array(det1["min_bbox"]) + np.array(det1["max_bbox"])) / 2.0
        center2 = (np.array(det2["min_bbox"]) + np.array(det2["max_bbox"])) / 2.0
        if (
            reason == "diff_O"
            or (reason == "X" and iou > 0.001)
            or (reason == "X" and center_dist < 0.3)
        ):
            center1[2] = np.array(det1["max_bbox"])[2] + 5.0
            center2[2] = np.array(det2["max_bbox"])[2] + 5.0
        mid_point = (center1 + center2) / 2.0
        text = f"IoU={iou:.2f}, d={center_dist:.2f}, by={reason}"

        color_map = {
            "iou_O": (0.0, 0.0, 1.0),
            "dist_O": (1.0, 0.0, 1.0),
            "matched_include": (0.0, 1.0, 0.0),
            "diff_O": (1.0, 0.5, 0.0),
            "X": (0.6, 0.6, 0.6),
        }
        color = color_map.get(reason, (0.5, 0.5, 0.5))

        marker_text = Marker()
        marker_text.header.frame_id = "map"
        marker_text.header.stamp = rospy.Time.now()
        marker_text.ns = "merge_debug_text"
        marker_text.id = hash((det1["id"], det2["id"])) % 100000
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD

        (
            marker_text.pose.position.x,
            marker_text.pose.position.y,
            marker_text.pose.position.z,
        ) = (mid_point[0], mid_point[1], mid_point[2] + 0.5)
        marker_text.scale.z = 0.2
        marker_text.color.r, marker_text.color.g, marker_text.color.b = color
        marker_text.color.a = 1.0
        marker_text.text = text

        marker_line = Marker()
        marker_line.header.frame_id = "map"
        marker_line.header.stamp = rospy.Time.now()
        marker_line.ns = "merge_debug_line"
        marker_line.id = hash(("line", det1["id"], det2["id"])) % 100000
        marker_line.type = Marker.LINE_LIST
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.01
        marker_line.color.r, marker_line.color.g, marker_line.color.b = color
        marker_line.color.a = 0.8
        marker_line.points = [Point(*center1), Point(*center2)]

        return [marker_text, marker_line]

    def publish_debug_markers(self, debug_markers: MarkerArray):
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "merge_debug"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL

        delete_array = MarkerArray()
        delete_array.markers.append(delete_marker)
        self.debug_marker_pub.publish(delete_array)

        self.debug_marker_pub.publish(debug_markers)
