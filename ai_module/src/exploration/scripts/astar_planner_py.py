#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/ws/external")
import rospy
import numpy as np
import math
import heapq
import cv2
from collections import deque

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from exploration.srv import AstarPath, AstarPathResponse

from ai_module.src.utils.logger import Logger


# AStarPlanner 클래스는 이전과 동일하게 유지됩니다.
class AStarPlanner:
    def __init__(self, grid_map, resolution):
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='AStarPlanner',
            log_path="/ws/external/log/exploration/astar_planner.log",
            no_intro=True
        )
        self.grid_map = grid_map
        self.resolution = resolution
        self.height, self.width = grid_map.shape
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def plan(self, start_pixel, end_pixel):
        if self.grid_map[start_pixel[1], start_pixel[0]] == 100 or \
           self.grid_map[end_pixel[1], end_pixel[0]] == 100:
            self.logger.logwarn("Start or end point is inside an obstacle.")
            return None, float('inf')

        open_set = []
        heapq.heappush(open_set, (0, start_pixel))
        came_from = {}
        g_score = { (x,y): float('inf') for y in range(self.height) for x in range(self.width) }
        g_score[start_pixel] = 0
        f_score = { (x,y): float('inf') for y in range(self.height) for x in range(self.width) }
        f_score[start_pixel] = self._heuristic(start_pixel, end_pixel)

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_pixel:
                return self._reconstruct_path(came_from, current)
            for dx, dy in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                if self.grid_map[neighbor[1], neighbor[0]] == 100:
                    continue
                move_cost = math.hypot(dx, dy)
                tentative_g_score = g_score[current] + move_cost
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end_pixel)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        self.logger.logwarn("A* planner could not find a path.")
        return None, float('inf')

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        path_length = 0.0
        for i in range(len(path) - 1):
            dist = math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            path_length += dist * self.resolution
        return path, path_length


class AStarServiceNode:
    def __init__(self):
        rospy.init_node('python_astar_service_node')
        quiet = rospy.get_param('~quiet', False)
        self.logger = Logger(
            quiet=quiet, prefix='AStar', log_path="/ws/external/log/exploration/astar.log")

        self.inflate_radius_m = rospy.get_param('~inflate_radius', 0.0)
        self.local_grid_resolution = rospy.get_param('~local_grid_resolution', 0.05)

        # 동적으로 생성될 지역 지도의 정보를 저장할 변수들
        self.local_map_info = None
        self.inflated_map = None
        self.map_received = False

        # 구독 토픽을 '/traversable_area'로 변경하고 콜백 함수를 새로 지정합니다.
        self.map_sub = rospy.Subscriber('/traversable_area_filtered', PointCloud2, self._pointcloud_callback,
                                        queue_size=1)
        self.service = rospy.Service('/astar_path', AstarPath, self._service_callback)
        self.path_pub = rospy.Publisher('~visualization/astar_path', Path, queue_size=1)


        self.logger.loginfo("Python A* service node is ready. Waiting for '/traversable_area' topic...")
        rospy.spin()


    def _pointcloud_callback(self, msg: PointCloud2):
        """
        PointCloud2 메시지를 받아 동적으로 OccupancyGrid를 생성하는 새로운 콜백 함수
        """
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        if len(points_list) < 10:
            return

        points_2d = np.array([p[:2] for p in points_list])
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)

        padding = 1.0 # 지도 경계에 여유 공간 추가
        local_origin = (min_x - padding, min_y - padding)
        local_width = int((max_x - min_x + 2 * padding) / self.local_grid_resolution)
        local_height = int((max_y - min_y + 2 * padding) / self.local_grid_resolution)

        local_grid = np.full((local_height, local_width), 100, dtype=np.uint8)

        for point in points_2d:
            px = int((point[0] - local_origin[0]) / self.local_grid_resolution)
            py = int((point[1] - local_origin[1]) / self.local_grid_resolution)
            if 0 <= px < local_width and 0 <= py < local_height:
                local_grid[local_height - 1 - py, px] = 0

        inflate_radius_px = int(self.inflate_radius_m / self.local_grid_resolution)
        if inflate_radius_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflate_radius_px + 1, 2 * inflate_radius_px + 1))
            self.inflated_map = cv2.dilate(local_grid, kernel)
        else:
            self.inflated_map = local_grid

        # 서비스 콜백에서 사용할 수 있도록 지도 정보를 self에 저장
        self.local_map_info = {
            "origin": local_origin,
            "resolution": self.local_grid_resolution,
            "width": local_width,
            "height": local_height
        }

        if not self.map_received:
            self.map_received = True
            self.logger.loginfo("Local map generated from PointCloud. A* service is now active.")

    # 좌표 변환 함수들이 self.local_map_info를 사용하도록 수정
    def _world_to_pixel(self, world_point):
        origin_x, origin_y = self.local_map_info["origin"]
        resolution = self.local_map_info["resolution"]
        px = int((world_point[0] - origin_x) / resolution)
        py = int((world_point[1] - origin_y) / resolution)
        return (px, py)

    def _pixel_to_world(self, pixel_point):
        origin_x, origin_y = self.local_map_info["origin"]
        resolution = self.local_map_info["resolution"]
        wx = (pixel_point[0] + 0.5) * resolution + origin_x
        wy = (pixel_point[1] + 0.5) * resolution + origin_y
        return (wx, wy)

    def _find_nearest_valid_pixel(self, pixel):
        map_width = self.local_map_info["width"]
        map_height = self.local_map_info["height"]

        px, py = pixel
        px = max(0, min(px, map_width - 1))
        py = max(0, min(py, map_height - 1))
        pixel = (px, py)

        if self.inflated_map[pixel[1], pixel[0]] != 100:
            return pixel

        q = deque([pixel])
        visited = {pixel}

        while q:
            current_px, current_py = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                next_px, next_py = current_px + dx, current_py + dy
                neighbor = (next_px, next_py)
                if not (0 <= next_px < map_width and 0 <= next_py < map_height):
                    continue
                if neighbor in visited:
                    continue
                if self.inflated_map[next_py, next_px] != 100:
                    self.logger.loginfo(f"Found nearest valid pixel {neighbor} for original {pixel}")
                    return neighbor
                visited.add(neighbor)
                q.append(neighbor)

        self.logger.logwarn("Could not find any valid pixel on the map.")
        return None

    def _service_callback(self, req: AstarPath._request_class):
        if not self.map_received:
            self.logger.logwarn("A* service request received, but local map is not ready.")
            return AstarPathResponse(path=Path(), path_length=-1.0)

        map_height = self.local_map_info["height"]

        start_world = (req.start_x, req.start_y)
        end_world = (req.end_x, req.end_y)
        start_px, start_py_raw = self._world_to_pixel(start_world)
        end_px, end_py_raw = self._world_to_pixel(end_world)

        start_pixel_orig = (start_px, map_height - 1 - start_py_raw)
        end_pixel_orig = (end_px, map_height - 1 - end_py_raw)

        start_pixel = self._find_nearest_valid_pixel(start_pixel_orig)
        end_pixel = self._find_nearest_valid_pixel(end_pixel_orig)

        if start_pixel is None or end_pixel is None:
            self.logger.logerr("Could not find a valid start or end point on the map.")
            return AstarPathResponse(path=Path(), path_length=-1.0)

        self.logger.loginfo(f"Original start: {start_pixel_orig}, Using: {start_pixel}")
        self.logger.loginfo(f"Original end: {end_pixel_orig}, Using: {end_pixel}")

        planner = AStarPlanner(self.inflated_map, self.local_map_info["resolution"])
        pixel_path, path_length = planner.plan(start_pixel, end_pixel)

        response = AstarPathResponse()
        if pixel_path:
            ros_path = Path()
            ros_path.header.stamp = rospy.Time.now()
            ros_path.header.frame_id = "map"

            for p_px, p_py_flipped in pixel_path:
                p_py_raw = map_height - 1 - p_py_flipped
                world_x, world_y = self._pixel_to_world((p_px, p_py_raw))

                pose = PoseStamped(header=ros_path.header)
                pose.pose.position.x = world_x
                pose.pose.position.y = world_y
                pose.pose.orientation.w = 1.0
                ros_path.poses.append(pose)

            response.path = ros_path
            response.path_length = path_length
            self.path_pub.publish(ros_path)
        else:
            response.path_length = -1.0

        return response

if __name__ == '__main__':
    try:
        AStarServiceNode()
    except rospy.ROSInterruptException:
        pass