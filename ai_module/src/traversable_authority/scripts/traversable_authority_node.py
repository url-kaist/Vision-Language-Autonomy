#!/usr/bin/env python3
import os
import json
import threading
import numpy as np
import rospy
import sys
import cv2
sys.path.append('/ws/external/ai_module/src/traversable_authority')
sys.path.append('/ws/external/ai_module/src/visual_grounding/scripts')
sys.path.append('/ws/external/')

from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

from traversable_authority.srv import (
    BlockBBox, BlockBBoxRequest, BlockBBoxResponse,
    BlockSegment, BlockSegmentRequest, BlockSegmentResponse,
)

from ai_module.src.utils.utils import pointcloud2_to_xy_array

from ai_module.src.utils.logger import Logger


class TraversableAuthority:
    """
    - /traversable_area (raw)를 구독
    - 내부에 XY 포인트를 보관
    - /traversable/apply_bbox, /traversable/apply_segment 서비스로 '차단 연산'을 적용
    - 결과를 /traversable_area_filtered 로 (latched) publish
    - /traversable/reset(Trigger)로 초기화
    """
    def __init__(self):
        quiet = rospy.get_param('~quiet', False)

        self.lock = threading.RLock()

        self.raw_topic      = rospy.get_param("~raw_topic", "/traversable_area")
        self.filtered_topic = rospy.get_param("~filtered_topic", "/traversable_area_filtered")

        self.raw_xy = None         # np.ndarray [N,2]
        self.filtered_xy = None    # np.ndarray [M,2]
        self.update_count = 0
        self.traversable_area_bounds = None

        """ Logger """
        self.log_dir = "/ws/external/log/traversable_authority"
        self.logger = Logger(
            quiet=quiet, prefix='TraversableAuthority', log_path=os.path.join(self.log_dir, "traversable_authority.log"))

        self.is_real_world = rospy.get_param('~quiet', False)
        if self.is_real_world:
            self.logger.loginfo(f"Hello Real World!!")
            self.frame_id = "world"
        else:
            self.frame_id = "map"

        """ ROS """
        # Subscribers
        self.sub_raw = rospy.Subscriber(self.raw_topic, PointCloud2, self.cb_raw, queue_size=1)

        # Publishers
        self.pub_filtered = rospy.Publisher(self.filtered_topic, PointCloud2, queue_size=1, latch=True)

        # Services
        self.srv_bbox    = rospy.Service("/traversable/apply_bbox",    BlockBBox,    self.srv_apply_bbox)
        self.srv_segment = rospy.Service("/traversable/apply_segment", BlockSegment, self.srv_apply_segment)
        self.srv_reset   = rospy.Service("/traversable/reset",         Trigger,      self.srv_reset)

        rospy.loginfo("[TraversableAuthority] ready.")

    # -------- subscribers --------
    def cb_raw(self, msg: PointCloud2):     
        if self.raw_xy is None:
            traversable_pts, _ = pointcloud2_to_xy_array(msg)
            self.logger.log(f"Received traversable area with {len(traversable_pts)} points")

            with self.lock:
                self.raw_xy = traversable_pts
                self.filtered_xy = traversable_pts.copy()

                self.save_traversable_debug_image(self.raw_xy, os.path.join(self.log_dir, 'traversable_area.png'))
        
        self.publish_filtered()

    # -------- services --------
    def srv_apply_bbox(self, req: BlockBBoxRequest) -> BlockBBoxResponse:
        with self.lock:
            if self.filtered_xy is None:
                return BlockBBoxResponse(False, "no traversable map yet")
            
            self.logger.loginfo(f"apply_bbox called: min={req.min_bbox}, max={req.max_bbox}, infl={req.inflation}")

            min_bbox = np.array(req.min_bbox, dtype=float)
            max_bbox = np.array(req.max_bbox, dtype=float)
            infl = float(req.inflation)

            min_bbox -= infl
            max_bbox += infl

            xy = self.filtered_xy
            keep = ~(
                (xy[:,0] >= min_bbox[0]) & (xy[:,0] <= max_bbox[0]) &
                (xy[:,1] >= min_bbox[1]) & (xy[:,1] <= max_bbox[1])
            )
            self.filtered_xy = xy[keep]

            self.publish_filtered()
            self.update_count += 1
            self.logger.log(f"Applied bbox filter, remaining points: {self.filtered_xy.shape[0]} (update count: {self.update_count})")
            self.save_traversable_debug_image(
                self.filtered_xy, os.path.join(self.log_dir, f'traversable_area_filtered_{self.update_count}.png'),
                draw_objects=[(min_bbox, max_bbox)]
            )
            return BlockBBoxResponse(True, f"applied bbox, remain={self.filtered_xy.shape[0]}")

    def srv_apply_segment(self, req: BlockSegmentRequest) -> BlockSegmentResponse:
        with self.lock:
            if self.filtered_xy is None:
                return BlockSegmentResponse(False, "no traversable map yet")

            a = np.array(req.center1[:2], dtype=float)
            b = np.array(req.center2[:2], dtype=float)
            r = float(req.radius)

            xy = self.filtered_xy
            v = b - a
            L = np.linalg.norm(v)
            if L < 1e-9:
                d = np.linalg.norm(xy - a[None,:], axis=1)
                keep = d >= r
            else:
                v = v / L
                diff = xy - a[None,:]
                t = diff @ v
                t = np.clip(t, 0.0, L)
                closest = a[None,:] + t[:,None]*v[None,:]
                d = np.linalg.norm(xy - closest, axis=1)
                keep = d >= r

            self.filtered_xy = xy[keep]
            self.publish_filtered()
            self.update_count += 1
            self.logger.log(f"Applied segment filter, remaining points: {self.filtered_xy.shape[0]} (update count: {self.update_count})")
            self.save_traversable_debug_image(
                self.filtered_xy, os.path.join(self.log_dir, f'traversable_area_filtered_{self.update_count}.png'),
                draw_objects=[(a-r, a+r), (b-r, b+r)]
            )
            return BlockSegmentResponse(True, f"applied segment, remain={self.filtered_xy.shape[0]}")

    def srv_reset(self, _):
        with self.lock:
            if self.raw_xy is None:
                return TriggerResponse(success=False, message="no raw map")
            self.filtered_xy = self.raw_xy.copy()
            self.update_count = 0
        self.publish_filtered()
        return TriggerResponse(success=True, message="reset to raw")

    # -------- utils --------
    def publish_filtered(self):
        with self.lock:
            if self.filtered_xy is None:
                return
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.frame_id
            pts_xyz = np.hstack([self.filtered_xy, np.zeros((self.filtered_xy.shape[0],1), dtype=float)])
            msg = pc2.create_cloud_xyz32(header, pts_xyz.tolist())
        self.pub_filtered.publish(msg)

    # -------- visualization --------
    def save_traversable_debug_image(
        self, pts,
        save_path="/tmp/traversable.png",
        resolution=0.05,          # 1 pixel = 5cm
        margin=0.5,               # 경계 여유(m)
        draw_objects=None        # [(min_bbox, max_bbox), ...] in world XY, 선택
    ):

        # 이미지 경계 계산(최초 1회는 포인트 기반, 이후에는 고정된 bounds 사용 가능)
        if self.traversable_area_bounds is None:
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            xmin -= margin; ymin -= margin
            xmax += margin; ymax += margin
            self.traversable_area_bounds = (xmin, xmax, ymin, ymax)
        else:
            xmin, xmax, ymin, ymax = self.traversable_area_bounds

        # 해상도 기반 픽셀 크기
        width  = int(np.ceil((xmax - xmin) / resolution))
        height = int(np.ceil((ymax - ymin) / resolution))
        width  = max(1, width)
        height = max(1, height)

        # 배경 초기화
        canvas = np.zeros((height, width, 3), dtype=np.uint8)  # BGR
        BG, FG = (30, 30, 30), (255, 255, 255)
        canvas[:] = BG

        # 월드→이미지 좌표 변환
        # u = (x - xmin)/res, v = (ymax - y)/res  (상단 원점)
        us = ((pts[:, 0] - xmin) / resolution).astype(np.int32)
        vs = ((ymax - pts[:, 1]) / resolution).astype(np.int32)

        # 이미지 경계 내로 클리핑
        valid = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
        us, vs = us[valid], vs[valid]

        # 포인트 렌더링
        canvas[vs, us] = FG

        # 선택: 객체 AABB 그리기
        # draw_objects: [(min_bbox, max_bbox), ...], 각 bbox는 [x_min, y_min], [x_max, y_max] in world
        if draw_objects:
            for minb, maxb in draw_objects:
                x0, y0 = minb[:2]; x1, y1 = maxb[:2]
                u0 = int((x0 - xmin) / resolution); v0 = int((ymax - y0) / resolution)
                u1 = int((x1 - xmin) / resolution); v1 = int((ymax - y1) / resolution)
                pt1 = (np.clip(u0, 0, width - 1), np.clip(v0, 0, height - 1))
                pt2 = (np.clip(u1, 0, width - 1), np.clip(v1, 0, height - 1))
                cv2.rectangle(canvas, pt1, pt2, (0, 0, 255), 2)

        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, canvas)

        # 메타데이터 로그
        self.logger.loginfo(f"Saved traversable image at {save_path} with size ({width}x{height}) and resolution {resolution}m/px")


if __name__ == "__main__":
    rospy.init_node("traversable_authority")
    TraversableAuthority()
    rospy.spin()
