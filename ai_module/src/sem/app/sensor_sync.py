import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from collections import deque
from utils import convert_pointcloud2_to_xyz


class SensorSyncManager:
    def __init__(self, image_queue, cloud_queue, odom_queue, time_bias=0):
        self.image_queue = image_queue
        self.cloud_queue = cloud_queue
        self.odom_queue = odom_queue
        self.time_bias = time_bias
        self.last_image_stamp = None
        self.last_cloud_stamp = None
        self.last_odom_t0 = None


    def get_latest_image(self):
        return self.image_queue[-1] if self.image_queue else (None, None)

    def find_closest(self, queue, target_stamp):
        # min_diff = float("inf")
        # closest_item = None
        # for stamp, item in queue:
        #     diff = abs(stamp - target_stamp)
        #     if diff < min_diff:
        #         min_diff = diff
        #         closest_item = item
        # # return (stamp, closest_item) if closest_item else (None, None)
        # if closest_item is not None:
        #     return (stamp, closest_item)
        # else:
        #     return (None, None)
        
        if not queue:
            return None, None
        queue = list(queue)
        # 중심 인덱스 찾기
        center_idx = min(range(len(queue)), key=lambda i: abs(queue[i][0] - target_stamp))

        # 앞뒤 3개 포함되도록 인덱스 범위 설정
        start_idx = max(0, center_idx - 3)
        end_idx = min(len(queue), center_idx + 4)  # end_idx는 exclusive, 총 7개 되도록

        # 부족하면 앞쪽에서 더 채움
        while (end_idx - start_idx) < 7 and start_idx > 0:
            start_idx -= 1
        while (end_idx - start_idx) < 7 and end_idx < len(queue):
            end_idx += 1

        selected = queue[start_idx:end_idx]
        if not selected:
            return None, None

        # 중심 timestamp는 가장 가까운 것
        center_stamp = queue[center_idx][0]
        # cloud는 numpy concatenate
        clouds = [cloud for _, cloud in selected]
        combined_cloud = np.concatenate(clouds, axis=0)

        return center_stamp, combined_cloud

    def find_odom_neighbors(self, target_stamp):
        sorted_queue = sorted(self.odom_queue, key=lambda x: x[0], reverse=False)
        if not self.odom_queue or target_stamp < self.odom_queue[0][0] or target_stamp > self.odom_queue[-1][0]:
            return None, None, None, None

        for i in range(len(sorted_queue) - 1):
            if sorted_queue[i][0] <= target_stamp <= sorted_queue[i+1][0]:
                t0, odom0 = sorted_queue[i]
                t1, odom1 = sorted_queue[i+1]
                return t0, t1, odom0, odom1
        return None, None, None, None

    def interpolate_pose(self, target_stamp, t0, t1, odom0, odom1):
        if None in [t0, t1, odom0, odom1]:
            return None

        ratio = (target_stamp - t0) / (t1 - t0) if t1 != t0 else 0.5

        pos = (1 - ratio) * odom0["position"] + ratio * odom1["position"]
        vel = (1 - ratio) * odom0["linear_velocity"] + ratio * odom1["linear_velocity"]

        rot_slerp = Slerp(
            [0, 1], Rotation.from_quat([odom0["orientation"], odom1["orientation"]])
        )
        quat = rot_slerp(ratio).as_quat()
        rot = Rotation.from_quat(quat).as_matrix()

        return {
            "position": pos,
            "velocity": vel,
            "rotation": rot,
            "quat": quat,
        }

    def get_synced_data(self):
        image_stamp, image = self.get_latest_image()
        if image_stamp is None:
            return None

        t0, t1, odom0, odom1 = self.find_odom_neighbors(image_stamp + self.time_bias)

        interp_pose = self.interpolate_pose(image_stamp, t0, t1, odom0, odom1)

        if interp_pose is None:
            # print("No odom data available")
            return None

        cloud_stamp, cloud_msg = self.find_closest(self.cloud_queue, image_stamp)
        # print(f"image queue len: {len(self.image_queue)}, t0: {t0} <- {image_stamp - t0} -> image: {image_stamp} <- {t1 - image_stamp} -> t1: {t1}, close cloud: {cloud_stamp}, t1 - image_stamp: {t1 - image_stamp}")
        if t1 - image_stamp > 1000: #> 100000000:
            
            print("Image is too old, skipping")
            return None

        if cloud_msg is None or not self.image_queue or not self.cloud_queue:
            print("No cloud / image data available")
            return None
        
        self.last_image_stamp = image_stamp
        self.last_cloud_stamp = cloud_stamp
        self.last_odom_t0 = t0

        return {
            "image_stamp": image_stamp,
            "image": image,
            "cloud_stamp": cloud_stamp,
            "cloud_msg": cloud_msg,
            "pose": interp_pose,
        }
        
    def get_synced_data1(self):
        image_stamp, image = self.get_latest_image()
        if image_stamp is None:
            return None

        t0, t1, odom0, odom1 = self.find_odom_neighbors(image_stamp + self.time_bias)

        interp_pose = self.interpolate_pose(image_stamp, t0, t1, odom0, odom1)

        if interp_pose is None:
            # print("No odom data available")
            return None
        
        print(f"t0: {t0} <- {image_stamp - t0} -> image: {image_stamp} <- {t1 - image_stamp} -> t1: {t1}")
        # diff_position = odom1["position"] - interp_pose["position"]
        # diff_rotation = odom1["orientation"] - interp_pose["quat"]
        # print(f"diff_position: {diff_position}")
        # print(f"diff_rotation: {diff_rotation}")
        
        if t1 - image_stamp > 1000:
            return None

        # cloud_stamp, cloud_msg = self.find_closest(self.cloud_queue, image_stamp)

        # if cloud_msg is None or not self.image_queue or not self.cloud_queue:
            # print("No cloud / image data available")
            # return None
        
        self.last_image_stamp = image_stamp
        # self.last_cloud_stamp = cloud_stamp
        self.last_odom_t0 = t0

        return {
            "image_stamp": image_stamp,
            "image": image,
            # "cloud_stamp": cloud_stamp,
            # "cloud_msg": cloud_msg,
            "pose": interp_pose,
        }
        
    def clear_queues(self):
        def purge_deque(dq: deque, cutoff, keep_equal=True):
            while dq and (dq[0][0] < cutoff if keep_equal else dq[0][0] <= cutoff):
                dq.popleft()

        if self.last_image_stamp is not None:
            purge_deque(self.image_queue, self.last_image_stamp, keep_equal=True)

        if self.last_cloud_stamp is not None:
            purge_deque(self.cloud_queue, self.last_cloud_stamp, keep_equal=True)

        # if self.last_odom_t0 is not None:
            # purge_deque(self.odom_queue, self.last_odom_t0, keep_equal=True)
            # while self.odom_queue and (self.odom_queue[0][0] < self.last_odom_t0 - 2.0):
                # self.odom_queue.popleft()


