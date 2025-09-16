import os
import sys
sys.path.append("/ws/external")
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from ai_module.src.utils.logger import Logger


def iou_loss(pred_bbox_2d, gt_bbox_2d):
    x1_p, y1_p, x2_p, y2_p = pred_bbox_2d
    x1_g, y1_g, x2_g, y2_g = gt_bbox_2d

    # 두 박스의 교집합 영역
    x_left = max(x1_p, x1_g)
    y_top = max(y1_p, y1_g)
    x_right = min(x2_p, x2_g)
    y_bottom = min(y2_p, y2_g)
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # 두 박스의 면적
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = pred_area + gt_area - intersection_area
    iou = intersection_area / (union_area + 1e-6)

    loss = 1 - iou

    return loss


def get_xyzlwh(obj, dtype=None):
    center = obj.center
    scale = np.array(obj.max_bbox) - np.array(obj.min_bbox)
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    l, w, h = float(scale[0]), float(scale[1]), float(scale[2])
    r = R.from_quat([0.0, 0.0, 0.0, 1.0])
    _, _, yaw = r.as_euler('xyz', degrees=False)
    if dtype == 'str':
        return f"[{x:.2f},{y:.2f},{z:.2f},{l:.2f},{w:.2f},{h:.2f},{yaw:.2f}]"
    else:
        return x, y, z, l, w, h, yaw


def point3d_to_xyzlwh(point_3d, dtype=None):
    xmin, ymin, zmin = np.min(point_3d, axis=0)
    xmax, ymax, zmax = np.max(point_3d, axis=0)
    x, y, z = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    l, w, h = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    r = R.from_quat([0.0, 0.0, 0.0, 1.0])
    _, _, yaw = r.as_euler('xyz', degrees=False)
    if dtype == 'str':
        return f"[{x:.2f},{y:.2f},{z:.2f},{l:.2f},{w:.2f},{h:.2f},{yaw:.2f}]"
    else:
        return x, y, z, l, w, h, yaw


def get_pub_msg(obj):
    point3d = obj.corners_3d
    xyzlwh = point3d_to_xyzlwh(point3d, dtype='str')
    return f"python3 /ws/external/tools/publish.py marker /visual_grounding/markers {xyzlwh}"


def refine_bbox(initial_bbox_3d, selected_obj, selected_kfs):
    total_loss = 0.0
    for pid, kf in selected_kfs.items():
        gt_bbox_2d = selected_obj.get_bbox(kf_id=kf.id)

        point_pixel_idx = selected_obj.project_3d_point_to_image(
            initial_bbox_3d, kf.pose, kf.image_size, is_real_world=kf.is_real_world,
        )
        _a = np.array(kf.pose)
        umin, umax = int(np.min(point_pixel_idx[:, 0])), int(np.max(point_pixel_idx[:, 0]))
        if umin == umax:
            umax = umin + 1
        vmin, vmax = int(np.min(point_pixel_idx[:, 1])), int(np.max(point_pixel_idx[:, 1]))
        if vmin == vmax:
            vmax = vmin + 1
        pred_bbox_2d = [umin, vmin, umax, vmax]

        loss = iou_loss(pred_bbox_2d, gt_bbox_2d.data)
        total_loss += loss
    return total_loss


if __name__ == "__main__":
    from ai_module.src.visual_grounding.scripts.test.cluster import TestCluster
    logger = Logger()

    SCENE = "object_reference-hotel_room_1-1"
    if SCENE == "object_reference-hotel_room_1-1":
        instruction = "Find the picture above the suitcase furthest from the floor."
        action = 'find'
        target_name = "the picture above the suitcase furthest from the floor"
        candidate_names, reference_names = ['picture'], ['suitcase', 'floor']
    else:
        raise TypeError(f"SCENE must be in ['office_1', 'hotel_room_1', 'chinese_room'], but {SCENE} was given.")

    DATA_DIR = f"/ws/external/test_data/{SCENE}"
    MAP_DIR = os.path.join(DATA_DIR, "offline_map")
    KEYFRAMES_DIR = os.path.join(DATA_DIR, "keyframes")

    DATA_DIR = f"/ws/external/test_data/{SCENE}"
    MAP_DIR = os.path.join(DATA_DIR, "offline_map")
    KEYFRAMES_DIR = os.path.join(DATA_DIR, "keyframes")
    tester = TestCluster(
        logger=logger, action=action, target_name=target_name,
        candidate_names=candidate_names, reference_names=reference_names,
        DATA_DIR=DATA_DIR,
    )

    map_dirs = [os.path.join(MAP_DIR, d) for d in os.listdir(MAP_DIR)
            if os.path.isdir(os.path.join(MAP_DIR, d))]
    map_dir_sorted = sorted(map_dirs, key=os.path.getmtime)

    for dir in map_dir_sorted[-2:]:
        tester.timer_callback(None, dir=dir)
        # time.sleep(0.1)

    """ Refine 2D BBox """
    # Hyperparameter
    idx = 1

    # Get data
    self = tester
    candidate_objs = self.sg.get_candidate_entities(etype='detection')
    kfs = self.sg.keyframes
    eid2pids = kfs.entity_id2place_ids
    eids = candidate_objs.ids

    eid = eids[idx]
    selected_obj = candidate_objs.get_single(eid)
    pids = eid2pids[eid]
    selected_kfs = kfs.get(pids)

    styles = {'candidate': {'show': True, 'color': 'green'}, 'reference': {'show': True, 'color': 'blue'}}
    for pid in pids:
        kf = kfs.get_single(pid)
        kf.annotate(styles, node_name='hello')
        break
    xmin, ymin, xmax, ymax = selected_obj.get_bbox(kf_id=pid).data
    u, v = ((xmax+xmin)/2.0, (ymax+ymin)/2.0)

    CAMERA_PARA = {"hfov": 360, "vfov": 120, "width": 1920, "height": 640}
    L2C_PARA = {"x": 0, "y": 0, "z": 0.235, "roll": 0, "pitch": 0, "yaw": 0}

    def _rot_mats_from_L2C(L2C_PARA):
        # 정방향에서 roll/pitch/yaw에 -가 붙은 정의와 동일하게 생성
        rR = -L2C_PARA["roll"] * np.pi / 180.0
        rP = -L2C_PARA["pitch"] * np.pi / 180.0
        rY = -L2C_PARA["yaw"] * np.pi / 180.0
        sR, cR = np.sin(rR), np.cos(rR)
        sP, cP = np.sin(rP), np.cos(rP)
        sY, cY = np.sin(rY), np.cos(rY)
        Rz = np.array([[cY, -sY, 0], [sY, cY, 0], [0, 0, 1]], dtype=float)
        Ry = np.array([[cP, 0, sP], [0, 1, 0], [-sP, 0, cP]], dtype=float)
        Rx = np.array([[1, 0, 0], [0, cR, -sR], [0, sR, cR]], dtype=float)
        return Rx, Ry, Rz


    def pixels2scan(  # TODO
            u, v, depth=1,
            pixel_center=(0.0, 0.0), v_scale_mode="width", depth_type="hori",
            flip_h=-1.0, flip_v=-1.0, yaw_bias_rad=0.0, pitch_bias_rad=0.0,
            vertPixelOffset=0, cameraOffsetZ=0, input_is_one_based=False,
    ):
        imageWidth, imageHeight = CAMERA_PARA['width'], CAMERA_PARA['height']

        # ----- 1) 1-based와 int 캐스팅 보정: 0-based로 내리고, 픽셀 중심(+0.5) 복원 -----
        # 정방향: u = -W/(2π)*atan2(y,x) + W/2 + 1  (그 뒤 int 캐스팅)
        # 역방향: (정수 u) -> u0(0-based) -> 중심 u_c(+0.5)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        r = np.asarray(depth, dtype=np.float64)  # horiDis

        if input_is_one_based:
            u_c = (u - 1.0) + pixel_center[0]
            v_c = (v - 1.0 - vertPixelOffset) + pixel_center[1]
        else:
            u_c = u + pixel_center[0]
            v_c = (v - vertPixelOffset) + pixel_center[1]

        # 2) 각도 복원 (원 코드와 일치: v도 W로 스케일)
        k_h = 2.0 * np.pi / imageWidth
        if v_scale_mode == "width":
            k_v = 2.0 * np.pi / imageWidth  # (원 코드)
        elif v_scale_mode == "height":
            k_v = np.pi / imageHeight  # (일반 equirect: φ ∈ [-π/2, π/2])
        else:
            raise ValueError("v_scale_mode must be 'width' or 'height'.")

        # θ, φ 복원 + 상수 바이어스 보정
        theta = flip_h * k_h * (u_c - imageWidth / 2.0) + yaw_bias_rad  # atan2(y,x)
        phi = flip_v * k_v * (v_c - imageHeight / 2.0) + pitch_bias_rad  # atan2(z,r)

        # 3) 카메라 좌표
        if depth_type == "hori":  # r = sqrt(x^2+y^2)가 주어졌을 때
            x_cam = r * np.cos(theta)
            y_cam = r * np.sin(theta)
            z_cam = r * np.tan(phi)
        elif depth_type == "range":  # s = 레이 거리(일반적)일 때  ✅
            x_cam = r * np.cos(phi) * np.cos(theta)
            y_cam = r * np.cos(phi) * np.sin(theta)
            z_cam = r * np.sin(phi)
        else:
            raise ValueError("depth_type must be 'range' or 'hori'")
        p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1) + np.array([0., 0., float(cameraOffsetZ)])

        # 4) 회전/병진 역적용 (정방향과 부호/순서 정확히 맞춤)
        Rx, Ry, Rz = _rot_mats_from_L2C(L2C_PARA)
        lidarX, lidarY, lidarZ = L2C_PARA["x"], L2C_PARA["y"], L2C_PARA["z"]

        # 역변환: p_lidar = ((p_cam) @ (Rx^T)) @ (Ry^T) @ (Rz^T) + lidar_offset
        # (벡터가 row 형태이므로 우측 곱 방식 유지)
        p_lidar = p_cam @ Rx.T @ Ry.T @ Rz.T
        lidar_offset = np.array([lidarX, lidarY, lidarZ], dtype=float)
        p_body = p_lidar - lidar_offset
        return p_body

    for id, depth in enumerate([0.1, 0.5, 1.0, 2.0]):
        uuvv = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
        points_body = pixels2scan(
            uuvv[:,0], uuvv[:,1], depth=depth,
            depth_type="range", input_is_one_based=True
        ) # (4, 3)

        pose = np.array(kf.pose)
        R_b2w = pose[:3, :3]
        t_b2w = pose[:3, 3]

        points = points_body @ R_b2w.T + t_b2w
        xyzlwh = point3d_to_xyzlwh(points, dtype='str')
        command_pub_refine = f"python3 /ws/external/tools/publish.py marker /visual_grounding/markers {xyzlwh} --color '1.0,0.0,0.0,0.8' --id {id}"
        print(command_pub_refine)

    """ Refine 3D BBox """
    # Hyperparameter
    idx = 1

    # Get data
    self = tester
    candidate_objs = self.sg.get_candidate_entities(etype='detection')
    kfs = self.sg.keyframes
    eid2pids = kfs.entity_id2place_ids
    eids = candidate_objs.ids

    eid = eids[idx]
    selected_obj = candidate_objs.get_single(eid)
    pids = eid2pids[eid]
    selected_kfs = kfs.get(pids)
    initial_bbox_3d = selected_obj.corners_3d

    # Optimize
    refined_result = minimize(
        refine_bbox,
        initial_bbox_3d,
        args=(selected_obj, selected_kfs,),
        method='Nelder-Mead',
        options={'disp': True}
    )

    # Get the result
    refined_point_3d = refined_result.x.reshape(-1, 3)
    xyzlwh = point3d_to_xyzlwh(refined_point_3d, dtype='str')

    # Publish original bbox_3d
    command_pub_origin = get_pub_msg(selected_obj)
    command_pub_refine = f"python3 /ws/external/tools/publish.py marker /visual_grounding/markers {xyzlwh} --color '1.0,0.0,0.0,0.8'"

    print(command_pub_origin)
    print(command_pub_refine)
