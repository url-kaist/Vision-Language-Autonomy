import pickle
import numpy as np
import os
import cv2
import open3d as o3d

from scipy.spatial.transform import Rotation

# def scan2pixels(laserCloud, LIDAR_PARA, CAMERA_PARA):
#     lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
#     lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
#     lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
#     lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]

#     cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
#     camRoll = CAMERA_PARA["roll"]
#     camPitch = CAMERA_PARA["pitch"]
#     camYaw = CAMERA_PARA["yaw"]

#     imageWidth = CAMERA_PARA["width"]
#     imageHeight = CAMERA_PARA["height"]
#     cameraOffsetZ= 0   #  additional pixel offset due to image cropping?
#     vertPixelOffset=0 #  additional vertical pixel offset due to image cropping

#     sinLidarRoll = np.sin(lidarRoll*np.pi / 180.)
#     cosLidarRoll = np.cos(lidarRoll*np.pi / 180.)
#     sinLidarPitch = np.sin(lidarPitch*np.pi / 180.)
#     cosLidarPitch = np.cos(lidarPitch*np.pi / 180.)
#     sinLidarYaw = np.sin(lidarYaw*np.pi / 180.)
#     cosLidarYaw = np.cos(lidarYaw*np.pi / 180.)

#     cloud = laserCloud[:, :3] - lidar_offset
#     R_z = np.array([[cosLidarYaw, -sinLidarYaw, 0], [sinLidarYaw, cosLidarYaw, 0], [0, 0, 1]])
#     R_y = np.array([[cosLidarPitch, 0, sinLidarPitch], [0, 1, 0], [-sinLidarPitch, 0, cosLidarPitch]])
#     R_x = np.array([[1, 0, 0], [0, cosLidarRoll, -sinLidarRoll], [0, sinLidarRoll, cosLidarRoll]])

#     camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
#     camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
#     camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])

#     cloud = cloud @ R_z @ R_y @ R_x
#     cloud = cloud - cam_offset
#     cloud = cloud @ camR_z @ camR_y @ camR_x

#     horiDis = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
#     horiPixelID = (imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 0], cloud[:, 2]) + imageWidth / 2 + 1).astype(int)
#     vertPixelID = (imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 1], horiDis) + imageHeight / 2 + 1 + vertPixelOffset).astype(int)
#     PixelDepth = horiDis

#     point_pixel_idx = np.array([horiPixelID, vertPixelID, PixelDepth]).T

#     return point_pixel_idx.astype(int)


def scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA):
    lidarX = L2C_PARA["x"]  #   lidarXStack[imageIDPointer]
    lidarY = L2C_PARA["y"]  # idarYStack[imageIDPointer]
    lidarZ = L2C_PARA["z"]  # lidarZStack[imageIDPointer]
    lidarRoll = -L2C_PARA["roll"]  #  lidarRollStack[imageIDPointer]
    lidarPitch = -L2C_PARA["pitch"]  # lidarPitchStack[imageIDPointer]
    lidarYaw = -L2C_PARA["yaw"]  # lidarYawStack[imageIDPointer]

    imageWidth = CAMERA_PARA["width"]
    imageHeight = CAMERA_PARA["height"]
    cameraOffsetZ = 0  #  additional pixel offset due to image cropping?
    vertPixelOffset = 0  #  additional vertical pixel offset due to image cropping

    sinLidarRoll = np.sin(lidarRoll * np.pi / 180.0)
    cosLidarRoll = np.cos(lidarRoll * np.pi / 180.0)
    sinLidarPitch = np.sin(lidarPitch * np.pi / 180.0)
    cosLidarPitch = np.cos(lidarPitch * np.pi / 180.0)
    sinLidarYaw = np.sin(lidarYaw * np.pi / 180.0)
    cosLidarYaw = np.cos(lidarYaw * np.pi / 180.0)

    lidar_offset = np.array([lidarX, lidarY, lidarZ])
    camera_offset = np.array([0, 0, cameraOffsetZ])

    cloud = laserCloud[:, :3] - lidar_offset
    R_z = np.array(
        [[cosLidarYaw, -sinLidarYaw, 0], [sinLidarYaw, cosLidarYaw, 0], [0, 0, 1]]
    )
    R_y = np.array(
        [
            [cosLidarPitch, 0, sinLidarPitch],
            [0, 1, 0],
            [-sinLidarPitch, 0, cosLidarPitch],
        ]
    )
    R_x = np.array(
        [[1, 0, 0], [0, cosLidarRoll, -sinLidarRoll], [0, sinLidarRoll, cosLidarRoll]]
    )
    cloud = cloud @ R_z @ R_y @ R_x
    cloud = cloud - camera_offset

    horiDis = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
    horiPixelID = (
        -imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 1], cloud[:, 0])
        + imageWidth / 2
        + 1
    ).astype(int)
    vertPixelID = (
        -imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 2], horiDis)
        + imageHeight / 2
        + 1
        + vertPixelOffset
    ).astype(int)
    PixelDepth = horiDis

    point_pixel_idx = np.array([horiPixelID, vertPixelID, PixelDepth]).T

    return point_pixel_idx.astype(int)


def scan2pixels_wheelchair(laserCloud):
    # project scan points to image pixels
    # https://github.com/jizhang-cmu/cmu_vla_challenge_unity/blob/noetic/src/semantic_scan_generation/src/semanticScanGeneration.cpp

    # Input:
    # [#points, 3], x-y-z coordinates of lidar points

    # Output:
    #    point_pixel_idx['horiPixelID'] : horizontal pixel index in the image coordinate
    #    point_pixel_idx['vertPixelID'] : vertical pixel index in the image coordinate

    L2C_PARA = {
        "x": 0,
        "y": 0,
        "z": 0.235,
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }  #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    CAMERA_PARA = {
        "hfov": 360,
        "vfov": 120,
        "width": 1920,
        "height": 640,
    }  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom
    LIDAR_PARA = {"hfov": 360, "vfov": 30}

    return scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA)

def scan2pixels_jackal(laserCloud):
    L2C_PARA = {
        "x": 0,
        "y": 0,
        "z": 0.21,
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }  #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    CAMERA_PARA = {
        "hfov": 360,
        "vfov": 120,
        "width": 1920,
        "height": 640,
    }  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom
    LIDAR_PARA = {"hfov": 360, "vfov": 30}

    return scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA)


def scan2pixels_mcanum(laserCloud):
    CAMERA_PARA = {
        "x": -0.12,
        "y": -0.075,
        "z": 0.255,
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
        "hfov": 360,
        "vfov": 120,
        "width": 1920,
        "height": 640,
    }  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom
    LIDAR_PARA = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"]  #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"]  # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]  # lidarYawStack[imageIDPointer]
    lidarR_z = np.array(
        [
            [np.cos(lidarYaw), -np.sin(lidarYaw), 0],
            [np.sin(lidarYaw), np.cos(lidarYaw), 0],
            [0, 0, 1],
        ]
    )
    lidarR_y = np.array(
        [
            [np.cos(lidarPitch), 0, np.sin(lidarPitch)],
            [0, 1, 0],
            [-np.sin(lidarPitch), 0, np.cos(lidarPitch)],
        ]
    )
    lidarR_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(lidarRoll), -np.sin(lidarRoll)],
            [0, np.sin(lidarRoll), np.cos(lidarRoll)],
        ]
    )
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array(
        [
            [np.cos(camYaw), -np.sin(camYaw), 0],
            [np.sin(camYaw), np.cos(camYaw), 0],
            [0, 0, 1],
        ]
    )
    camR_y = np.array(
        [
            [np.cos(camPitch), 0, np.sin(camPitch)],
            [0, 1, 0],
            [-np.sin(camPitch), 0, np.cos(camPitch)],
        ]
    )
    camR_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(camRoll), -np.sin(camRoll)],
            [0, np.sin(camRoll), np.cos(camRoll)],
        ]
    )
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (
        CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2])
        + CAMERA_PARA["width"] / 2
        + 1
    ).astype(int)
    vertPixelID = (
        CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis)
        + CAMERA_PARA["height"] / 2
        + 1
    ).astype(int)
    pixelDepth = horiDis

    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T

    return point_pixel_idx

## for GT
def generate_sem_cloud(
    cloud: np.ndarray,
    masks,
    labels,
    R_b2w,
    t_b2w,
    sem_image=None,
    platform="wheelchair",
):
    # Project the cloud points to image pixels
    if platform == "wheelchair":
        point_pixel_idx = scan2pixels_wheelchair(cloud)
    elif platform == "mcanum":
        point_pixel_idx = scan2pixels_mcanum(cloud)
    elif platform == "jackal":
        point_pixel_idx = scan2pixels_jackal(cloud)
    else:
        raise ValueError

    if masks is None:
        return None
    image_shape = None
    for m in masks:
        if m is None:
            continue
        image_shape = m.shape
        break
    # print(f"image_shape {image_shape}")
    if image_shape is None:
        return
    out_of_bound_filter = (
        (point_pixel_idx[:, 0] >= 0)
        & (point_pixel_idx[:, 0] < image_shape[1])
        & (point_pixel_idx[:, 1] >= 0)
        & (point_pixel_idx[:, 1] < image_shape[0])
    )

    point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    cloud = cloud[out_of_bound_filter]

    horDis = point_pixel_idx[:, 2]
    point_pixel_idx = point_pixel_idx.astype(int)

    all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
    obj_cloud_world_list = []
    colors = []
    available_indices = []
    
    for i in range(len(labels)):
        obj_colors = []
        obj_mask = masks[i]
        if obj_mask is None:
            continue
        cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)

        all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
        obj_cloud = cloud[cloud_mask]
        obj_pixel = point_pixel_idx[cloud_mask]

        uv = obj_pixel[:, :2]
        depth = obj_pixel[:, 2]
        key = uv[:, 1] * image_shape[1] + uv[:, 0]

        sort_idx = np.argsort(depth)  # depth 작은 순 정렬
        key_sorted = key[sort_idx]
        _, unique_indices = np.unique(key_sorted, return_index=True)
        best_indices = sort_idx[unique_indices]

        obj_cloud = obj_cloud[best_indices]
        obj_pixel = obj_pixel[best_indices]

        obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
        for j in range(obj_pixel.shape[0]):
            px, py = obj_pixel[j, 0], obj_pixel[j, 1]
            obj_colors.append(sem_image[py, px])

        if not obj_colors:
            # print("no obj_colors for ",labels[i])
            continue
        unique_colors, counts = np.unique(obj_colors, axis=0, return_counts=True)
        majority_color_index = np.argmax(counts)
        majority_color = unique_colors[majority_color_index]
        if len(majority_color) != 3:
            continue
        
        majority_color_rgb = majority_color.astype(sem_image.dtype)

        colors.append(majority_color_rgb)
        obj_cloud_world_list.append(obj_cloud_world)
        available_indices.append(i)

    if sem_image is not None:
        all_obj_point_pixel_idx = point_pixel_idx

        maxRange = 6.0
        pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
        sem_image[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = (
            np.array([pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)]).T
        )  # assume RGB

    return obj_cloud_world_list, colors, available_indices

def generate_seg_cloud(
    cloud: np.ndarray,
    masks,
    labels,
    R_b2w,
    t_b2w,
    image_src=None,
    viz_cloud=False,
    platform="wheelchair",
):
    # Project the cloud points to image pixels
    if platform == "wheelchair":
        point_pixel_idx = scan2pixels_wheelchair(cloud)
    elif platform == "mcanum":
        point_pixel_idx = scan2pixels_mcanum(cloud)
    elif platform == "jackal":
        point_pixel_idx = scan2pixels_jackal(cloud)
    else:
        raise ValueError

    if masks is None:
        return None, None

    image_shape = masks[0].shape

    out_of_bound_filter = (
        (point_pixel_idx[:, 0] >= 0)
        & (point_pixel_idx[:, 0] < image_shape[1])
        & (point_pixel_idx[:, 1] >= 0)
        & (point_pixel_idx[:, 1] < image_shape[0])
    )

    point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    cloud = cloud[out_of_bound_filter]


    horDis = point_pixel_idx[:, 2]
    point_pixel_idx = point_pixel_idx.astype(int)

    # if image_src is not None:
    #     # project points to image
    #     image_src[point_pixel_idx[:, 1], point_pixel_idx[:, 0]] = [0, 0, 255] # return by reference
    #     # cv2.imwrite(f'test.png', image_src)

    # cloud_annotation = np.zeros([cloud.shape[0], 2], dtype=np.float32)

    all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
    obj_cloud_world_list = []
    colors = []
    for i in range(len(labels)):
        obj_mask = masks[i]
        cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
        all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
        obj_cloud = cloud[cloud_mask]
        obj_pixel = point_pixel_idx[cloud_mask]

        uv = obj_pixel[:, :2]
        depth = obj_pixel[:, 2]
        key = uv[:, 1] * image_shape[1] + uv[:, 0]

        sort_idx = np.argsort(depth)  # depth 작은 순 정렬
        key_sorted = key[sort_idx]
        _, unique_indices = np.unique(key_sorted, return_index=True)
        best_indices = sort_idx[unique_indices]

        obj_cloud = obj_cloud[best_indices]
        obj_pixel = obj_pixel[best_indices]

        # obj_cloud_list.append(obj_cloud)

        obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
        obj_cloud_world_list.append(obj_cloud_world)
        for j in range(obj_pixel.shape[0]):
            px, py = obj_pixel[j, 0], obj_pixel[j, 1]
            colors.append(image_src[py, px])

    if image_src is not None:
        all_obj_point_pixel_idx = point_pixel_idx

        # all_obj_cloud = cloud[all_obj_cloud_mask]
        # all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
        # horDis = horDis[all_obj_cloud_mask]
        maxRange = 6.0
        pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
        image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = (
            np.array([pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)]).T
        )  # assume RGB

    return obj_cloud_world_list, colors

    # visualize the colored cloud
    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cloud)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd, coord])
def generate_seg_comp_cloud_jackal(
    cloud: np.ndarray,
    masks,
    labels,
    R_b2w,
    t_b2w,
    image_src=None,
    viz_cloud=False,
    platform="jackal",
):
    point_pixel_idx = scan2pixels_jackal(cloud)

    if masks is None:
        return None, None

    image_shape = masks[0].shape

    point_pixel_idx[:,1] += 200

    out_of_bound_filter = (
        (point_pixel_idx[:, 0] >= 0)
        & (point_pixel_idx[:, 0] < image_shape[1])
        & (point_pixel_idx[:, 1] >= 0)
        & (point_pixel_idx[:, 1] < image_shape[0])
    )

    vis = image_src.copy()
    u = point_pixel_idx[:,0]
    v = point_pixel_idx[:,1]
    valid = (u>=0)&(u<image_shape[1])&(v>=0)&(v<image_shape[0])
    u,v = u[valid], v[valid]
    vis[v,u] = (0,0,255)
    # cv2.imshow("Reproj", vis)
    # cv2.waitKey(1)

    # out_of_bound_filter = (
    #     (u >= 0)
    #     & (u < image_shape[1])
    #     & (v >= 0)
    #     & (v < image_shape[0])
    # )

    # u, v =u[out_of_bound_filter], v[out_of_bound_filter]

    point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    cloud = cloud[out_of_bound_filter]


    horDis = point_pixel_idx[:, 2]
    point_pixel_idx = point_pixel_idx.astype(int)

    all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
    obj_cloud_world_list = []
    colors = []
    semantic_labels = []
    for i in range(len(labels)):
        obj_colors = []
        obj_mask = masks[i]
        cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
        all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
        obj_cloud = cloud[cloud_mask]
        obj_pixel = point_pixel_idx[cloud_mask]

        uv = obj_pixel[:, :2]
        depth = obj_pixel[:, 2]
        key = uv[:, 1] * image_shape[1] + uv[:, 0]

        sort_idx = np.argsort(depth)  # depth 작은 순 정렬
        key_sorted = key[sort_idx]
        _, unique_indices = np.unique(key_sorted, return_index=True)
        best_indices = sort_idx[unique_indices]

        obj_cloud = obj_cloud[best_indices]
        obj_pixel = obj_pixel[best_indices]

        obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
        obj_cloud_world_list.append(obj_cloud_world)
        for j in range(obj_pixel.shape[0]):
            px, py = obj_pixel[j, 0], obj_pixel[j, 1]
            obj_colors.append(image_src[py, px])
        if len(obj_colors) > 0:
            avg_color = np.mean(obj_colors, axis=0)
        else:
            avg_color = [0, 0, 0]
        colors.append(avg_color)

        semantic_labels.append(labels[i])

    if image_src is not None:
        all_obj_point_pixel_idx = point_pixel_idx

        maxRange = 6.0
        pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
        image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = (
            np.array([pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)]).T
        )  # assume RGB

    return obj_cloud_world_list, colors, semantic_labels

def extend_along_global_z_for_wall(pts_xyz: np.ndarray,
                                   z_max: float = 2.0,
                                   dz: float = 0.05) -> np.ndarray:
    """
    pts_xyz: (N, 3) 글로벌 좌표계의 포인트들 (x,y,z)
    z_min~z_max 구간을 dz 간격으로 수직 복제해 (x,y 동일, z만 변화) 반환
    """
    if pts_xyz.size == 0:
        return pts_xyz

    z0, z1 = pts_xyz[:, 2].min(), pts_xyz[:, 2].max()
    if z1 >= z_max - 1e-3:
        return pts_xyz
    # print(f"extend z {z0:.2f}~{z1:.2f} to {z_min:.2f}~{z_max:.2f}")

    z_grid = np.arange(z0, z_max + 1e-9, dz, dtype=np.float32)  # 포함되도록 +eps

    # 브로드캐스팅으로 (N * K, 3) 생성
    N = pts_xyz.shape[0]
    K = z_grid.shape[0]
    x = pts_xyz[:, 0:1]                      # (N,1)
    y = pts_xyz[:, 1:2]                      # (N,1)

    X = np.repeat(x, K, axis=1).reshape(-1, 1)       # (N*K,1)
    Y = np.repeat(y, K, axis=1).reshape(-1, 1)       # (N*K,1)
    Z = np.tile(z_grid, (N, 1)).reshape(-1, 1)       # (N*K,1)

    pts_out = np.hstack([X, Y, Z]).astype(np.float32)  # (N*K,3)
    return pts_out


def generate_seg_comp_cloud(
    cloud: np.ndarray,
    masks,
    labels,
    R_b2w,
    t_b2w,
    image_src=None,
    viz_cloud=False,
    platform="wheelchair",
    label_ids=["wall"]
):
    
    if masks is None:
        return None, None

    image_shape = masks[0].shape

    if platform == "wheelchair":
        point_pixel_idx = scan2pixels_wheelchair(cloud)
    elif platform == "mcanum":
        point_pixel_idx = scan2pixels_mcanum(cloud)
    else:
        raise ValueError

    out_of_bound_filter = (
        (point_pixel_idx[:, 0] >= 0)
        & (point_pixel_idx[:, 0] < image_shape[1])
        & (point_pixel_idx[:, 1] >= 0)
        & (point_pixel_idx[:, 1] < image_shape[0])
    )
    point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    point_pixel_idx = point_pixel_idx.astype(int)

    cloud = cloud[out_of_bound_filter]
    all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)

    merged_cloud = cloud.copy()
     # 먼저 wall 확장
    for i in range(len(labels)):
        if label_ids[labels[i]][0] != 'wall':
            continue
        obj_mask = masks[i]
        cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
        all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
        obj_cloud = cloud[cloud_mask]
        obj_pixel = point_pixel_idx[cloud_mask]

        uv = obj_pixel[:, :2]
        depth = obj_pixel[:, 2]
        key = uv[:, 1] * image_shape[1] + uv[:, 0]

        sort_idx = np.argsort(depth)  # depth 작은 순 정렬
        key_sorted = key[sort_idx]
        _, unique_indices = np.unique(key_sorted, return_index=True)
        best_indices = sort_idx[unique_indices]

        obj_cloud = obj_cloud[best_indices]
        obj_pixel = obj_pixel[best_indices]

        if label_ids[labels[i]][0] == 'wall':
            obj_cloud = np.array(obj_cloud)
            wall_cloud_extended = extend_along_global_z_for_wall(obj_cloud, z_max=3.0, dz=0.05)
            if len(wall_cloud_extended)>0:
                # print(f"wall extended from {merged_cloud.shape[0]}")
                merged_cloud = np.concatenate((merged_cloud, wall_cloud_extended), axis=0)
                # print(f"to {merged_cloud.shape[0]}")

    # Project the cloud points to image pixels
    if platform == "wheelchair":
        point_pixel_idx = scan2pixels_wheelchair(merged_cloud)
    elif platform == "mcanum":
        point_pixel_idx = scan2pixels_mcanum(merged_cloud)
    else:
        raise ValueError


    out_of_bound_filter = (
        (point_pixel_idx[:, 0] >= 0)
        & (point_pixel_idx[:, 0] < image_shape[1])
        & (point_pixel_idx[:, 1] >= 0)
        & (point_pixel_idx[:, 1] < image_shape[0])
    )
    point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    cloud = merged_cloud[out_of_bound_filter]

    horDis = point_pixel_idx[:, 2]
    point_pixel_idx = point_pixel_idx.astype(int)

    all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
    obj_cloud_world_list = []
    colors = []
    semantic_labels = []
    for i in range(len(labels)):
        # if label_ids[labels[i]][0] == 'wall':
        #     obj_cloud_world_list.append([])
        #     colors.append([0, 0, 0])
        #     semantic_labels.append(labels[i])
        #     continue

        obj_colors = []
        obj_mask = masks[i]
        cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
        all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
        obj_cloud = cloud[cloud_mask]
        obj_pixel = point_pixel_idx[cloud_mask]

        uv = obj_pixel[:, :2]
        depth = obj_pixel[:, 2]
        key = uv[:, 1] * image_shape[1] + uv[:, 0]

        sort_idx = np.argsort(depth)  # depth 작은 순 정렬
        key_sorted = key[sort_idx]
        _, unique_indices = np.unique(key_sorted, return_index=True)
        best_indices = sort_idx[unique_indices]

        obj_cloud = obj_cloud[best_indices]
        obj_pixel = obj_pixel[best_indices]

        obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
        
        obj_cloud_world_list.append(obj_cloud_world)
        for j in range(obj_pixel.shape[0]):
            px, py = obj_pixel[j, 0], obj_pixel[j, 1]
            obj_colors.append(image_src[py, px])
        if len(obj_colors) > 0:
            avg_color = np.mean(obj_colors, axis=0)
        else:
            avg_color = [0, 0, 0]
        colors.append(avg_color)

        semantic_labels.append(labels[i])

    if image_src is not None:
        all_obj_point_pixel_idx = point_pixel_idx

        maxRange = 6.0
        pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
        image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = (
            np.array([pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)]).T
        )  # assume RGB

        # cv2.imshow("Reproj", image_src)
        # cv2.waitKey(1)

    return obj_cloud_world_list, colors, semantic_labels


def generate_color_cloud(
    cloud_body: np.ndarray,
    R_b2w,
    t_b2w,
    image_src=None,
    platform="wheelchair",
):
    # Project the cloud points to image pixels
    if platform == "wheelchair":
        point_pixel_idx = scan2pixels_wheelchair(cloud_body)
    elif platform == "mcanum":
        point_pixel_idx = scan2pixels_mcanum(cloud_body)
    else:
        raise ValueError

    image_shape = image_src.shape

    out_of_bound_filter = (
        (point_pixel_idx[:, 0] >= 0)
        & (point_pixel_idx[:, 0] < image_shape[1])
        & (point_pixel_idx[:, 1] >= 0)
        & (point_pixel_idx[:, 1] < image_shape[0])
    )

    point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    cloud_body = cloud_body[out_of_bound_filter]
    cloud_world = cloud_body[:, :3] @ R_b2w.T + t_b2w

    horDis = point_pixel_idx[:, 2]
    point_pixel_idx = point_pixel_idx.astype(int)

    color = image_src[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(np.uint8)

    maxRange = 6.0
    pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
    image_src[point_pixel_idx[:, 1], point_pixel_idx[:, 0]] = np.array(
        [pixelVal, 255 - pixelVal, np.zeros_like(pixelVal)]
    ).T

    colored_cloud = np.hstack([cloud_world, color])

    return cloud_world, color