import cv2
from omegaconf import DictConfig
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

import os

def map_grid_to_point_cloud(occupancy_grid_map, resolution, point_cloud):
    """
    Map the occupancy grid back to the original coordinates in the point cloud.

    Parameters:
        occupancy_grid_map (numpy.array): Occupancy grid map as a 2D numpy array, where each cell is marked as either 0 (unoccupied) or 1 (occupied).
        grid_size (tuple): A tuple (width, height) representing the size of the occupancy grid map in meters.
        resolution (float): The resolution of each cell in the grid map in meters.
        point_cloud (numpy.array): 2D numpy array of shape (N, 2), where N is the number of points and each row represents a point (x, y).

    Returns:
        numpy.array: A subset of the original point cloud containing points that correspond to occupied cells in the occupancy grid.
    """

    # make sure image is binary
    occupancy_grid_map = (occupancy_grid_map > 0).astype(np.uint8)

    # Get the occupied cell indices
    y_cells, x_cells = np.where(occupancy_grid_map == 1)

    # Compute the corresponding point coordinates for occupied cells
    # NOTE: The coordinates are shifted by 10.5 cells to account for the padding added to the grid map
    mapped_x_coords = (x_cells - 10.5) * resolution + np.min(point_cloud[:, 0])
    mapped_y_coords = (y_cells - 10.5) * resolution + np.min(point_cloud[:, 1])

    # Stack the mapped x and y coordinates to form the mapped point cloud
    mapped_point_cloud = np.column_stack((mapped_x_coords, mapped_y_coords))

    return mapped_point_cloud

def distance_transform(occupancy_map, reselotion, tmp_path):
    """
        Perform distance transform on the occupancy map to find the distance of each cell to the nearest occupied cell.
        :param occupancy_map: 2D numpy array representing the occupancy map.
        :param reselotion: The resolution of each cell in the grid map in meters.
        :param path: The path to save the distance transform image.
        :return: The distance transform of the occupancy map.
    """

    print("occupancy_map shape: ", occupancy_map.shape)
    bw = occupancy_map.copy()
    full_map = occupancy_map.copy()

    # invert the image
    bw = np.where(occupancy_map > 0, 255, 0).astype(np.uint8)
    bw = cv2.bitwise_not(bw)

    # Perform the distance transform algorithm
    bw = np.uint8(bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    print("range of dist: ", np.min(dist), np.max(dist))
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist.png"))

    dist = np.uint8(dist)
    # apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(dist, (11, 1), 10)
    plt.figure()
    plt.imshow(blur, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_blur.png"))
    _, dist = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_thresh.png"))

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype("uint8")
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("number of seeds, aka rooms: ", len(contours))

    # print the area of each seed
    for i in range(len(contours)):
        print("area of seed {}: ".format(i), cv2.contourArea(contours[i]))

    # remove small seed contours
    min_area_m = 0.5
    min_area = (min_area_m / reselotion) ** 2
    print("min_area: ", min_area)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    print("number of contours after remove small seeds: ", len(contours))

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    circle_radius = 1  # in pixels
    cv2.circle(markers, (3, 3), circle_radius, len(contours) + 1, -1)

    # Perform the watershed algorithm
    full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
    cv2.watershed(full_map, markers)

    plt.figure()
    plt.imshow(markers, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "markers.png"))

    # find the vertices of each room
    room_vertices = []
    for i in range(len(contours)):
        room_vertices.append(np.where(markers == i + 1))
    room_vertices = np.array(room_vertices, dtype=object).squeeze()
    print("room_vertices shape: ", room_vertices.shape)

    return room_vertices

def segment_rooms(ply_path, output_path, floor_zero_level=0.01, floor_height=3.0):
    assert os.path.exists(ply_path), f"{ply_path} not found"

    # Floor 객체 대체
    floor_pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(floor_pcd.points)
    print("xyz shape: ", xyz.shape)
    xyz_full = xyz.copy()

    ## Slice below the ceiling ##
    # xyz = xyz[xyz[:, 2] < floor_zero_level + floor_height - 0.3]
    # xyz = xyz[xyz[:, 2] >= floor_zero_level] # + 1.5]
    # xyz_full = xyz_full[xyz_full[:, 2] < floor_zero_level + floor_height - 0.2]

    # project the point cloud to 2d
    pcd_2d = xyz[:, [0, 1]]
    xyz_full = xyz_full[:, [0, 1]]

    # define the grid size and resolution based on the 2d point cloud
    grid_size = (
        int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
        int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
    )
    grid_size = (grid_size[0] + 1, grid_size[1] + 1)
    resolution = 0.1 #self.cfg.pipeline.grid_resolution
    print("grid_size: ", resolution)

    # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
    num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
    num_bins = (num_bins[1] + 1, num_bins[0] + 1)
    hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
    if True: #self.cfg.pipeline.save_intermediate_results:
        # plot the histogram
        plt.figure()
        plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, "2D_histogram.png"))

    # applythresholding
    hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.GaussianBlur(hist, (5, 5), 1)
    hist_threshold = 0.25 * np.max(hist)
    _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

    # create a bigger image to avoid losing the walls
    walls_skeleton = cv2.copyMakeBorder(
        walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    # apply closing to the walls skeleton
    kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    walls_skeleton = cv2.morphologyEx(
        walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
    )

    # extract outside boundary from histogram of xyz_full
    hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
    hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
    _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

    # create a bigger image to avoid losing the walls
    outside_boundary = cv2.copyMakeBorder(
        outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    # apply closing to the outside boundary
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    outside_boundary = cv2.morphologyEx(
        outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
    )

    # extract the outside contour from the outside boundary
    contours, _ = cv2.findContours(
        outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outside_boundary = np.zeros_like(outside_boundary)
    cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
    outside_boundary = outside_boundary.astype(np.uint8)

    if True: #self.cfg.pipeline.save_intermediate_results:
        plt.figure()
        plt.imshow(walls_skeleton, cmap="gray", origin="lower")
        plt.savefig(os.path.join(output_path, "walls_skeleton.png"))

        plt.figure()
        plt.imshow(outside_boundary, cmap="gray", origin="lower")
        plt.savefig(os.path.join(output_path, "outside_boundary.png"))

    # combine the walls skelton and outside boundary
    full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))
    # inside_mask = cv2.bitwise_not(outside_boundary)
    # full_map = cv2.bitwise_and(walls_skeleton, inside_mask)

    # apply closing to the full map
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)
    full_map = cv2.morphologyEx(walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=2)

    if True: #self.cfg.pipeline.save_intermediate_results:
        # plot the full map
        plt.figure()
        plt.imshow(full_map, cmap="gray", origin="lower")
        plt.savefig(os.path.join(output_path, "full_map.png"))
    # apply distance transform to the full map
    room_vertices = distance_transform(full_map, resolution, output_path)

    # using the 2D room vertices, map the room back to the original point cloud using KDTree
    # room_pcds = []
    # room_masks = []
    # room_2d_points = []
    # floor_tree = cKDTree(np.array(floor_pcd.points))
    # for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
    #     room = np.zeros_like(full_map)
    #     room[room_vertices[i][0], room_vertices[i][1]] = 255
    #     room_masks.append(room)
    #     room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
    #     room_2d_points.append(room_m)
    #     # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
    #     z_levels = np.arange(
    #         floor_zero_level, floor_zero_level + floor_height, 0.05
    #     )
    #     z_levels = z_levels.reshape(-1, 1)
    #     z_levels *= -1
    #     room_m3dd = []
    #     for z in z_levels:
    #         room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
    #         room_m3dd.append(room_m3d)
    #     room_m3d = np.concatenate(room_m3dd, axis=0)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(room_m3d)
    #     # rotate floor pcd to align with the original point cloud
    #     T1 = np.eye(4)
    #     T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    #     pcd.transform(T1)
    #     # find the nearest point in the original point cloud
    #     _, idx = floor_tree.query(np.array(pcd.points), workers=-1)
    #     pcd = floor_pcd.select_by_index(idx)
    #     room_pcds.append(pcd)
        
    # self.room_masks[floor.floor_id] = room_masks

    # compute the features of room: input a list of poses and images, output a list of embeddings list
    # rgb_list = []
    # pose_list = []
    # F_g_list = []

    # all_global_clip_feats = dict()
    # for i, img_id in tqdm(enumerate(range(0, len(self.dataset), self.cfg.pipeline.skip_frames)), desc="Computing room features"):
    #     rgb_image, _, pose, _, _ = self.dataset[img_id]
    #     F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
    #     all_global_clip_feats[str(img_id)] = F_g
    #     rgb_list.append(rgb_image)
    #     pose_list.append(pose)
    #     F_g_list.append(F_g)
    # np.savez(
    #     os.path.join(self.graph_tmp_folder, "room_views.npz"),
    #     **all_global_clip_feats,
    # )

    # pcd_min = np.min(np.array(floor_pcd.points), axis=0)
    # pcd_max = np.max(np.array(floor_pcd.points), axis=0)
    # assert pcd_min.shape[0] == 3

    # repr_embs_list, repr_img_ids_list = compute_room_embeddings(
    #     room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 10, tmp_floor_path
    # )
    # assert len(repr_embs_list) == len(room_2d_points)
    # assert len(repr_img_ids_list) == len(room_2d_points)

    # room_index = 0
    # for i in range(len(room_2d_points)):
    #     room = Room(
    #         str(floor.floor_id) + "_" + str(room_index),
    #         floor.floor_id,
    #         name="room_" + str(room_index),
    #     )
    #     room.pcd = room_pcds[i]
    #     room.vertices = room_2d_points[i]
    #     self.floors[int(floor.floor_id)].add_room(room)
    #     room.room_height = floor_height
    #     room.room_zero_level = floor.floor_zero_level
    #     room.embeddings = repr_embs_list[i]
    #     room.represent_images = [int(k * self.cfg.pipeline.skip_frames) for k in repr_img_ids_list[i]]
    #     self.rooms.append(room)
    #     room_index += 1
    # print(
    #     "number of rooms in floor {} is {}".format(
    #         floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
    #     )
    # )

#main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Segment rooms from a point cloud")
    # parser.add_argument("--ply_path", type=str, required=True, help="Path to the input PLY file")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output images")
    parser.add_argument("--floor_zero_level", type=float, default=0.01, help="Floor zero level")
    parser.add_argument("--floor_height", type=float, default=3.0, help="Floor height")
    args = parser.parse_args()

    # segment_rooms(args.ply_path, args.output_path, args.floor_zero_level, args.floor_height)
    
    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    occupancy_map = np.where(img < 128, 1, 0).astype(np.uint8)
    # for i in range(occupancy_map.shape[0]):
        # print(occupancy_map[i, :])
    room_vertices = distance_transform(occupancy_map, 0.1, args.output_path)