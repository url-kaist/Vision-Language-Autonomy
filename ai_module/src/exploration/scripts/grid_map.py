from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import rospy
from collections import deque
from scipy import ndimage
import numpy as np


class BGProcessor:
    def __init__(self, voxel_size=0.1, grid_size=0.1, z_threshold=0.1, map_width=200, map_height=200, map_origin=(-12.5, -5.0)):
        self.voxel_size = voxel_size
        self.grid_size = grid_size

        self.floor_cloud = np.empty((0, 3), dtype=np.float32)
        self.wall_cloud = np.empty((0, 3), dtype=np.float32)

        self.wall_grids = set()
        self.wall_grid_dict = defaultdict(set)

        self.detected_planes = []
        
        self.occupancy_grid_size = 0.1
        self.z_threshold = z_threshold
        self.map_width = map_width
        self.map_height = map_height
        self.map_origin = map_origin  # (x, y)

        self.occupancy_grid_pub = rospy.Publisher("/occupancy_grid_map", OccupancyGrid, queue_size=1)
        self.occupancy_data = [0] * (self.map_width * self.map_height)
        self.occupancy_grid_msg = self.init_occupancy_grid()

    def init_occupancy_grid(self):
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.info.resolution = self.occupancy_grid_size
        grid.info.width = self.map_width
        grid.info.height = self.map_height

        origin = Pose()
        origin.position.x = self.map_origin[0]
        origin.position.y = self.map_origin[1]
        origin.position.z = 0.0
        grid.info.origin = origin

        return grid

    def update_occupancy_grid(self, cloud: np.ndarray):
        origin_x, origin_y = self.map_origin

        occupied_grids = []
        for pt in cloud:
            if pt[2] < self.z_threshold or pt[2] > 1.5:
                continue
            gx = int((pt[0] - origin_x) / self.occupancy_grid_size)
            gy = int((pt[1] - origin_y) / self.occupancy_grid_size)
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                idx = gy * self.map_width + gx
                if self.occupancy_data[idx] != 100:
                    self.occupancy_data[idx] = 100
                    occupied_grids.append((gx, gy)) 

        map_array = np.array(self.occupancy_data).reshape(self.map_height, self.map_width)
        filled_map = map_array.copy()
        filled_map[0, :] = 0 
        filled_map[-1, :] = 0
        filled_map[:, 0] = 0
        filled_map[:, -1] = 0

        filled_map = ndimage.binary_fill_holes(map_array == 100).astype(int)
        
        labeled_array, num_features = ndimage.label(filled_map)
        for region_num in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_num)
            if region_size > 10000: #assume object smaller than 10mx10m
                filled_map[labeled_array == region_num] = 0
                
        filled_map[filled_map > 0] = 100
        self.occupancy_data = filled_map.flatten()
        
        self.occupancy_grid_msg.header.stamp = rospy.Time.now()
        self.occupancy_grid_msg.data = self.occupancy_data
        self.occupancy_grid_pub.publish(self.occupancy_grid_msg)



def main():
    rospy.init_node("bg_processor", anonymous=True)
    bg_processor = BGProcessor()
    rospy.loginfo("BG Processor Node Initialized")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass