import numpy as np

class CustomOccupancyGrid:
    def __init__(self, occupancy_grid):
        self.data = occupancy_grid.data
        self.info = occupancy_grid.info
        self.grid_data = np.array(self.data, dtype=np.int8).reshape((self.info.height, self.info.width))

    def grid_to_world(self, grid_x, grid_y):
        return self.info.origin.position.x + grid_x * self.info.resolution, self.info.origin.position.y + grid_y * self.info.resolution
    
    
    def world_to_grid(self, world_x, world_y):
        return int((world_x - self.info.origin.position.x) / self.info.resolution), int((world_y - self.info.origin.position.y) / self.info.resolution)
    
    def crop(self, center_x, center_y, crop_width, crop_height):
        """
        Crop the occupancy grid around a center point
        
        Args:
            center_x: Center x coordinate in grid coordinates
            center_y: Center y coordinate in grid coordinates  
            crop_width: Width of the crop area
            crop_height: Height of the crop area
            
        Returns:
            CustomOccupancyGrid: New cropped occupancy grid instance
        """
        # Calculate crop boundaries
        half_width = crop_width // 2
        half_height = crop_height // 2
        
        x_min = max(0, int(center_x) - half_width)
        x_max = min(self.info.width, int(center_x) + half_width)
        y_min = max(0, int(center_y) - half_height)
        y_max = min(self.info.height, int(center_y) + half_height)
        
        # Crop the grid data
        cropped_grid_data = self.grid_data[y_min:y_max, x_min:x_max]
        
        # Create a mock occupancy grid message for the cropped area
        class MockOccupancyGrid:
            def __init__(self, data, info):
                self.data = data
                self.info = info
        
        # Create new info with updated origin and dimensions
        import copy
        new_info = copy.deepcopy(self.info)
        new_info.width = x_max - x_min
        new_info.height = y_max - y_min
        new_info.origin.position.x = self.info.origin.position.x + x_min * self.info.resolution
        new_info.origin.position.y = self.info.origin.position.y + y_min * self.info.resolution
        
        # Create mock occupancy grid
        mock_occupancy_grid = MockOccupancyGrid(cropped_grid_data.flatten(), new_info)
        
        # Return new CustomOccupancyGrid instance
        return CustomOccupancyGrid(mock_occupancy_grid)
    
    def __str__(self):
        return f"OccupancyGrid(width={self.info.width}, height={self.info.height}, resolution={self.info.resolution}, origin={self.info.origin.position.x}, {self.info.origin.position.y})"
    
    def __repr__(self):
        return self.__str__()

    def save_npz(self, path):
        import numpy as np
        np.savez(
            path,
            grid=self.grid_data.astype(np.int8),
            width=self.info.width,
            height=self.info.height,
            resolution=self.info.resolution,
            origin_x=self.info.origin.position.x,
            origin_y=self.info.origin.position.y
        )

    @staticmethod
    def load_npz(path):
        import numpy as np, types
        z = np.load(path, allow_pickle=False)
        grid = z["grid"]
        # msg 모양 흉내(필요 필드만)
        info = types.SimpleNamespace()
        info.width = int(z["width"])
        info.height = int(z["height"])
        info.resolution = float(z["resolution"])
        pos = types.SimpleNamespace(x=float(z["origin_x"]), y=float(z["origin_y"]))
        origin = types.SimpleNamespace(position=pos)
        info.origin = origin
        mock = types.SimpleNamespace(data=grid.flatten(), info=info)
        return CustomOccupancyGrid(mock)
