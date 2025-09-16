import math
import heapq


class MultiGoalPlanner:
    def __init__(self, grid_map, resolution):
        """
        Initializes the A* planner.
        Input:
            grid_map (np.ndarray): The 2D occupancy grid.
            resolution (float): The resolution of the map in meters/pixel.
        Output:
            None
        Parameter description:
            self.grid_map: The occupancy grid map.
            self.resolution: The map resolution.
            self.height, self.width: Dimensions of the grid map.
            self.neighbors: 8-connectivity neighborhood for searching.
        """
        self.grid_map = grid_map
        self.resolution = resolution
        self.height, self.width = grid_map.shape
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def shortest_to_many(self, start_px, start_py, goal_pixels):
        """goal_pixels: set of (x,y). return: dict[(x,y)] = dist"""
        open_set = []
        start = (start_px, start_py)
        heapq.heappush(open_set, (0.0, start))
        g_score = {start: 0.0}
        visited = set()
        remaining = set(goal_pixels)
        dists = {}

        while open_set and remaining:
            cur_g, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)

            if cur in remaining:
                dists[cur] = cur_g
                remaining.remove(cur)
                if not remaining:  # If all destinations are decided => end
                    break

            cx, cy = cur
            for dx, dy in self.neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.grid_map[ny, nx] == 100:
                    continue
                ng = cur_g + math.hypot(dx, dy)
                if ng < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = ng
                    heapq.heappush(open_set, (ng, (nx, ny)))

        # Failed to reach the destination
        for g in goal_pixels:
            if g not in dists:
                dists[g] = float('inf')
        return dists
