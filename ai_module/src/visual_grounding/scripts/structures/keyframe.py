import cv2
import os
import time
import json
import logging
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple, List, Literal
from functools import reduce
import copy

from ai_module.src.visual_grounding.scripts.structures.data import Datas
from ai_module.src.visual_grounding.scripts.structures.place import Place
from ai_module.src.visual_grounding.scripts.structures.entity import Entities, Entity
from ai_module.src.visual_grounding.scripts.structures.utils import parse_id
from ai_module.src.visual_grounding.scripts.structures.bbox import BBox, BBoxes
from ai_module.src.visual_grounding.scripts.structures.occupancy_grid import CustomOccupancyGrid
from ai_module.src.utils.visualizer import Visualizer

class Keyframe:
    default_styles = {'default': {'show': True, 'color': 'green'}}

    def __init__(self, place: Place, entities: Union[Entities, Dict[str, Entities]]):
        self.is_real_world = False # TODO: BUG

        self.image_path = place.image_path

        if self.image_path and isinstance(self.image_path, str):
            self.directory, self.filename = os.path.split(self.image_path)
            self.ext = os.path.splitext(self.image_path)[1]
        else:
            self.directory = None
            self.filename = None
            self.ext = None
            logging.warning(f"Place has invalid image_path: {self.image_path}")
        
        self.image = None
        if self.image_path and isinstance(self.image_path, str):
            try:
                self.image = cv2.imread(self.image_path)
                if self.image is None:
                    logging.warning(f"Failed to read image (file may not exist): {self.image_path}")
            except Exception as e:
                logging.warning(f"Failed to read image: {self.image_path}, error: {e}")
        else:
            logging.warning(f"Invalid image_path: {self.image_path}")

        self.pose = place.pose
        self.entities = entities if isinstance(entities, Dict) else {'default': entities}
        self.related_entities = reduce(operator.add, self.entities.values())
        self.movable_points = None
        self.dist_idx2movable_points_indices = None
        
    def __repr__(self) -> str:
        try:
            repr = f"Keyframe(image_path: {self.image_path}, related_objects: {self.related_entities('object')}, related_detections: {self.related_entities('detection')})"
        except:
            repr = f"Keyframe(image_path: {self.image_path})"
        return repr

    @property
    def id(self) -> int:
        return int(self.filename.split('_')[-1].split('.')[0])

    @property
    def image_size(self) -> Tuple[int, int]:
        height, width, _ = self.image.shape
        return width, height

    def get_entities(self, ids, etype: Literal['object', 'detection', 'all'] = 'object') -> Entities:
        return self.related_entities(etype).get(ids)

    def object_union_area(self, ids: Union[List[int], None] = None):
        entities = {entity_id: entity for entity_id, entity in self.related_entities.items() if entity_id in ids}
        entities = Entities(entities)
        bboxes = entities.get_bboxes(pose=self.pose, image_size=self.image_size, is_real_world=self.is_real_world, kf_id=self.id)
        return bboxes.union_area

    def save_path(self, node_name="", suffix="") -> str:
        return os.path.join(
            self.directory, node_name,
            self.filename.replace(self.ext, f"{suffix}{self.ext}"))

    def apply_rewrap(self, sorted_objects: List[Dict], cut_x: float, image_width: int, vis_image: np.ndarray) -> Tuple[List[BBox], np.ndarray]:
        # Apply rewrap to image
        rewarped_image = np.roll(vis_image, -int(cut_x), axis=1)
        
        # Apply rewrap to objects and convert to BBox format
        rewarped_bboxes = []
        cut_x_int = int(cut_x)
        
        for obj in sorted_objects:
            u_min, v_min, u_max, v_max = obj['bbox']
            
            # Apply rewrap transformation
            if u_max < cut_x_int:
                bbox_data = np.array([u_min + image_width - cut_x_int, v_min, 
                                     u_max + image_width - cut_x_int, v_max])
            else:
                bbox_data = np.array([u_min - cut_x_int, v_min, 
                                     u_max - cut_x_int, v_max])
            
            # Convert to BBox format
            bbox_data = bbox_data.astype(int)
            bbox = BBox(bbox_data, object_id=obj['id'])
            rewarped_bboxes.append(bbox)
        
        return rewarped_bboxes, rewarped_image

    def annotate(self, styles=None, node_name="", suffix="", etype='object', given_image=None, save_image=True, **kwargs) -> Union[str, None]:
        """
        Annotate the keyframe image with bounding boxes.
        
        Args:
            styles: Dictionary with style configurations:
                   - For all objects in group: {'candidate': {'color': 'blue'}}
                   - For specific object IDs in group: {'candidate': {'ids': [1, 2, 3], 'color': 'red'}}
            node_name: Directory name for saving
            suffix: File suffix for saving
            
        Returns:
            Path to saved annotated image, or None if failed
        """
        if self.image is None:
            logging.warning(f"Cannot annotate keyframe: image is None (image_path: {self.image_path})")
            return None

        styles = styles if styles else self.default_styles

        vis_image = self.image.copy() if given_image is None else given_image.copy()
        
        # Collect all entities from all styles for unified rewrap
        all_entities_list = []
        style_entities_map = {}  # Map style key to its entities for later drawing
        
        for key, style in styles.items():
            # Get objects from the specified group
            entities_group = self.entities.get(key)
            if entities_group is None:
                logging.warning(f"{key} doesn't exist in keyframe, which will be not annotated: {self}")
                continue
            entities = entities_group(etype)
            if entities is None:
                logging.warning(f"{key} doesn't exist in keyframe, which will be not annotated: {self}")
                continue

            # Apply filtering if 'ids' is specified in style
            if 'ids' in style:
                filter_ids = style['ids']
                if not isinstance(filter_ids, list):
                    filter_ids = [filter_ids]
                entities = entities.get(filter_ids)
                if entities is None:
                    logging.warning(f"Filtered entities for {key} is None, which will be not annotated: {self}")
                    continue
                
            # rewrap functionality
            bboxes = entities.get_bboxes(
                pose=self.pose, image_size=self.image_size, is_real_world=self.is_real_world, kf_id=self.id)
            
            if len(bboxes) > 0:
                # Convert bboxes to object format for rewrap
                entities_list = []
                for bbox in bboxes:
                    entities_list.append({
                        'id': bbox.object_id,
                        'bbox': [bbox.u_min, bbox.v_min, bbox.u_max, bbox.v_max]
                    })
                
                # Store entities for this style
                style_entities_map[key] = {
                    'entities_list': entities_list,
                    'style': style,
                    'bboxes': bboxes
                }
                
                # Add to all entities list
                all_entities_list.extend(entities_list)
        
        # Apply unified rewrap if we have entities
        if len(all_entities_list) > 0:
            # Apply rewrap logic
            image_width = self.image.shape[1]
            sorted_entities = sorted(all_entities_list, key=lambda x: x['bbox'][0])
            num_entities = len(sorted_entities)
            
            if num_entities >= 1:
                # Find maximum gap for optimal cut point
                max_gap = float("-inf")
                max_gap_idx = 0
                for curr_i in range(num_entities):
                    next_i = (curr_i + 1) % num_entities
                    curr_u_max = sorted_entities[curr_i]['bbox'][2]
                    next_u_min = sorted_entities[next_i]['bbox'][0] + (image_width if next_i == 0 else 0.0)
                    gap = next_u_min - curr_u_max
                    if gap > max_gap:
                        max_gap = gap
                        max_gap_idx = curr_i

                max_gap_next_idx = (max_gap_idx + 1) % num_entities
                rightmost_u_max = sorted_entities[max_gap_next_idx]['bbox'][0]
                leftmost_u_min = sorted_entities[max_gap_idx]['bbox'][2]
                cut_x = leftmost_u_min + ((rightmost_u_max - leftmost_u_min + image_width) % image_width / 2.0)

                # Apply rewrap to image and objects (only once)
                rewarped_bboxes_all, vis_image = self.apply_rewrap(sorted_entities, cut_x, image_width, vis_image)
                
                # Create a mapping from object_id to rewarped bbox for quick lookup
                rewarped_bbox_map = {bbox.object_id: bbox for bbox in rewarped_bboxes_all}
                
                # Draw bounding boxes for each style
                for key, style_data in style_entities_map.items():
                    style = style_data['style']
                    entities_list = style_data['entities_list']
                    
                    # Get rewarped bboxes for this style
                    style_rewarped_bboxes = []
                    for entity in entities_list:
                        if entity['id'] in rewarped_bbox_map:
                            style_rewarped_bboxes.append(rewarped_bbox_map[entity['id']])
                    
                    if len(style_rewarped_bboxes) > 0:
                        # Draw rewarped bounding boxes using Visualizer
                        vis_image = Visualizer().draw_bboxes(
                            vis_image, BBoxes(style_rewarped_bboxes), color=style.get('color', 'green'),
                            alpha=style.get('alpha', 1.0), draw_id=style.get('draw_id', True))
        else:
            # No objects, use original drawing for each style
            for key, style_data in style_entities_map.items():
                style = style_data['style']
                bboxes = style_data['bboxes']
                vis_image = Visualizer().draw_bboxes(
                    vis_image, bboxes, color=style.get('color', 'green'),
                    alpha=style.get('alpha', 1.0), draw_id=style.get('draw_id', True))

        if save_image:
            save_path = self.save_path(node_name=node_name, suffix=suffix)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            success = cv2.imwrite(save_path, vis_image)
            return save_path if success else None
        return vis_image

    def get_movable_points(self, occupancy_grid, distances=[1.0, 1.7, 3.0], step_angle=40, style=None, node_name="", suffix="", path_history=None):
        if occupancy_grid is None:
            logging.warning(f"Occupancy grid is None")
            return np.array([])
        
        if self.pose is None:
            logging.warning(f"Keyframe {self.image_path} has no pose data")
            return np.array([])
        
        # Extract position and orientation from pose
        pose_matrix = np.array(self.pose)
        position = pose_matrix[:3, 3]  # Extract translation
        rotation_matrix = pose_matrix[:3, :3]  # Extract rotation matrix
        
        # Convert rotation matrix to euler angles
        # Extract yaw from rotation matrix
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Convert current position to grid coordinates
        current_x = position[0]
        current_y = position[1]
        current_gx, current_gy = occupancy_grid.world_to_grid(current_x, current_y)
        
        # List to store movable points and their distance indices
        movable_points = []
        movable_points_grid = []
        point_distance_indices = []  # Store distance index for each point
        
        # Generate points for each angle
        angles = np.arange(0, 360, step_angle)
        
        for angle_deg in angles:
            # Relative angle based on current orientation
            relative_angle = math.radians(angle_deg)
            absolute_angle = yaw + relative_angle
            
            # Generate points at specified distances
            for dist_idx, dist in enumerate(distances):
                # Calculate point in world coordinates
                target_x = current_x + dist * math.cos(absolute_angle)
                target_y = current_y + dist * math.sin(absolute_angle)
                
                # Convert to grid coordinates
                target_gx, target_gy = occupancy_grid.world_to_grid(target_x, target_y)
                
                # Check if within grid bounds
                if 0 <= target_gx < occupancy_grid.info.width and 0 <= target_gy < occupancy_grid.info.height:
                    # Ray casting from current position to target point
                    if self._is_path_clear(current_gx, current_gy, target_gx, target_gy,
                                       occupancy_grid.grid_data, occupancy_grid.info.width, occupancy_grid.info.height):
                        movable_points.append([target_x, target_y])
                        movable_points_grid.append([target_gx, target_gy])
                        point_distance_indices.append(dist_idx)

        # Convert to numpy array
        movable_points_array = np.array(movable_points)
        self.movable_points = movable_points_array
        self.dist_idx2movable_points_indices = {}
        for idx in range(len(distances)):
            self.dist_idx2movable_points_indices[idx] = []
        for i, idx in enumerate(point_distance_indices):
            self.dist_idx2movable_points_indices[idx].append(i)
            
        if len(movable_points_grid) == 0:
            logging.info(f"No movable points to visualize on occupancy grid")
            return None
        
        # Apply visualization if style is provided
        if style is not None and 'image' in style and style['image'].get('show', True):
            self._visualize_movable_points_on_image(movable_points_array, style['image'], node_name, suffix + "_image", point_distance_indices)
        
        if style is not None and 'occupancy_grid' in style and style['occupancy_grid'].get('show', True):
            self._visualize_movable_points_on_occupancy_grid(occupancy_grid, movable_points_grid, [current_gx, current_gy], 
                                                             style['occupancy_grid'], node_name, suffix + "_occupancy_grid", 
                                                             point_distance_indices, path_history)
        

        
    def _visualize_movable_points_on_image(self, movable_points, style=None, node_name="", suffix="", point_distance_indices=None) -> Union[str, None]:
        if self.image is None:
            logging.warning(f"Cannot visualize movable points: image is None (image_path: {self.image_path})")
            return None

        if movable_points is None or len(movable_points) == 0:
            logging.info(f"No movable points to visualize for keyframe")
            return None

        # Default styles for movable points
        default_style = {
            'show': True,
            'circle': {
                'color': 'green',
                'radius': 5,
                'thickness': -1
            },
            'text': {
                'color': 'white',
                'size': 0.5,
                'thickness': 1,
                'offset': (7, -7),  # (x_offset, y_offset) from circle center
                'distance_colors': {
                    'enabled': False,
                    'colors': ['green', 'red', 'blue'],  # Default colors for different distances
                }
            },
            'object_annotation': {
                'candidate': {
                    'show': True,
                    'color': 'green'
                },
                'etype': 'detection'
            }
        }
            
        style = style if style else default_style

        vis_image = self.image.copy()

        if not style.get('show', True):
            return

        if len(movable_points) == 0:
            return

        try:
            # Apply object_annotation if specified in style
            if 'object_annotation' in style:
                etype = style['object_annotation'].get('etype', 'object')
                vis_image = self.annotate(style['object_annotation'], node_name, suffix, etype=etype, given_image=vis_image, save_image=False)

            # Convert points to 3D (z=0)
            points_3d = np.array([[pt[0], pt[1], 0.0] for pt in movable_points])

            # Get pose matrix
            if self.pose is not None:
                pose_matrix = np.array(self.pose)
            else:
                logging.warning(f"Keyframe has no pose data")
                return

            # Project 3D points to image coordinates
            image_size = (vis_image.shape[1], vis_image.shape[0])  # (width, height)
            pixel_coords = Entity.project_3d_point_to_image(
                points_3d, pose_matrix, image_size=image_size, is_real_world=self.is_real_world
            )

            # Use Visualizer's color mapping for consistency
            color_map = Visualizer._COLOR_MAP.copy()

            # Draw points on image
            for i, (u, v, depth) in enumerate(pixel_coords):
                if 0 <= u < image_size[0] and 0 <= v < image_size[1]:
                    center_x, center_y = int(u), int(v)
                    
                    # Draw circle if enabled
                    if 'circle' in style:
                        circle_style = style['circle']
                        circle_color = color_map.get(circle_style.get('color', 'green'), (0, 255, 0))
                        circle_radius = circle_style.get('radius', 5)
                        circle_thickness = circle_style.get('thickness', -1)
                        cv2.circle(vis_image, (center_x, center_y), circle_radius, circle_color, circle_thickness)

                    # Draw text if enabled
                    if 'text' in style:
                        text_style = style['text']
                        # Determine text color based on distance index if enabled
                        text_color_name = text_style.get('color', 'white')
                        
                        # Check if distance-based colors are enabled
                        if (point_distance_indices is not None and 
                            'distance_colors' in text_style and 
                            text_style['distance_colors'].get('enabled', False)):
                            
                            distance_colors_config = text_style['distance_colors']
                            colors = distance_colors_config.get('colors', ['green', 'red', 'blue'])
                            
                            # Use distance index to get color
                            if i < len(point_distance_indices):
                                dist_idx = point_distance_indices[i]
                                if dist_idx < len(colors):
                                    text_color_name = colors[dist_idx]
                        
                        text_color = color_map.get(text_color_name, (255, 255, 255))
                        text_size = text_style.get('size', 0.5)
                        text_thickness = text_style.get('thickness', 1)
                        
                        # Calculate text size to center it properly
                        text_str = str(i)
                        (text_width, text_height), baseline = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
                        
                        # Determine text position
                        if 'circle' in style and style['circle'].get('show', True):
                            # If circle is shown, place text at offset from circle center
                            offset = style['text'].get('offset', (7, -7))
                            text_x = center_x + offset[0] - text_width // 2
                            text_y = center_y + offset[1] + text_height // 2
                        else:
                            # If no circle, place text at the point center
                            text_x = center_x - text_width // 2
                            text_y = center_y + text_height // 2
                        
                        cv2.putText(vis_image, text_str, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

        except Exception as e:
            logging.error(f"Error visualizing movable points: {e}")
            return

        # Save the visualized image
        save_path = self.save_path(node_name=node_name, suffix=suffix)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        success = cv2.imwrite(save_path, vis_image)
        return save_path if success else None

    def _visualize_movable_points_on_occupancy_grid(self, occupancy_grid, movable_points_grid, current_point, 
                                                    style, node_name="", suffix="", point_distance_indices=None, path_history=None):
        if occupancy_grid is None:
            logging.warning(f"Cannot visualize movable points: occupancy grid is None")
            return None

        # Default styles for occupancy grid visualization
        default_style = {
            'show': True,
            'grid': {
                'occupied_color': 'black', 
                'free_color': 'white', 
                'unknown_color': 'gray',
                'upsample_scale': 10,
                'width': None,
                'height': None
            },
            'movable_points': {
                'color': 'green',          
                'radius': 3,
                'thickness': -1
            },
            'current_point': {
                'color': 'red',            
                'radius': 5,
                'thickness': -1
            },
            'text': {
                'color': 'green', 
                'size': 0.4,
                'thickness': 1,
                'offset': (7, -7),  # (x_offset, y_offset) from movable point
                'distance_colors': {
                    'enabled': False,
                    'colors': ['green', 'red', 'blue']  # Default colors for different distances
                }
            },
            'path_history': {
                'color': 'yellow',
                'alpha': 0.3,
                'alpha_decay': 0.1,  # How much alpha decreases per step (older points get smaller alpha)
            },
            'object': {
                'color': 'blue',
                'thickness': -1,
                'alpha': 0.3,
                'group_colors': {
                    'candidate': 'orange',
                    'reference': 'orange'
                }
            }
        }
        
        style = style if style else default_style

        if not style.get('show', True):
            return None

        try:
            # Create visualization image
            height, width = occupancy_grid.info.height, occupancy_grid.info.width
            
            grid_style = style['grid'] if 'grid' in style else default_style['grid']
                                    
            # Crop image around current_point if width and height are specified (do this first for efficiency)
            crop_offset_x = 0
            crop_offset_y = 0
            occupancy_grid_cropped = copy.deepcopy(occupancy_grid)
            cropped_width = width
            cropped_height = height
            
            if current_point is not None:
                crop_width = grid_style.get('width')
                crop_height = grid_style.get('height')
                
                if crop_width is not None and crop_height is not None:
                    gx, gy = current_point
                    
                    # Use CustomOccupancyGrid crop method
                    occupancy_grid_cropped = occupancy_grid.crop(gx, gy, crop_width, crop_height)
                    
                    # Store crop offset for coordinate adjustment
                    crop_offset_x = max(0, int(gx) - crop_width // 2)
                    crop_offset_y = max(0, int(gy) - crop_height // 2)
                    
                    cropped_width = occupancy_grid_cropped.info.width
                    cropped_height = occupancy_grid_cropped.info.height
            
            # Color mapping for occupancy grid
            occupied_color = grid_style.get('occupied_color', 'black')
            free_color = grid_style.get('free_color', 'white')
            unknown_color = grid_style.get('unknown_color', 'gray')
            
            # Store both coordinates and their alpha values based on index
            path_history_grid_coords = {}
            
            if 'path_history' in style and path_history is not None and len(path_history) > 0:
                history_style = style['path_history']
                base_alpha = history_style.get('alpha', 0.3)
                alpha_decay = history_style.get('alpha_decay', 0.1)  # How much alpha decreases per step
                
                total_points = len(path_history)
                for index, (world_x, world_y) in enumerate(path_history):
                    # Convert world coordinates to grid coordinates
                    gx, gy = occupancy_grid_cropped.world_to_grid(world_x, world_y)
                    
                    # Check if within grid bounds
                    if 0 <= gx < cropped_width and 0 <= gy < cropped_height:
                        point_alpha = max(0.05, base_alpha - ((total_points - 1 - index) * alpha_decay))
                        # Store with flipped Y coordinate for image coordinate system
                        path_history_grid_coords[(gx, gy)] = point_alpha
            
            # Color the cropped grid based on occupancy values
            vis_image = np.zeros((cropped_height, cropped_width, 3), dtype=np.uint8)
            for y in range(cropped_height):
                for x in range(cropped_width):
                    cell_value = occupancy_grid_cropped.grid_data[y, x]
                    path_history_alpha = path_history_grid_coords.get((x, y))
                    
                    # Flip Y coordinate to match image coordinate system (top-left origin)
                    image_y = cropped_height - 1 - y
                    
                    if cell_value == 100:  # Occupied
                        vis_image[image_y, x] = Visualizer._parse_color(occupied_color)
                    elif cell_value == 0:  # Free
                        vis_image[image_y, x] = Visualizer._parse_color(free_color)
                        if path_history_alpha is not None:
                            history_style = style['path_history']
                            history_color = history_style.get('color', 'yellow')
                            blended_color = Visualizer.blend_color(history_color, free_color, path_history_alpha)
                            vis_image[image_y, x] = blended_color
                    else:  # Unknown
                        vis_image[image_y, x] = Visualizer._parse_color(unknown_color)
            
            # Calculate scale factor for coordinate adjustment
            upsample_scale = grid_style.get('upsample_scale', 10)
            upsample_width = vis_image.shape[1] * upsample_scale
            upsample_height = vis_image.shape[0] * upsample_scale
            
            # Upsample
            vis_image = cv2.resize(vis_image, (upsample_width, upsample_height), interpolation=cv2.INTER_NEAREST)
            
            # Draw object bounding boxes (with scaled coordinates)
            if 'object' in style:
                object_style = style['object']
                object_thickness = int(object_style.get('thickness', -1))
                group_colors = object_style.get('group_colors', {'candidate': 'blue', 'reference': 'orange'})
                
                # Get object bounding boxes from keyframe entities
                for group_name, entities in self.entities.items():
                    if len(entities) == 0:
                        continue
                        
                    # Get color for this group
                    group_color = group_colors.get(group_name, object_style.get('color', 'blue'))
                    
                    # Process each entity in this group
                    for entity_id, entity in entities.items():
                        if not entity.is_object:
                            continue
                        if entity_id == -1:
                            continue
                            
                        # Get 3D bounding box corners
                        corners_3d = entity.corners_3d
                        if corners_3d is None:
                            continue
                        
                        # Convert 3D corners to occupancy grid coordinates
                        grid_corners = []
                        for corner in corners_3d:
                            # Convert world coordinates to grid coordinates
                            gx, gy = occupancy_grid_cropped.world_to_grid(corner[0], corner[1])
                            grid_corners.append([gx, gy])
                        
                        grid_corners = np.array(grid_corners)
                        
                        # Get bounding box of the footprint (top-down view)
                        grid_x_min = int(np.min(grid_corners[:, 0])) * upsample_scale
                        grid_y_min = int(np.min(grid_corners[:, 1])) * upsample_scale
                        grid_x_max = int(np.max(grid_corners[:, 0])) * upsample_scale
                        grid_y_max = int(np.max(grid_corners[:, 1])) * upsample_scale
                        
                        # Flip Y coordinates to match image coordinate system
                        flipped_y_min = upsample_height - 1 - grid_y_max
                        flipped_y_max = upsample_height - 1 - grid_y_min
                        
                        # Draw bounding box rectangle
                        if (0 <= grid_x_min < upsample_width and 0 <= flipped_y_min < upsample_height and
                            0 <= grid_x_max < upsample_width and 0 <= flipped_y_max < upsample_height):
                            
                            # Get alpha
                            alpha = object_style.get('alpha', 0.3)
                            vis_image = Visualizer.draw_rectangle(vis_image, ((int(grid_x_min), int(flipped_y_min)), (int(grid_x_max), int(flipped_y_max))), 
                                                      group_color, object_thickness, alpha)
                            
            # Draw movable points (with scaled coordinates)
            if 'movable_points' in style and movable_points_grid is not None and len(movable_points_grid) > 0:
                movable_style = style['movable_points']
                movable_color = movable_style.get('color', 'green')
                movable_radius = int(movable_style.get('radius', 3) * upsample_scale)
                movable_thickness = movable_style.get('thickness', -1)
                
                for i, (gx, gy) in enumerate(movable_points_grid):
                    # Adjust coordinates for crop and scale
                    adjusted_gx = (gx - crop_offset_x) * upsample_scale
                    adjusted_gy = (gy - crop_offset_y) * upsample_scale
                    
                    # Flip Y coordinate to match image coordinate system
                    flipped_gy = upsample_height - 1 - adjusted_gy
                    
                    if 0 <= adjusted_gx < upsample_width and 0 <= flipped_gy < upsample_height:
                        vis_image = Visualizer.draw_circle(vis_image, ((int(adjusted_gx), int(flipped_gy)), movable_radius), movable_color, movable_thickness)
            # Draw current position (with scaled coordinates)
            if 'current_point' in style and current_point is not None:
                current_style = style['current_point']
                current_color = current_style.get('color', 'red')
                current_radius = int(current_style.get('radius', 5))
                current_thickness = current_style.get('thickness', -1)
                
                gx, gy = current_point
                # Adjust coordinates for crop and scale
                adjusted_gx = (gx - crop_offset_x) * upsample_scale
                adjusted_gy = (gy - crop_offset_y) * upsample_scale
                
                # Flip Y coordinate to match image coordinate system
                flipped_gy = upsample_height - 1 - adjusted_gy
                
                if 0 <= adjusted_gx < upsample_width and 0 <= flipped_gy < upsample_height:
                    vis_image = Visualizer.draw_circle(vis_image, ((int(adjusted_gx), int(flipped_gy)), current_radius), current_color, current_thickness)
            # Draw text (with scaled coordinates)
            if 'text' in style and movable_points_grid is not None and len(movable_points_grid) > 0:
                text_style = style['text']
                text_size = text_style.get('size', 0.4)
                text_thickness = max(1, int(text_style.get('thickness', 1)))
                
                for i, (gx, gy) in enumerate(movable_points_grid):
                    # Determine text color based on distance index if enabled
                    text_color = text_style.get('color', 'green')
                    
                    # Check if distance-based colors are enabled
                    if (point_distance_indices is not None and len(point_distance_indices) > 0 and 
                        'distance_colors' in text_style and 
                        text_style['distance_colors'].get('enabled', False)):
                        
                        distance_colors_config = text_style['distance_colors']
                        colors = distance_colors_config.get('colors', ['green', 'red', 'blue'])
                        
                        # Use distance index to get color
                        if i < len(point_distance_indices):
                            dist_idx = point_distance_indices[i]
                            if dist_idx < len(colors):
                                text_color_name = colors[dist_idx]
                    
                    # Adjust coordinates for crop and scale
                    adjusted_gx = (gx - crop_offset_x) * upsample_scale
                    adjusted_gy = (gy - crop_offset_y) * upsample_scale
                    
                    # Flip Y coordinate to match image coordinate system
                    flipped_gy = upsample_height - 1 - adjusted_gy
                    
                    if 0 <= adjusted_gx < upsample_width and 0 <= flipped_gy < upsample_height:
                        vis_image = Visualizer.draw_text(vis_image, str(i), (adjusted_gx, flipped_gy), 
                                             text_color_name, text_size, text_thickness, True)
                        
        except Exception as e:
            logging.error(f"Error visualizing movable points on occupancy grid: {e}")
            print(f"Error visualizing movable points on occupancy grid: {e}")
            return None

        # Save the visualized image
        save_path = self.save_path(node_name=node_name, suffix=suffix)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        success = cv2.imwrite(save_path, vis_image)
        return save_path if success else None

    def _is_path_clear(self, start_gx, start_gy, end_gx, end_gy, grid_data, grid_width, grid_height):
        """
        Check if path from start to end point is not blocked (using Bresenham's line algorithm)
        
        Args:
            start_gx, start_gy: start point grid coordinates
            end_gx, end_gy: end point grid coordinates
            grid_data: occupancy grid data (2D numpy array)
            grid_width, grid_height: grid size
        
        Returns:
            bool: True if path is not blocked
        """
        # Use Bresenham's line algorithm to check all points along the path
        dx = abs(end_gx - start_gx)
        dy = abs(end_gy - start_gy)
        
        sx = 1 if start_gx < end_gx else -1
        sy = 1 if start_gy < end_gy else -1
        
        err = dx - dy
        x, y = start_gx, start_gy
        
        while True:
            # Check if current point is within grid bounds and occupied
            if 0 <= x < grid_width and 0 <= y < grid_height:
                if grid_data[y, x] > 50:  # occupied (consider as obstacle if >= 50)
                    return False
            
            if x == end_gx and y == end_gy:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

class Keyframes(Datas):
    def __init__(self, init=None, candidate_names=None, reference_names=None, *args, **kwargs):
        super().__init__(init=init, *args, **kwargs)
        self.candidate_names = list(candidate_names or [])
        self.reference_names = list(reference_names or [])
        self.related_names = list(candidate_names or []) + list(reference_names or [])

        self.place_id2entity_ids = None
        self.entity_id2place_ids = None

    def __repr__(self) -> str:
        repr = f"Keyframes(#{len(self)}): place_ids={list(self.keys())}"
        return repr

    def to_list(self):
        data = []
        T = type(self)
        for id, kf in self.items():
            data.append(T({id: kf}))
        return data

    def get_entity_ids(self, ids, etype: Literal['object', 'detection', 'all'] = 'object') -> "Keyframes":
        if not isinstance(ids, list):
            ids = [ids]
        keyframes = {}
        for id, kf in self.items():
            entity = kf.related_entities(etype).get(ids)
            if len(entity) > 0:
                keyframes.update({id: kf})
        return Keyframes(keyframes)

    def get_entity_names(self, names, etype: Literal['object', 'detection', 'all'] = 'object') -> "Keyframes":
        if not isinstance(names, list):
            names = [names]
        keyframes = {}
        for id, kf in self.items():
            if len(kf.related_entities(etype).get_names(names)) > 0:
                keyframes.update({id: kf})
        return Keyframes(keyframes)

    def get_closest_keyframe(self, position) -> "Keyframe":
        min_distance = float('inf')
        closest_id = None
        closest_kf = None
        for id, kf in self.items():
            kf_pose_matrix = np.array(kf.pose)
            kf_position = kf_pose_matrix[:3, 3]
            distance = np.linalg.norm(position - np.array(kf_position))
            if distance < min_distance:
                min_distance = distance
                closest_id = id
                closest_kf = kf
        return closest_id, closest_kf

    def on_update(self, key, old, value) -> None:
        is_changed = False
        for id_new, entity_new in value.related_entities.items():
            entities_old = old.related_entities.get(id_new)
            if len(entities_old) == 0:
                continue
            if not entities_old.single().equal(entity_new):
                is_changed = True
                break
            if is_changed:
                break

        if is_changed:
            pass # TODO: COMPUTE CENTER

    def update_id2id(self, scene_graph, places, objects) -> None:
        try:
            # ROS version
            place_id2entity_ids = {}
            entity_id2place_ids = {}
            for source, target, attr in scene_graph.edges(data=True):
                source_id, target_id = parse_id(source), parse_id(target)
                if 'place' in source:
                    if not source_id in place_id2entity_ids.keys():
                        place_id2entity_ids[source_id] = []
                if 'object' in target:
                    if not target_id in entity_id2place_ids.keys():
                        entity_id2place_ids[target_id] = []

                if ('place' in source) and ('object' in target):
                    place_id, entity_id = source_id, target_id
                    place_id2entity_ids[place_id].append(entity_id)
                    if not entity_id in objects.ids:
                        logging.warning(f"Entity id={entity_id} is not in {objects.ids}")

                    entity_id2place_ids[entity_id].append(place_id)
                    if not place_id in places.ids:
                        logging.warning(f"Place id={place_id} is not in {places.ids}")
            
            # Ensure all places have entries, even if they have no associated objects
            for place_id in places.ids:
                if place_id not in place_id2entity_ids:
                    place_id2entity_ids[place_id] = []
            
            self.place_id2entity_ids = place_id2entity_ids
            self.entity_id2place_ids = entity_id2place_ids
        except:
            # Python version
            place_id2entity_ids = {}
            entity_id2place_ids = {}
            for attr in scene_graph['links']:
                source, target = attr['source'], attr['target']
                source_id, target_id = parse_id(source), parse_id(target)

                if 'place' in source:
                    if not source_id in place_id2entity_ids.keys():
                        place_id2entity_ids[source_id] = []
                if 'object' in target:
                    if not target_id in entity_id2place_ids.keys():
                        entity_id2place_ids[target_id] = []

                if ('place' in source) and ('object' in target):
                    place_id, entity_id = source_id, target_id
                    place_id2entity_ids[place_id].append(entity_id)
                    if not entity_id in objects.ids:
                        logging.warning(f" Entity id={entity_id} is not in objects{objects.ids}")

                    entity_id2place_ids[entity_id].append(place_id)
                    if not place_id in places.ids:
                        logging.warning(f" Place id={place_id} is not in places{places.ids}")

            # Ensure all places have entries, even if they have no associated objects
            for place_id in places.ids:
                if place_id not in place_id2entity_ids:
                    place_id2entity_ids[place_id] = []

            self.place_id2entity_ids = place_id2entity_ids
            self.entity_id2place_ids = entity_id2place_ids

    def update(self, scene_graph, places, objects) -> None:
        self.update_id2id(scene_graph, places, objects)

        for id, place in places.items():
            object_ids = self.place_id2entity_ids[id]
            objects_in_place = objects.get(object_ids)
            candidate_objects_in_place = objects_in_place.get_names(self.candidate_names)
            reference_objects_in_place = objects_in_place.get_names(self.reference_names)
            self[id] = Keyframe(
                place, {'candidate': candidate_objects_in_place, 'reference': reference_objects_in_place})

    def annotate(self, styles=None, *args, **kwargs) -> Union[str, None]:
        """Annotate all keyframes in the collection"""
        for kf_id, kf in self.items():
            if kf.image is not None:
                kf.annotate(styles, *args, **kwargs)
            else:
                logging.warning(f"Skipping annotation for keyframe {kf_id}: no valid image")

class CurrentKeyframe(Keyframe):
    """Simplified Keyframe class that only requires image_path and image data"""
    
    def __init__(self, image_path: str = None, image: np.ndarray = None, pose = None, pose_dict = None):
        # Create a minimal Place-like object
        class MinimalPlace:
            def __init__(self, image_path, pose):
                self.image_path = image_path
                if image_path:
                    self.directory, self.filename = os.path.split(image_path)
                    self.ext = os.path.splitext(image_path)[1]
                else:
                    self.directory = None
                    self.filename = None
                    self.ext = None
                self.pose = pose
            
        # Create empty entities
        empty_entities = Entities({})
        
        # Initialize parent Keyframe with minimal data
        super().__init__(MinimalPlace(image_path, pose), empty_entities)
        
        # Override image if provided directly
        if image is not None:
            self.image = image
            
        self.image_path = image_path
        self.pose_dict = pose_dict
    
    @property
    def id(self) -> int:
        """Override id property to handle cases where filename might not exist"""
        if self.filename and '_' in self.filename:
            try:
                return int(self.filename.split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                return 0
        return 0
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Override image_size to handle cases where image might be None"""
        if self.image is not None:
            height, width, _ = self.image.shape
            return width, height
        return (0, 0)

if __name__ == "__main__":
    DATA_DIR = "/ws/external/test_data/offline_map"
    dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    dir_sorted = sorted(dirs, key=os.path.getmtime)

    kfs = Keyframes(candidate_names='pillow', reference_names=['sofa'])

    for dir in dir_sorted:
        with open(os.path.join(dir, 'scene_graph.json'), 'r', encoding='utf-8') as f:
            scene_graph = json.load(f)
        with open(os.path.join(dir, 'objects.json'), 'r', encoding='utf-8') as f:
            objects = json.load(f)
        # sg.update(scene_graph, objects)
        # kfs.update(sg)

        time.sleep(0.1)
