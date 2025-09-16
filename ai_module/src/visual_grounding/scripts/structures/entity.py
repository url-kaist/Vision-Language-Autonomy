import logging
import numpy as np
from typing import List, Literal, Union
from scipy.spatial import ConvexHull, QhullError
from .data import Data, Datas
from .utils import parse_id
from ai_module.src.visual_grounding.scripts.structures.bbox import BBox, BBoxes
from ai_module.src.sem.app.generate_seg_cloud import scan2pixels_wheelchair, scan2pixels_jackal

def _convex_hull_safe(points_2d: np.ndarray):
    if points_2d is None:
        return None
    pts = np.asarray(points_2d, dtype=float)

    # Filter NaN/Inf
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return None

    # Filter duplicated points
    pts = np.unique(pts, axis=0)
    n = len(pts)
    if n == 1: return pts.copy()
    if n == 2: return pts

    # Determine whether or not to be selected by rank
    rank = np.linalg.matrix_rank(pts - pts.mean(axis=0))
    if rank < 2:
        # Collinear: Return both endpoints
        i_min = np.argmin(pts[:, 0] + 1e-9 * pts[:, 1])
        i_max = np.argmax(pts[:, 0] + 1e-9 * pts[:, 1])
        if i_min == i_max:
            return pts[[i_min]]
        return pts[[i_min, i_max]]

    # Regular case: return ConvexHull
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except QhullError:
        logging.warning("QhullError occurs, return None")
        return None # TODO: convex_hull_monotonic_chain


def _convex_hull_safe(points_2d: np.ndarray):
    if points_2d is None:
        return None
    pts = np.asarray(points_2d, dtype=float)

    # Filter NaN/Inf
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return None

    # Filter duplicated points
    pts = np.unique(pts, axis=0)
    n = len(pts)
    if n == 1: return pts.copy()
    if n == 2: return pts

    # Determine whether or not to be selected by rank
    rank = np.linalg.matrix_rank(pts - pts.mean(axis=0))
    if rank < 2:
        # Collinear: Return both endpoints
        i_min = np.argmin(pts[:, 0] + 1e-9 * pts[:, 1])
        i_max = np.argmax(pts[:, 0] + 1e-9 * pts[:, 1])
        if i_min == i_max:
            return pts[[i_min]]
        return pts[[i_min, i_max]]

    # Regular case: return ConvexHull
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except QhullError:
        logging.warning("QhullError occurs, return None")
        return None # TODO: convex_hull_monotonic_chain


class Entity(Data):
    _rename_map = {'class_name': 'name', 'instance_id': 'id'}
    _delete_keys = ['class_ids', 'n_points', 'point_hash_key']
    _equal_keys = ['name', 'id']
    _repr_keys = ['name', 'id', 'is_object']

    def __init__(self, data, *args, **kwargs):
        self._init_vars()
        super().__init__(data, *args, **kwargs)

    def _eq_key(self):
        kind = 'obj' if self.is_object else 'det'
        # id가 반드시 존재한다고 가정하는 게 가장 안전합니다.
        # 혹시 None일 수 있다면, 동일 None끼리만 같도록 처리(= 사실상 병합 안 됨).
        return (kind, int(self.id) if self.id is not None else None)

    def __eq__(self, other):
        return isinstance(other, Entity) and self._eq_key() == other._eq_key()

    def __hash__(self):
        kind, eid = self._eq_key()
        if eid is None:
            # id가 없으면 해시가 불안정해지므로 개별 인스턴스로 취급(병합 X).
            # id를 항상 세팅할 수 있다면 이 분기는 없어도 됩니다.
            return hash((kind, id(self)))
        return hash((kind, eid))

    def _init_vars(self):
        self.name = None
        self.id = None
        self.bbox_by_kf = None
        self.min_bbox = None
        self.max_bbox = None
        self.points = None
        self.center = None
        self.updated_points = False
        
    def update(self, data):
        mapped = {self._rename_map.get(k, k): v for k, v in data.items() if not k in self._delete_keys}
        for k, new in mapped.items():
            ## Debug
            # old = getattr(self, k) if hasattr(self, k) else None
            # if old is not None:
            #     if old != new:
            #         logging.warning(f"Entity[{self.id}] already has {k}, but tried to update the data: \n"
            #                         f"  > old: {old}\n"
            #                         f"  > new: {new}")
            setattr(self, k, new)

    @property
    def convex_hull(self):
        assert self.is_object, f"Only object has convex_hull"
        points = self.points
        if points is None:
            return None

        points_2d = np.array(points)[:, :2]
        hull_pts = _convex_hull_safe(points_2d)
        if hull_pts is None or len(hull_pts) == 0:
            return None
        return np.asarray(hull_pts)

    @property
    def corners_3d(self):
        assert self.is_object, f"Only object has corners_3d"
        min_bbox, max_bbox = self.min_bbox, self.max_bbox
        x_min, y_min, z_min = min_bbox
        x_max, y_max, z_max = max_bbox
        corners_3d = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max]
        ])
        return corners_3d

    @property
    def width(self):
        assert self.is_object, f"Only object has width"
        return float(self.max_bbox[0] - self.min_bbox[0])
    
    @property
    def depth(self):
        assert self.is_object, f"Only object has depth"
        return float(self.max_bbox[1] - self.min_bbox[1])


    @staticmethod
    def project_3d_point_to_image(points_3d, pose, image_size=None, is_real_world=False):
        """
        Args:
            - image_size: (width, height)
        Return: point_pixel_idx: (u, v, depth)
        """
        if not isinstance(pose, np.ndarray):
            pose = np.array(pose)
        points_3d = points_3d.reshape(-1, 3)
        rotation = pose[:3, :3]
        position = pose[:3, 3]
        R_w2b = rotation.T
        t_w2b = -R_w2b @ position
        points_3d_body = points_3d @ R_w2b.T + t_w2b
        if is_real_world:
            point_pixel_idx = scan2pixels_jackal(points_3d_body)
            point_pixel_idx[:, 1] += 200
        else:
            point_pixel_idx = scan2pixels_wheelchair(points_3d_body)
        if image_size is not None:
            out_of_bound_filter = (
                    (point_pixel_idx[:, 0] >= 0) & (point_pixel_idx[:, 0] < image_size[0]) &
                    (point_pixel_idx[:, 1] >= 0) & (point_pixel_idx[:, 1] < image_size[1])
            )
            point_pixel_idx = point_pixel_idx[out_of_bound_filter]

        return point_pixel_idx
    
    @staticmethod
    def split_x(xs: np.ndarray, image_width: int):
        xs = sorted(xs)
        num_pts = len(xs)
        gaps = [xs[i + 1] - xs[i] for i in range(num_pts - 1)]
        gaps.append(xs[0] + image_width - xs[-1])

        max_gap_idx = max(range(num_pts), key=lambda k: gaps[k])
        max_gap = gaps[max_gap_idx]

        start = xs[(max_gap_idx + 1) % num_pts]
        end = xs[max_gap_idx]

        is_split = (end < start)
        return (start, end if not is_split else end + image_width)

    @property
    def is_object(self):
        return (self.points is not None)

    def get_bbox(self, pose=None, image_size=None, is_real_world=False, kf_id=None, *args, **kwargs) -> Union[BBox, None]:
        if self.is_object and (kf_id not in self.bbox_by_kf):
            assert pose and image_size, f"both pose and image_size are required to bbox of the object"
            image_width, image_height = image_size
            point_pixel_idx = self.project_3d_point_to_image(
                self.corners_3d, pose, image_size, is_real_world=is_real_world)
            u_min, u_max = self.split_x(point_pixel_idx[:, 0], image_width)
            v_min, v_max = int(np.min(point_pixel_idx[:, 1])), int(np.max(point_pixel_idx[:, 1]))
            bbox = np.array([u_min, v_min, u_max, v_max], dtype=np.uint32)
        else:
            bbox = self.bbox_by_kf.get(kf_id)
            bbox = np.array(bbox, dtype=np.uint32)
        return BBox(bbox, object_id=self.id)
    
    def get_closest_traversable_point(self, traversable_points, *args, **kwargs):
        if traversable_points is None or len(traversable_points) == 0:
            return None
                
        if self.is_object:
            try:
                agent_position = kwargs.get('agent_position', self.center)
                tol_line = kwargs.get('tol_line', 1.0)
                w_pos = kwargs.get('w_pos', 1.0)
                w_agent = kwargs.get('w_agent', 0.1)
                
                c1 = np.array([self.center[0], self.center[1]]) # target point
                c2 = np.array([agent_position[0], agent_position[1]]) # agent position
                
                u = c2 - c1
                L = float(np.linalg.norm(u))
                if L < 1e-9:
                    # If the two objects are almost at the same position, use an arbitrary axis
                    u = np.array([1.0, 0.0], dtype=float); L = 1.0
                u /= L  # unit vector along the center line

                # Traversable points(2D)
                T = np.asarray(traversable_points, dtype=float)
                T2 = T[:, :2]

                # Calculate the projection of each traversable point onto the center line
                # q = c1 + t*u, t ∈ [0, L]
                v = T2 - c1[None, :]
                t = (v @ u)                 # (M,)
                perp = v - t[:, None] * u[None, :]
                d_perp = np.linalg.norm(perp, axis=1)  # distance to the center line

                # Filter points that are around the center line
                mask = (d_perp <= tol_line) & (t >= -0.1*L) & (t <= 1.1*L)
                if not np.any(mask):
                    cand = T2
                else:
                    cand = T2[mask]

                # Calculate the distance from the target position and agent position
                d_pos = np.linalg.norm(cand - c1[None, :], axis=1)
                d_agent = np.linalg.norm(cand - c2[None, :], axis=1)

                score = w_pos * d_pos + \
                        w_agent * d_agent

                k = int(np.argmin(score))
                p_best = cand[k]

                # Get the closest traversable point to the best candidate
                closest_point = [[float(p_best[0]), float(p_best[1]), 0.0]]            
            except Exception as e:
                print(f"Error in get_closest_traversable_point (object): {e}")
                return None
        else:
            try:
                kf_id = kwargs.get('kf_id')
                kf = kwargs.get('kf')

                if kf_id is None or kf is None:
                    return None
                
                points = self.project_3d_point_to_image(traversable_points, kf.pose, is_real_world=kf.is_real_world) # image plane points
                
                # Filter out of bound points
                oob_filter = (points[:, 0] >= 0) & (points[:, 0] < kf.image_size[0]) & (points[:, 1] >= 0) & (points[:, 1] < kf.image_size[1])
                points = points[oob_filter]
                traversable_points = traversable_points[oob_filter]
                
                # Filter points that are similar column
                bbox = self.bbox_by_kf.get(kf_id)
                x_min, y_min, x_max, y_max = bbox
                center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                
                center_x = center[0]
                same_column = np.abs(points[:, 0] - center_x) < 1
                same_column_points = points[same_column]
                traversable_points_same_column = traversable_points[same_column]

                # Filter points that are above the minimum y-coordinate
                below_y_max = same_column_points[:, 1] > y_max
                points_below_y_max = same_column_points[below_y_max]
                traversable_points_below_y_max = traversable_points_same_column[below_y_max]
                
                if len(points_below_y_max) > 0:
                    # Return the point with the largest y-coordinate (the lowest point)
                    max_y_idx = np.argmin(points_below_y_max[:, 1])
                    closest_point = traversable_points_below_y_max[max_y_idx]
                    
                    return closest_point
                else:
                    return None
            except Exception as e:
                print(f"Error in get_closest_traversable_point (detection): {e}")
                return None
            
        return closest_point

class Entities(Datas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        repr = f"Entities(#{len(self)}):\n"
        for id, object in self.items():
            repr += f"  > [{id}] {object}\n"
        return repr

    def __eq__(self, other) -> bool:
        if not isinstance(other, Entities):
            return False
        
        # Compare the number of entities
        if len(self) != len(other):
            return False
        
        # Compare each entity by ID and content
        for id in self.keys():
            if id not in other:
                return False
            if not self[id].equal(other[id]):
                return False
        
        return True

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __call__(self, etype: Literal['object', 'detection', 'all'] = 'object', include_untracked: bool = True):
        if etype == 'object':
            if include_untracked:
                entities = {id: entity for id, entity in self.items() if entity.is_object}
            else:
                entities = {id: entity for id, entity in self.items() if entity.is_object and entity.id > 0}
            return type(self)(entities)
        elif etype == 'detection':
            if include_untracked:
                entities = {id: entity for id, entity in self.items() if not entity.is_object}
            else:
                entities = {id: entity for id, entity in self.items() if not entity.is_object and entity.id > 0}
            return type(self)(entities)
        elif etype == 'all':
            if include_untracked:
                return self
            else:
                entities = {id: entity for id, entity in self.items() if entity.id > 0}
                return type(self)(entities)
        elif etype == 'image':
            return type(self)()
        else:
            raise TypeError(f"`etype` must be in ['object', 'detection', 'entity', 'all', 'image'], but {etype} was given.")

    def get_names(self, names):
        if not isinstance(names, list):
            names = [names]
        entities = {id: entity for id, entity in self.items() if entity.name in names}
        return type(self)(entities)

    def name2ids(self, name) -> List[int]:
        ids = [id for id, entity in self.items() if entity.name == name]
        return ids

    def update(self, scene_graph, entities, dtype: Literal['object', 'detection', 'all'] = 'object') -> None:
        try:
            # ROS version
            for id_str, data in scene_graph.nodes.items():
                if 'object' in id_str:
                    eid = parse_id(id_str)
                    data['id'] = eid
                    if 'bbox_by_kf' in data:
                        data['bbox_by_kf'] = {int(k): v for k, v in data['bbox_by_kf'].items()}
                    if eid in self.keys():  # Update
                        self[eid].update(data)
                    else:  # Add
                        self[eid] = Entity(data)
        except:
            # Python version
            for data in scene_graph.get('nodes', []):
                id_str = data.get('id')
                if 'object' in id_str:
                    eid = parse_id(id_str)
                    data['id'] = eid
                    if 'bbox_by_kf' in data:
                        data['bbox_by_kf'] = {int(k): v for k, v in data['bbox_by_kf'].items()}
                    if eid in self.keys():  # Update
                        self[eid].update(data)
                    else:  # Add
                        self[eid] = Entity(data)

        if entities is not None:
            for eid, data in entities.items():
                eid = int(eid)
                if eid in self.keys():  # Update
                    self[eid].update(data)
                else:  # Add
                    self[eid] = Entity(data)

    @classmethod
    def merge(cls, left, right, prefer='error'):
        res = cls()

        for k, v in left.items():
            res[k] = v

        for k, v in right.items():
            if k not in res:
                res[k] = v
                continue

            if prefer == 'error':
                raise KeyError(f"ID conflict on {k}: left={res[k]}, right={v}")
            else:
                raise ValueError(f"Invalid prefer={prefer}")

        return res

    def get_bboxes(self, *args, **kwargs) -> BBoxes:
        bboxes = []
        for object_id, object_data in self.items():
            bbox = object_data.get_bbox(*args, **kwargs)
            bboxes.append(bbox)
        return BBoxes(bboxes)


if __name__ == "__main__":
    pass
