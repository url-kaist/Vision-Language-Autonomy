import numpy as np
from typing import Optional, Tuple, Union, List


def rewrap_image(image: np.ndarray, cut_x: float) -> np.ndarray:
    height, width = image.shape[:2]
    x = int(round(cut_x)) % width
    return np.concatenate([image[:, x:], image[:, :x]], axis=1)


class BBox:
    def __init__(self, data, object_id=None):
        self.data = data
        self.object_id = object_id

    def __repr__(self):
        return f"BBox(data={self.data.tolist()}, object_id={self.object_id})"

    @property
    def u_min(self) -> int: return int(self.data[0])

    @property
    def v_min(self) -> int: return int(self.data[1])

    @property
    def u_max(self) -> int: return int(self.data[2])

    @property
    def v_max(self) -> int: return int(self.data[3])

    def with_u(self, u_min: Union[int, float], u_max: Union[int, float]) -> "BBox":
        d = self.data.copy()
        d[0], d[2] = u_min, u_max
        return BBox(d, self.object_id)

    def shift_u(self, delta: float) -> "BBox":
        return self.with_u(self.u_min + delta, self.u_max + delta)

    def mod_u(self, width: float) -> "BBox":
        u0 = self.u_min % width
        u1 = self.u_max % width
        if u0 > u1:
            u0, u1 = min(u0, u1), max(u0, u1)
        return self.with_u(u0, u1)

    @property
    def area(self) -> float:
        area = (self.u_max - self.u_min) * (self.v_max - self.v_min)
        return float(area)


class BBoxes:
    def __init__(self, boxes=None):
        self.boxes = list(boxes) if boxes is not None else []

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def __getitem__(self, idx):
        return self.boxes[idx]

    @classmethod
    def from_array(cls, arr: np.ndarray, object_ids=None):
        if object_ids is None:
            object_ids = [None] * len(arr)
        boxes = [BBox(data, oid) for data, oid in zip(arr, object_ids)]
        return cls(boxes)

    def __repr__(self):
        return f"BBoxes(n={len(self)}, boxes={self.boxes})"

    def append(self, bbox: BBox):
        self.boxes.append(bbox)

    def extend(self, bboxes: "BBoxes"):
        self.boxes.extend(bboxes.boxes)

    def to_array(self) -> np.ndarray:
        if not self.boxes:
            return np.empty((0, 4))
        return np.stack([b.data for b in self.boxes], axis=0)

    @property
    def union_area(self) -> float:
        arr = self.to_array()
        if arr.shape[0] == 0:
            return 0.0

        # 유효 박스만 사용 (면적 > 0)
        u1, v1, u2, v2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        mask = (u2 > u1) & (v2 > v1)
        rects = arr[mask]
        if rects.shape[0] == 0:
            return 0.0

        # x-스윕 이벤트: (x, type, y1, y2), type: +1(진입), -1(이탈)
        events: List[Tuple[float, int, float, float]] = []
        for a in rects:
            events.append((float(a[0]), +1, float(a[1]), float(a[3])))
            events.append((float(a[2]), -1, float(a[1]), float(a[3])))
        events.sort(key=lambda e: e[0])

        active: List[Tuple[float, float]] = []  # (y1, y2)

        def merged_y_len(intervals: List[Tuple[float, float]]) -> float:
            if not intervals:
                return 0.0
            intervals = sorted(intervals)  # y1 기준
            total = 0.0
            y1, y2 = intervals[0]
            for a, b in intervals[1:]:
                if a <= y2:  # 겹치거나 닿으면 병합
                    if b > y2:
                        y2 = b
                else:
                    total += (y2 - y1)
                    y1, y2 = a, b
            total += (y2 - y1)
            return total

        area = 0.0
        x_prev = events[0][0]
        i = 0
        while i < len(events):
            x = events[i][0]
            area += merged_y_len(active) * (x - x_prev)
            x_prev = x

            # 같은 x의 이벤트 모두 처리
            while i < len(events) and events[i][0] == x:
                _, typ, y1, y2 = events[i]
                if typ == +1:
                    active.append((y1, y2))
                else:
                    # 첫 일치 항목 하나 제거
                    for j, (ay1, ay2) in enumerate(active):
                        if ay1 == y1 and ay2 == y2:
                            active.pop(j)
                            break
                i += 1

        return float(area)

    def sort_by_u_min(self, inplace=True, reverse=False):
        sorted_bboxes = sorted(self.boxes, key=lambda b: b.u_min, reverse=reverse)
        if inplace:
            self.boxes = sorted_bboxes
            return self
        return BBoxes(sorted_bboxes)

    def find_largest_gap(self, image_width: float) -> Tuple[float, int]:
        if len(self.boxes) == 0:
            return 0, -1
        if len(self.boxes) == 1:
            return image_width, 0

        self.sort_by_u_min()
        num_boxes = len(self)
        max_gap = float("-inf")
        max_gap_idx = 0
        for curr_i in range(num_boxes):
            next_i = (curr_i + 1) % num_boxes
            curr_u_max = self.boxes[curr_i].u_max
            next_u_min = self.boxes[next_i].u_min + (image_width if next_i == 0 else 0.0)
            gap = next_u_min - curr_u_max
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = curr_i

        return max_gap, max_gap_idx

    def compute_cut_x(self, image_width: float) -> float:
        num_boxes = len(self.boxes)
        if num_boxes == 0:
            return 0.0
        if num_boxes == 1:
            b = self.boxes[0]
            return ((b.u_min + b.u_max) / 2.0) % image_width

        _, idx = self.find_largest_gap(image_width)
        next_idx = (idx + 1) % num_boxes

        leftmost_u_min = self.boxes[idx].u_max
        rightmost_u_max = self.boxes[next_idx].u_min

        cut_x = leftmost_u_min + ((rightmost_u_max - leftmost_u_min + image_width) % image_width) / 2.0
        return cut_x % image_width

    def rewrap(self, image_width: float, cut_x: Optional[float]=None, inplace=True) -> "BBoxes":
        if cut_x is None:
            cut_x = self.compute_cut_x(image_width)

        shifted = [b.shift_u(-cut_x).mod_u(image_width) for b in self.boxes]

        if inplace:
            self.boxes = shifted
            return self
        else:
            return BBoxes(shifted)

    def rewrap_image(self, image: np.ndarray, inplace=False) -> Tuple["BBoxes", np.ndarray]:
        width = image.shape[1]
        cut_x = self.compute_cut_x(width)

        out_bboxes = self.rewrap(width, cut_x=cut_x, inplace=inplace)
        out_image = rewrap_image(image, cut_x)
        return (self if inplace else out_bboxes), out_image

    def get_objects(self, object_ids):
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        return [bbox for bbox in self.boxes if bbox.object_id in object_ids]


if __name__ == "__main__":
    pass
