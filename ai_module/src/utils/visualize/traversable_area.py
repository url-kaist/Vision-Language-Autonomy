import os
import cv2
import json
import numpy as np
import open3d as o3d


def visualize_traversable_area(traversable_area: np.ndarray, scale: int = 100, pad: int = 10):
    """
    traversable_area: (N,2) or (N,>=2) → x,y만 사용
    반환: image, meta(dict: xmin,xmax,ymin,ymax,scale,pad,W,H)
    """
    xy = np.asarray(traversable_area, dtype=float)[:, :2]
    x = xy[:, 0]
    y = xy[:, 1]

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    W = int((xmax - xmin) * scale) + 2 * pad
    H = int((ymax - ymin) * scale) + 2 * pad

    image = np.zeros((H, W, 3), dtype=np.uint8)
    for xi, yi in xy:
        col = int(round((xi - xmin) * scale)) + pad
        row = int(round((ymax - yi) * scale)) + pad
        if 0 <= row < H and 0 <= col < W:
            image[row, col] = (0, 255, 0)  # BGR: 연두

    meta = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                scale=scale, pad=pad, W=W, H=H)
    return image, meta


if __name__ == '__main__':
    DATA_DIR = f"/ws/external/test_data/hotel_room_1"
    traversable_points_path = os.path.join(DATA_DIR, "traversable_area.ply")
    traversable_img_path = os.path.join(DATA_DIR, "traversable_area.jpg")
    traversable_meta_path = os.path.join(DATA_DIR, "traversable_area_meta.json")

    traversable_points = o3d.io.read_point_cloud(traversable_points_path)
    traversable_points = np.asarray(traversable_points.points)
    traversable_img, traversable_meta = visualize_traversable_area(traversable_points)

    cv2.imwrite(traversable_img_path, traversable_img)
    with open(traversable_meta_path, "w", encoding="utf-8") as f:
        json.dump(traversable_meta, f, ensure_ascii=False, indent=2)
