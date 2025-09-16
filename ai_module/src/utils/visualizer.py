import cv2
import colorsys
import numpy as np
from typing import Tuple, Union, List

from ai_module.src.visual_grounding.scripts.structures.bbox import BBox, BBoxes


_rgb_to_bgr = lambda rgb: (rgb[2], rgb[1], rgb[0])


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int,int,int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)  # 0..1
    return (int(b * 255 + 0.5), int(g * 255 + 0.5), int(r * 255 + 0.5))


BLIND_FRIENDLY_COLOR_MAP = {
    'orange': (230, 159, 0),
    'sky blue': (86, 180, 233),
    'bluish green': (0, 158, 115),
    'yellow': (240, 228, 66),
    'blue': (0, 114, 178),
    'vermillion': (213, 94, 0),
    'reddish purple': (204, 121, 167),
    'black': (0, 0, 0),
    'gray': (187, 187, 187),
    'purple (Tol)': (51, 34, 136),
    'light blue': (136, 204, 238),
    'green': (17, 119, 51),
    'olive': (153, 153, 51),
    'sand': (221, 204, 119),
}

BASIC_COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
    "gray": (128, 128, 128),
}


def _color_palette(n: int, alpha=None) -> List[Tuple[int, int, int]]:
    """
    OpenCV BGR 색 팔레트. n개 요청 시 선명하고 서로 다른 색을 반환.
    앞 12~14개는 색약 친화 팔레트 기반, 이후는 HSV 황금비 샘플링.
    """
    base = [_rgb_to_bgr(c) for c in BLIND_FRIENDLY_COLOR_MAP.values()]

    if n <= len(base):
        return base[:n]

    # 부족분은 HSV에서 황금비 간격으로 생성(인접 색이 비슷해지지 않게)
    out = list(base)
    phi = 0.6180339887498948
    h = 0.0  # 시작 hue (고정하면 재현성 확보)
    need = n - len(base)

    for i in range(need):
        h = (h + phi) % 1.0  # hue 골든 스텝
        # 채도/명도는 약간씩 바꿔 인접색 대비 확보
        s = 0.70 if (i % 3 == 0) else (0.85 if (i % 3 == 1) else 0.95)
        v = 0.95 if (i % 2 == 0) else 0.82
        out.append(_hsv_to_bgr(h, s, v))

    if alpha is not None:
        out = [tuple(list(c) + [alpha]) for c in out]

    return out[:n]


class Visualizer:
    _COLOR_MAP = BASIC_COLOR_MAP

    def __init__(self):
        pass

    def draw_box(self, image: np.ndarray, bbox: BBox,
                 color: Union[Tuple[int, int, int], str] = "green",
                 alpha: float = 1.0, draw_id: bool = True) -> np.ndarray:
        color = self._parse_color(color)

        top_left = (bbox.u_min, bbox.v_min)
        bottom_right = (bbox.u_max, bbox.v_max)
        
        # Create overlay for alpha blending
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color=color, thickness=2)
        
        # Apply alpha blending
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        else:
            image = overlay
        
        # Draw ID text if requested
        if draw_id:
            cv2.putText(
                image, f"{bbox.object_id}", (bbox.u_min, bbox.v_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        
        return image

    @classmethod
    def _parse_color(cls, color: Union[Tuple[int, int, int], str]) -> tuple:
        if isinstance(color, str):
            if color.lower() not in cls._COLOR_MAP:
                raise ValueError(f"Unknown color name: {color}")
            return cls._COLOR_MAP[color.lower()]
        elif isinstance(color, tuple) and len(color) == 3:
            return color
        else:
            raise TypeError("color must be str or tuple of length 3 (BGR)")

    @classmethod
    def blend_color(self, color1: Union[Tuple[int, int, int], str], color2: Union[Tuple[int, int, int], str],
                    alpha: float = 0.5) -> Union[Tuple[int, int, int], str]:
        color1 = self._parse_color(color1)
        color2 = self._parse_color(color2)
        return tuple(np.array(color1) * alpha + np.array(color2) * (1 - alpha))

    def draw_bboxes(self, image: np.ndarray, bboxes: BBoxes,
                    color: Union[Tuple[int, int, int], str] = "green",
                    alpha: float = 1.0, draw_id: bool = True) -> np.ndarray:
        color = self._parse_color(color)

        img = image.copy()
        for bbox in bboxes:
            img = self.draw_box(img, bbox, color=color, alpha=alpha, draw_id=draw_id)
        return img

    @classmethod
    def draw_rectangle(self, image: np.ndarray, rectangle: Tuple[Tuple[int, int], Tuple[int, int]],
                       color: Union[Tuple[int, int, int], str] = "green", thickness: int = 2,
                       alpha: float = 1.0) -> np.ndarray:
        color = self._parse_color(color)

        top_left, bottom_right = rectangle
        
        # Create overlay for alpha blending
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color=color, thickness=thickness)
        
        # Apply alpha blending
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        else:
            image = overlay
            
        return image
    
    @classmethod
    def draw_circle(self, image: np.ndarray, circle: Tuple[Tuple[int, int], int],
                    color: Union[Tuple[int, int, int], str] = "green", thickness: int = 2,
                    alpha: float = 1.0) -> np.ndarray:
        color = self._parse_color(color)
        overlay = image.copy()
        cv2.circle(overlay, circle[0], circle[1], color, thickness)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        else:
            image = overlay
        return image
    
    @classmethod
    def draw_text(self, image: np.ndarray, text: str, text_position: Tuple[int, int],
                  color: Union[Tuple[int, int, int], str] = "green",
                  font_size: float = 0.8, thickness: int = 2, center = False,
                  alpha: float = 1.0) -> np.ndarray:
        color = self._parse_color(color)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
        
        if center:
            text_x = int(text_position[0] - text_width // 2)
            text_y = int(text_position[1] + text_height // 2)
        else:
            text_x = text_position[0]
            text_y = text_position[1]
        
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        return image