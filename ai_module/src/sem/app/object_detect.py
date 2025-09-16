import torch
import numpy as np
from ultralytics import YOLO, SAM


class ObjectDetector:
    def __init__(
        self,
        object_classes,
        device="cuda",
        yolo_model_path="yolov8l-world.pt",
        sam_model_path="mobile_sam.pt",  # "sam2.1_l.pt",
    ):
        self.device = device
        self.object_classes = object_classes

        self.yolo_model = YOLO(yolo_model_path).to(device)
        self.yolo_model.eval()
        self.sam_model = SAM(sam_model_path).to(device)
        self.sam_model.eval()

        template = "a photo of a {}."
        self.classnames = [
            template.format(name.replace("_", " ")) for name in self.object_classes
        ]
        self.yolo_model.set_classes(self.classnames)

    def infer(self, image, conf_threshold=0.4):
        yolo_result = self.yolo_model.predict(image.copy(), conf=conf_threshold)[0]
        class_ids = yolo_result.boxes.cls.int().tolist()

        if not class_ids:
            return [], [], [], None

        boxes = yolo_result.boxes.xyxy  # todo, boxes scale not match
        confs = yolo_result.boxes.conf.cpu().numpy()
        sam_results = self.sam_model.predict(image.copy(), bboxes=boxes)
        masks = sam_results[0].masks.data  # torch.Tensor (N, H, W)

        boxes = []
        for mask in masks:
            box = self.mask_to_bbox(mask)
            if box:
                boxes.append(box)

        return class_ids, boxes, confs, masks

    def mask_to_bbox(self, mask: torch.Tensor):
        y_indices, x_indices = torch.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x1, y1 = x_indices.min().item(), y_indices.min().item()
        x2, y2 = x_indices.max().item(), y_indices.max().item()
        return [x1, y1, x2, y2]
