from typing import Dict, List, Union

import albumentations as A
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

from .box_utils import decode, decode_landm
from .network import RetinaFace
from .prior_box import priorbox
from .utils import tensor_from_rgb_image, pad_to_size, unpad_from_size


class Model:

    def __init__(self, max_size: int = 960, device: str = 'cpu') -> None:
        self.model = RetinaFace(
            name='Resnet50',
            pretrained=False,
            return_layers={
                'layer2': 1,
                'layer3': 2,
                'layer4': 3
            },
            in_channels=256,
            out_channels=256,
        ).to(device)
        self.device = device
        self.transform = A.Compose(
            [A.LongestMaxSize(max_size=max_size, p=1),
             A.Normalize(p=1)])
        self.max_size = max_size
        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(self.max_size, self.max_size),
        ).to(device)
        self.variance = [0.1, 0.2]

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self):
        self.model.eval()

    def predict_jsons(
            self,
            image: np.array,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4) -> List[Dict[str, Union[List, float]]]:
        with torch.no_grad():
            original_height, original_width = image.shape[:2]

            scale_landmarks = torch.from_numpy(
                np.tile([self.max_size, self.max_size],
                        5)).to(self.device).float()
            scale_bboxes = torch.from_numpy(
                np.tile([self.max_size, self.max_size],
                        2)).to(self.device).float()

            transformed_image = self.transform(image=image)['image']

            paded = pad_to_size(
                target_size=(self.max_size, self.max_size),
                image=transformed_image)

            pads = paded['pads']

            torched_image = tensor_from_rgb_image(paded['image']).to(
                self.device)

            loc, conf, land = self.model(torched_image.unsqueeze(0))

            conf = F.softmax(conf, dim=-1)

            annotations: List[Dict[str, Union[List, float]]] = []

            boxes = decode(loc.data[0], self.prior_box, self.variance)

            boxes *= scale_bboxes
            scores = conf[0][:, 1]

            landmarks = decode_landm(land.data[0], self.prior_box,
                                     self.variance)
            landmarks *= scale_landmarks

            # ignore low scores
            valid_index = scores > confidence_threshold
            boxes = boxes[valid_index]
            landmarks = landmarks[valid_index]
            scores = scores[valid_index]

            # Sort from high to low
            order = scores.argsort(descending=True)
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            # do NMS
            keep = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep, :].int()

            if boxes.shape[0] == 0:
                return [{'bbox': [], 'score': -1, 'landmarks': []}]

            landmarks = landmarks[keep]

            scores = scores[keep].cpu().numpy().astype(np.float64)
            boxes = boxes.cpu().numpy()
            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape([-1, 2])

            unpadded = unpad_from_size(pads, bboxes=boxes, keypoints=landmarks)

            resize_coeff = max(original_height, original_width) / self.max_size

            boxes = (unpadded['bboxes'] * resize_coeff).astype(int)
            landmarks = (unpadded['keypoints'].reshape(-1, 10)
                         * resize_coeff).astype(int)

            for box_id, bbox in enumerate(boxes):
                x_min, y_min, x_max, y_max = bbox

                x_min = np.clip(x_min, 0, original_width - 1)
                x_max = np.clip(x_max, x_min + 1, original_width - 1)

                if x_min >= x_max:
                    continue

                y_min = np.clip(y_min, 0, original_height - 1)
                y_max = np.clip(y_max, y_min + 1, original_height - 1)

                if y_min >= y_max:
                    continue

                annotations += [{
                    'bbox':
                    bbox.tolist(),
                    'score':
                    scores[box_id],
                    'landmarks':
                    landmarks[box_id].reshape(-1, 2).tolist(),
                }]

            return annotations

    def predict_jsons_batch(
            self,
            images: list,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4) -> list:
        """Run RetinaFace on a list of RGB numpy arrays; return per-image annotation lists."""
        with torch.no_grad():
            scale_landmarks = torch.from_numpy(
                np.tile([self.max_size, self.max_size], 5)).to(self.device).float()
            scale_bboxes = torch.from_numpy(
                np.tile([self.max_size, self.max_size], 2)).to(self.device).float()

            original_sizes, pads_list, tensors = [], [], []
            for image in images:
                oh, ow = image.shape[:2]
                original_sizes.append((oh, ow))
                transformed = self.transform(image=image)['image']
                paded = pad_to_size(target_size=(self.max_size, self.max_size), image=transformed)
                pads_list.append(paded['pads'])
                tensors.append(tensor_from_rgb_image(paded['image']))

            batch = torch.stack(tensors).to(self.device)
            loc, conf, land = self.model(batch)
            conf = F.softmax(conf, dim=-1)

        results_per_image = []
        for i, (oh, ow) in enumerate(original_sizes):
            boxes = decode(loc.data[i], self.prior_box, self.variance)
            boxes *= scale_bboxes
            scores = conf[i][:, 1]
            landmarks = decode_landm(land.data[i], self.prior_box, self.variance)
            landmarks *= scale_landmarks

            valid = scores > confidence_threshold
            boxes, landmarks, scores = boxes[valid], landmarks[valid], scores[valid]

            order = scores.argsort(descending=True)
            boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

            keep = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep, :].int()

            if boxes.shape[0] == 0:
                results_per_image.append([{'bbox': [], 'score': -1, 'landmarks': []}])
                continue

            landmarks = landmarks[keep]
            scores    = scores[keep].cpu().numpy().astype(np.float64)
            boxes     = boxes.cpu().numpy()
            landmarks = landmarks.cpu().numpy().reshape([-1, 2])

            unpadded = unpad_from_size(pads_list[i], bboxes=boxes, keypoints=landmarks)
            resize_coeff = max(oh, ow) / self.max_size
            boxes     = (unpadded['bboxes'] * resize_coeff).astype(int)
            landmarks = (unpadded['keypoints'].reshape(-1, 10) * resize_coeff).astype(int)

            annotations = []
            for box_id, bbox in enumerate(boxes):
                x_min, y_min, x_max, y_max = bbox
                x_min = np.clip(x_min, 0, ow - 1)
                x_max = np.clip(x_max, x_min + 1, ow - 1)
                if x_min >= x_max:
                    continue
                y_min = np.clip(y_min, 0, oh - 1)
                y_max = np.clip(y_max, y_min + 1, oh - 1)
                if y_min >= y_max:
                    continue
                annotations.append({
                    'bbox':      bbox.tolist(),
                    'score':     scores[box_id],
                    'landmarks': landmarks[box_id].reshape(-1, 2).tolist(),
                })

            results_per_image.append(
                annotations if annotations else [{'bbox': [], 'score': -1, 'landmarks': []}]
            )

        return results_per_image
