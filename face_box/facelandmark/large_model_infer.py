import sys
sys.path.append('../')
import numpy as np
from face_box.retinaface.predict_single import Model
import cv2
from face_box.facelandmark.large_base_lmks_infer import LargeBaseLmkInfer
import math
import torch
import time
import os
INPUT_SIZE = 224
ENLARGE_RATIO = 1.35
# from util.util_ import spread_flow, viz_flow
# from util.image_liquify import image_warp_grid1


def resize_on_long_side(img, long_side=800):
    src_height = img.shape[0]
    src_width = img.shape[1]

    if src_height > src_width:
        scale = long_side * 1.0 / src_height
        _img = cv2.resize(img, (int(src_width * scale), long_side), interpolation=cv2.INTER_CUBIC)


    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(img, (long_side, int(src_height * scale)), interpolation=cv2.INTER_CUBIC)

    return _img, scale

def draw_line(im, points, color, stroke_size=2, closed=False):
    points = points.astype(np.int32)
    for i in range(len(points) - 1):
        cv2.line(im, tuple(points[i]), tuple(points[i + 1]), color, stroke_size)
    if closed:
        cv2.line(im, tuple(points[0]), tuple(points[-1]), color, stroke_size)

def enlarged_bbox(bbox, img_width, img_height, enlarge_ratio=0.2):
    '''
    :param bbox: [xmin,ymin,xmax,ymax]
    :return: bbox: [xmin,ymin,xmax,ymax]
    '''

    left = bbox[0]
    top = bbox[1]

    right = bbox[2]
    bottom = bbox[3]

    roi_width = right - left
    roi_height = bottom - top

    new_left = left - int(roi_width * enlarge_ratio)
    new_left = 0 if new_left < 0 else new_left

    new_top = top - int(roi_height * enlarge_ratio)
    new_top = 0 if new_top < 0 else new_top

    new_right = right + int(roi_width * enlarge_ratio)
    new_right = img_width if new_right > img_width else new_right

    new_bottom = bottom + int(roi_height * enlarge_ratio)
    new_bottom = img_height if new_bottom > img_height else new_bottom

    bbox = [new_left, new_top, new_right, new_bottom]

    bbox = [int(x) for x in bbox]

    return bbox


def _extract_square_crop(rgb_image, cx, cy, sz):
    """Extract a square crop centred at (cx, cy) with side sz; resize to INPUT_SIZE.
    Returns (crop, trans_x1, trans_y1, sz).
    """
    H, W = rgb_image.shape[:2]
    x1 = cx - sz / 2;  y1 = cy - sz / 2
    trans_x1, trans_y1 = x1, y1
    x2 = x1 + sz;      y2 = y1 + sz
    dx  = max(0, -x1);  x1 = max(0, x1)
    dy  = max(0, -y1);  y1 = max(0, y1)
    edx = max(0, x2 - W);  x2 = min(W, x2)
    edy = max(0, y2 - H);  y2 = min(H, y2)
    crop = rgb_image[int(y1):int(y2), int(x1):int(x2)]
    if dx > 0 or dy > 0 or edx > 0 or edy > 0:
        crop = cv2.copyMakeBorder(crop, int(dy), int(edy), int(dx), int(edx),
                                  cv2.BORDER_CONSTANT, value=(103.94, 116.78, 123.68))
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE)), trans_x1, trans_y1, sz


def _box_to_crop(rgb_image, x1_det, y1_det, x2_det, y2_det):
    """First-pass crop from a RetinaFace bounding box."""
    cx = (x1_det + x2_det) / 2
    cy = (y1_det + y2_det) / 2
    sz = max(x2_det - x1_det + 1, y2_det - y1_det + 1) * ENLARGE_RATIO
    return _extract_square_crop(rgb_image, cx, cy, sz)


def _landmarks_to_crop(rgb_image, affine_lmks):
    """Second-pass crop centred on the bounding box of first-pass landmarks."""
    x1, y1 = np.min(affine_lmks[:, 0]), np.min(affine_lmks[:, 1])
    x2, y2 = np.max(affine_lmks[:, 0]), np.max(affine_lmks[:, 1])
    cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
    sz = max(x2 - x1 + 1, y2 - y1 + 1) * ENLARGE_RATIO
    return _extract_square_crop(rgb_image, cx, cy, sz)


def _map_to_image(raw_flat, trans_x1, trans_y1, sz):
    """Map flat 212-element landmark output to (106, 2) image-space coordinates."""
    inv_scale = sz / INPUT_SIZE
    affine = np.zeros((106, 2))
    for idx in range(106):
        affine[idx, 0] = raw_flat[idx * 2]     * inv_scale + trans_x1
        affine[idx, 1] = raw_flat[idx * 2 + 1] * inv_scale + trans_y1
    return affine


class FaceInfo:
    def __init__(self):
        self.rect = np.asarray([0, 0, 0, 0])
        self.points_array = np.zeros((106, 2))
        self.eye_left = np.zeros((22, 2))
        self.eye_right = np.zeros((22, 2))
        self.eyebrow_left = np.zeros((13, 2))
        self.eyebrow_right = np.zeros((13, 2))
        self.lips = np.zeros((64, 2))


class LargeModelInfer:

    def __init__(self,ckpt,  device='cuda'):
        self.large_base_lmks_model = LargeBaseLmkInfer.model_preload(ckpt,  device.lower() == "cuda")
        self.device = device.lower()
        self.detector = Model(max_size=512, device=device)
        state_dict = torch.load(os.path.join(os.path.dirname(ckpt), 'retinaface_resnet50_2020-07-20_old_torch.pth' ) , map_location="cpu")
        # torch.save(state_dict, './models/retinaface_resnet50_2020-07-20_old_torch.pth', _use_new_zipfile_serialization=False)
        self.detector.load_state_dict(state_dict)
        self.detector.eval()

        
    def infer(self, img_bgr):
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results   = self.detector.predict_jsons(rgb_image)
        boxes = [{'x1': a['bbox'][0], 'y1': a['bbox'][1], 'x2': a['bbox'][2], 'y2': a['bbox'][3]}
                 for a in results if a['score'] != -1]
        if not boxes:
            return boxes, []

        use_gpu = self.device.lower() == 'cuda'

        # Pass 1: all face crops in one batch
        crops1  = [_box_to_crop(rgb_image, b['x1'], b['y1'], b['x2'], b['y2']) for b in boxes]
        lmks1   = LargeBaseLmkInfer.process_imgs_batch(
            self.large_base_lmks_model, [c[0] for c in crops1], use_gpu)
        affine1 = [_map_to_image(lmks1[n], crops1[n][1], crops1[n][2], crops1[n][3])
                   for n in range(len(boxes))]

        # Pass 2: refined crops in one batch
        crops2     = [_landmarks_to_crop(rgb_image, aff) for aff in affine1]
        lmks2      = LargeBaseLmkInfer.process_imgs_batch(
            self.large_base_lmks_model, [c[0] for c in crops2], use_gpu)
        landmarks  = [_map_to_image(lmks2[n], crops2[n][1], crops2[n][2], crops2[n][3])
                      for n in range(len(boxes))]

        return boxes, landmarks

    def infer_batch(self, imgs_bgr):
        """Run detection + 2-pass landmark inference on a list of BGR images.
        Returns list of (boxes, landmarks) — same per-image format as infer().
        """
        rgb_images  = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_bgr]
        all_results = self.detector.predict_jsons_batch(rgb_images)
        img_boxes   = [
            [{'x1': a['bbox'][0], 'y1': a['bbox'][1], 'x2': a['bbox'][2], 'y2': a['bbox'][3]}
             for a in res if a['score'] != -1]
            for res in all_results
        ]

        # Collect all pass-1 crops across every image
        crops1, crop_origin = [], []
        for img_idx, (rgb, boxes) in enumerate(zip(rgb_images, img_boxes)):
            for box_idx, b in enumerate(boxes):
                crops1.append(_box_to_crop(rgb, b['x1'], b['y1'], b['x2'], b['y2']))
                crop_origin.append((img_idx, box_idx))

        if not crops1:
            return [([], []) for _ in imgs_bgr]

        use_gpu = self.device.lower() == 'cuda'
        lmks1   = LargeBaseLmkInfer.process_imgs_batch(
            self.large_base_lmks_model, [c[0] for c in crops1], use_gpu)
        affine1 = [_map_to_image(lmks1[n], crops1[n][1], crops1[n][2], crops1[n][3])
                   for n in range(len(crops1))]

        # Collect all pass-2 crops
        crops2 = [_landmarks_to_crop(rgb_images[img_idx], affine1[n])
                  for n, (img_idx, _) in enumerate(crop_origin)]
        lmks2  = LargeBaseLmkInfer.process_imgs_batch(
            self.large_base_lmks_model, [c[0] for c in crops2], use_gpu)

        # Reconstruct per-image landmarks
        out_lmks = [[] for _ in imgs_bgr]
        for n, (img_idx, _) in enumerate(crop_origin):
            out_lmks[img_idx].append(
                _map_to_image(lmks2[n], crops2[n][1], crops2[n][2], crops2[n][3]))

        return [(img_boxes[i], out_lmks[i]) for i in range(len(imgs_bgr))]



    def find_face_contour(self, image):

        boxes, landmarks = self.infer(image)
        landmarks = np.array(landmarks)
        # print('boxes:{}'.format(boxes))
        canvas_channels = 9 

        args = [[0, 33, False], [33, 38, False], [42, 47, False], [51, 55, False], [57, 64, False], [66, 74, True],
                [75, 83, True], [84, 96, True]]

        roi_bboxs = []

        for i in range(len(boxes)):
            roi_bbox = enlarged_bbox([boxes[i]['x1'], boxes[i]['y1'], boxes[i]['x2'], boxes[i]['y2']],
                                     image.shape[1],
                                     image.shape[0], 0.5)
            # roi_bbox = track_box[i]
            roi_bbox = [int(x) for x in roi_bbox]
            roi_bboxs.append(roi_bbox)

        people_maps = []

        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :, :]
            maps = []
            whole_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            roi_box = roi_bboxs[i]
            roi_box_width = roi_box[2] - roi_box[0]
            roi_box_height = roi_box[3] - roi_box[1]
            short_side_length = roi_box_width if roi_box_width < roi_box_height else roi_box_height

            line_width = short_side_length // 10

            if line_width == 0:
                line_width = 1

            kernel_size = line_width * 2
            gaussian_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            for t, arg in enumerate(args):
                mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
                draw_line(mask, landmark[arg[0]:arg[1]], (255, 255, 255), line_width, arg[2])
                mask = cv2.GaussianBlur(mask, (gaussian_kernel, gaussian_kernel), 0)
                if t>=1:
                    draw_line(whole_mask, landmark[arg[0]:arg[1]], (255, 255, 255), line_width*2, arg[2])
                maps.append(mask)
            whole_mask = cv2.GaussianBlur(whole_mask, (gaussian_kernel, gaussian_kernel), 0)
            maps.append(whole_mask)
            people_maps.append(maps)

        return people_maps[0], boxes




    def face2contour(self, image, stack_mode="column"):
        '''

        :param facer:
        :param image:
        :param stack_mode:
        :return: final_maps: [map0, map1,....]
                 roi_bboxs: [bbox0, bbox1, ...]  bbox0的格式[xmin, ymin, xmax, ymax]
        '''

        boxes, landmarks = self.infer(image)
        landmarks = np.array(landmarks)
        # print('boxes:{}'.format(boxes))
        canvas_channels = 9 

        args = [[0, 33, False], [33, 38, False], [42, 47, False], [51, 55, False], [57, 64, False], [66, 74, True],
                [75, 83, True], [84, 96, True]]

        roi_bboxs = []

        for i in range(len(boxes)):
            roi_bbox = enlarged_bbox([boxes[i]['x1'], boxes[i]['y1'], boxes[i]['x2'], boxes[i]['y2']],
                                     image.shape[1],
                                     image.shape[0], 0.5)
            # roi_bbox = track_box[i]
            roi_bbox = [int(x) for x in roi_bbox]
            roi_bboxs.append(roi_bbox)

        people_maps = []

        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :, :]
            maps = []
            whole_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            roi_box = roi_bboxs[i]
            roi_box_width = roi_box[2] - roi_box[0]
            roi_box_height = roi_box[3] - roi_box[1]
            short_side_length = roi_box_width if roi_box_width < roi_box_height else roi_box_height

            line_width = short_side_length // 50

            if line_width == 0:
                line_width = 1

            kernel_size = line_width * 4
            gaussian_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            for arg in args:
                mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
                draw_line(mask, landmark[arg[0]:arg[1]], (255, 255, 255), line_width, arg[2])
                mask = cv2.GaussianBlur(mask, (gaussian_kernel, gaussian_kernel), 0)
                draw_line(whole_mask, landmark[arg[0]:arg[1]], (255, 255, 255), line_width, arg[2])
                maps.append(mask)
            whole_mask = cv2.GaussianBlur(whole_mask, (gaussian_kernel, gaussian_kernel), 0)
            maps.append(whole_mask)
            people_maps.append(maps)

        if stack_mode == "depth":
            final_maps = []
            for i, maps in enumerate(people_maps):
                final_map = np.dstack(maps)
                final_map = final_map[roi_bboxs[i][1]:roi_bboxs[i][3], roi_bboxs[i][0]:roi_bboxs[i][2], :]
                final_maps.append(final_map)
            return final_maps, roi_bboxs

        elif stack_mode == "column":
            final_maps = []
            for i, maps in enumerate(people_maps):
                joint_maps = [x[roi_bboxs[i][1]:roi_bboxs[i][3], roi_bboxs[i][0]:roi_bboxs[i][2]] for x in maps]
                final_map = np.column_stack(joint_maps)
                final_maps.append(final_map)
            return final_maps, roi_bboxs
        

    def fat_face(self,img, degree= 0.1):
        t1 = time.time()

        _img, scale = resize_on_long_side(img, 400)

        contour_maps, boxes = self.find_face_contour(_img)

        # print('|' * 50, 'time find_face_contour: {}'.format(time.time() - t1))
        # cv2.imwrite(f'all_joint.jpg', np.column_stack(contour_maps))

        contour_map = contour_maps[0]

        boxes = boxes[0]

        Flow = np.zeros(shape=(contour_map.shape[0], contour_map.shape[1], 2), dtype=np.float32)

        # cv2.rectangle(bgr,[boxes['x1'], boxes['y1']], [boxes['x2'], boxes['y2']],(0,0,255) )

        box_center = [(boxes['x1'] + boxes['x2']) / 2, (boxes['y1'] + boxes['y2']) / 2]

        box_length = max(abs(boxes['y1'] - boxes['y2']), abs(boxes['x1'] - boxes['x2']))

        flow_box_length = min(box_length * 2, 2 * (box_center[0] - 1), 2 * (box_center[1] - 1),
                              2 * (Flow.shape[0] - box_center[1] - 1), 2 * (Flow.shape[1] - box_center[0] - 1))
        flow_box_length = int(flow_box_length)

        # print('flow_box_length:{}'.format(flow_box_length))
        # print('box_center:{}'.format(box_center))
        t1 =time.time()

        sf = spread_flow(100, flow_box_length * degree)
        sf = cv2.resize(sf, (flow_box_length, flow_box_length))
        # print('|' * 50, 'time spread_flow: {}'.format(time.time() - t1))

        t1 = time.time()
        Flow[int(box_center[1] - flow_box_length / 2):int(box_center[1] + flow_box_length / 2),
        int(box_center[0] - flow_box_length / 2):int(box_center[0] + flow_box_length / 2)] = sf

        Flow = Flow * np.dstack((contour_map, contour_map)) / 255.0

        inter_face_maps = contour_maps[-1]

        Flow = Flow * (1.0 - np.dstack((inter_face_maps, inter_face_maps)) / 255.0)

        Flow = cv2.resize(Flow,(img.shape[1], img.shape[0]) )

        Flow = Flow /scale
        # print('|' * 50, 'time flow process: {}'.format(time.time() - t1))

        t1 = time.time()
        pred, top_bound, bottom_bound, left_bound, right_bound = image_warp_grid1(Flow[..., 0], Flow[..., 1], img, 1.0,
                                                                                  [0, 0, 0, 0])

        # print('|' * 50, 'time image_warp_grid1: {}'.format(time.time() - t1))
        return pred