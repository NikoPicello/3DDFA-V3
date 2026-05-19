import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
import os
import cv2
from mtcnn import MTCNN
from util.preprocess import load_lm3d, align_img
from .facelandmark.large_model_infer import LargeModelInfer

def no_crop(im):
    if np.array(im).shape==(224,224,3):
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return None, im
    else:
        print('the original face image should be well cropped and resized to (224,224,3).')
        return None, None

class retinaface:
    def __init__(self, device):
        self.landmark_model = LargeModelInfer("assets/large_base_net.pth", device=device)
        self.lm3d_std = load_lm3d()

    def detector(self, im):
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        H = img.shape[0]
        _, results_all = self.landmark_model.infer(img)
        if not results_all:
            return None, None

        idx_range = [0] if len(results_all) == 1 else [0, 1]
        trans_results, im_results = {}, {}
        for res_idx in idx_range:
            lmks_106 = results_all[res_idx]
            landmarks = np.array([[lmks_106[idx][0], lmks_106[idx][1]]
                                   for idx in [74, 83, 54, 84, 90]], dtype=np.float32)
            landmarks[:, -1] = H - 1 - landmarks[:, -1]
            trans_params, im_aligned, _, _ = align_img(im, landmarks, self.lm3d_std)
            im_tensor = torch.tensor(np.array(im_aligned)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            trans_results[res_idx] = trans_params
            im_results[res_idx]    = im_tensor
        return trans_results, im_results

    def detector_batch(self, images):
        """Run detection on a list of PIL images.
        Returns a list of (trans_results, im_results) — same format as detector(),
        with (None, None) for frames where no face was detected.
        """
        imgs_bgr  = [cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR) for im in images]
        heights   = [img.shape[0] for img in imgs_bgr]
        batch_out = self.landmark_model.infer_batch(imgs_bgr)

        results = []
        for (boxes, all_landmarks), im_orig, H in zip(batch_out, images, heights):
            if not all_landmarks:
                results.append((None, None))
                continue

            idx_range = [0] if len(all_landmarks) == 1 else [0, 1]
            trans_results, im_results = {}, {}
            for res_idx in idx_range:
                lmks_106  = all_landmarks[res_idx]
                landmarks = np.array([[lmks_106[idx][0], lmks_106[idx][1]]
                                       for idx in [74, 83, 54, 84, 90]], dtype=np.float32)
                landmarks[:, 1] = H - 1 - landmarks[:, 1]
                trans_params, im_aligned, _, _ = align_img(im_orig, landmarks, self.lm3d_std)
                im_tensor = torch.tensor(np.array(im_aligned) / 255.,
                                          dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                trans_results[res_idx] = trans_params
                im_results[res_idx]    = im_tensor
            results.append((trans_results, im_results))

        return results


class mtcnnface:
    def __init__(self):
        self.landmark_model = MTCNN()
        self.lm3d_std = load_lm3d()

    def detector(self, im):
        img = np.asarray(im)
        H = img.shape[0]
        facial_landmarks = self.landmark_model.detect_faces(img)

        if len(facial_landmarks) > 0:
            highest = max(facial_landmarks, key=lambda x: x['confidence'])
            if highest['confidence'] > 0.6:
                landmarks = np.array([[v[0], v[1]] for v in highest['keypoints'].values()],
                                      dtype=np.float32)
                trans_params, im, _, _ = align_img(im, landmarks, self.lm3d_std)
                im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                return trans_params, im

        return None, None


class face_box:
    def __init__(self, args):
        if args.iscrop:
            if args.detector == 'mtcnn':
                m = mtcnnface()
                self.detector = m.detector
                print('use mtcnn for face box')
            elif args.detector == 'retinaface':
                r = retinaface(args.device)
                self.detector       = r.detector
                self.detector_batch = r.detector_batch
                print('use retinaface for face box')
            else:
                raise ValueError(f'Unknown detector: {args.detector}')
        else:
            print('run original image in (224,224,3) size')
            self.detector = no_crop
