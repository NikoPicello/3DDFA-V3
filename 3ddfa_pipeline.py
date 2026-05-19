"""
3DDFA-V3 pipeline — process videos and save per-frame face detections.

Output format (saved as .pkl per video):
    List of dicts, one entry per detected face per frame:
    {
        'frame_index': int,
        'ldm68':   np.ndarray (68,  2) — 68 landmarks in original image space (pixels),
        'ldm106':  np.ndarray (106, 2) — 106 landmarks in original image space,
        'v2d':     np.ndarray (35709, 2) — BFM mesh vertices projected to image space,
        'v3d':     np.ndarray (35709, 3) — BFM mesh vertices in camera space,
        'tri':     np.ndarray (70789, 3) — triangle faces (shared across frames),
    }

Note: landmarks and v2d are mapped back from the internal 224×224 crop to the
original image pixel coordinates using `back_resize_ldms` from util/preprocess.py.
"""

import os
import sys
import glob
import pickle
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image
import torch
from tqdm import trange

# ── path setup: add 3DDFA-V3 root so internal imports resolve ───────────────
_3DDFA_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _3DDFA_ROOT)

from face_box import face_box
from model.recon import face_model
from util.preprocess import back_resize_ldms


cam_map = {
    'GC': 'GB', 'HC': 'GF',
    'Z1': 'FC1', 'Z2': 'FC2',
    'N1': 'HA1', 'N2': 'HA2',
}

activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']
FRAME_BATCH = 8  # frames accumulated per recon_model forward pass; tune to GPU memory

def build_args(device='cuda'):
    """Build a minimal args namespace that face_model and face_box expect."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',      default=device)
    parser.add_argument('--iscrop',      default=True,         type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--detector',    default='retinaface')
    parser.add_argument('--ldm68',       default=True,         type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--ldm106',      default=True,         type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--ldm106_2d',   default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--ldm134',      default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--seg',         default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--seg_visible', default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--useTex',      default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--extractTex',  default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--backbone',    default='resnet50')
    parser.add_argument('--inputpath',   default='')
    parser.add_argument('--savepath',    default='')
    return parser.parse_args([])


def back_resize_pts(pts, trans_params):
    """Map (N, 2) points from 3DDFA's 224×224 crop space to original image space.

    3DDFA's to_image() uses the BFM coordinate system where y=0 is at the
    bottom of the crop (math convention / y-up).  back_resize_ldms expects
    y=0 at the top (image convention / y-down), so we flip y first.
    """
    ldms = pts.copy().astype(np.float64)
    ldms[:, 1] = 224.0 - ldms[:, 1]   # y-up → y-down in crop space
    return back_resize_ldms(ldms, trans_params)


def main():
    main_path     = '/'.join(sys.path[0].split('/')[:-2]) + '/'
    resources_path = os.path.join(main_path, 'resources')
    sessions_path  = os.path.join(resources_path, 'sessions')
    out_path       = os.path.join(resources_path, '3ddfa_results')
    sid_paths      = sorted(glob.glob(sessions_path + '/*'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args   = build_args(device)

    # ── load models (once) ──────────────────────────────────────────────────
    recon_model      = face_model(args)
    facebox_detector = face_box(args).detector

    for sid_path in sid_paths:
        session_id = Path(sid_path).stem
        if '005013' not in session_id: continue

        for activity in activities:
            print(f'[3DDFA] {activity} — {session_id}')
            vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
            vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]

            for vid_path in vid_paths:
                video_name = Path(vid_path).stem

                cap = cv.VideoCapture(vid_path)
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                # total_frames = 25

                curr_out_path = os.path.join(out_path, session_id, activity)
                os.makedirs(curr_out_path, exist_ok=True)
                out_pkl = os.path.join(curr_out_path, f'{video_name}_3ddfa.pkl')

                frame_results  = {}
                samples        = []  # im_tensor (1, 3, 224, 224) per face
                sample_fidx    = []  # original frame index per sample
                sample_pid     = []  # person id per sample
                sample_trans   = []  # trans_params per sample
                frames_seen    = 0

                def flush():
                    if not samples:
                        return

                    # ── single recon_model forward pass for all accumulated faces ──
                    batch_tensor = torch.cat(samples, dim=0).to(args.device)  # (N, 3, 224, 224)
                    recon_model.input_img = batch_tensor
                    with torch.no_grad():
                        results = recon_model.forward()

                    tri = results['tri']  # constant (70789, 3)

                    for n in range(len(samples)):
                        fidx         = sample_fidx[n]
                        pid          = sample_pid[n]
                        trans_params = sample_trans[n]

                        ldm68  = back_resize_pts(results['ldm68'][n],  trans_params)
                        ldm106 = back_resize_pts(results['ldm106'][n], trans_params)
                        v2d    = back_resize_pts(results['v2d'][n],    trans_params)
                        v3d    = results['v3d'][n]

                        if fidx not in frame_results:
                            frame_results[fidx] = {}
                        frame_results[fidx][pid] = {
                            'ldm68':  ldm68.astype(np.float32),
                            'ldm106': ldm106.astype(np.float32),
                            'v2d':    v2d.astype(np.float32),
                            'v3d':    v3d.astype(np.float32),
                            'tri':    tri,
                        }

                    samples.clear()
                    sample_fidx.clear()
                    sample_pid.clear()
                    sample_trans.clear()

                for fidx in trange(total_frames, desc=video_name):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_bgr = cv.resize(frame_bgr, (1280, 720))
                    frame_pil = Image.fromarray(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))

                    try:
                        trans_results, im_results = facebox_detector(frame_pil)
                    except Exception as e:
                        print(f'  Frame {fidx}: face detection failed — {e}')
                        frames_seen += 1
                        if frames_seen % FRAME_BATCH == 0:
                            flush()
                        continue
                    if trans_results is None:
                        frames_seen += 1
                        if frames_seen % FRAME_BATCH == 0:
                            flush()
                        continue

                    for pid in trans_results.keys():
                        samples.append(im_results[pid])
                        sample_fidx.append(fidx)
                        sample_pid.append(pid)
                        sample_trans.append(trans_results[pid])

                    frames_seen += 1
                    if frames_seen % FRAME_BATCH == 0:
                        flush()

                flush()  # process any remaining samples

                cap.release()

                with open(out_pkl, 'wb') as f:
                    pickle.dump(frame_results, f)
                print(f'  Saved {len(frame_results)} face detections → {out_pkl}')


if __name__ == '__main__':
    main()
    print('=== 3DDFA pipeline done')
