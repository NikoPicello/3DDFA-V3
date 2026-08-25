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
    }

Note: triangle connectivity ('tri') is not included — it's a constant shared by
every frame/video, defined in assets/face_model.npy, so it isn't duplicated here.
Mesh rasterization (extractTex=False, seg_visible=False) is also skipped entirely
in model/recon.py, since neither render_shape/render_face/face_texture nor tri
are consumed by this pipeline.

Note: landmarks and v2d are mapped back from the internal 224×224 crop to the
original image pixel coordinates using `back_resize_ldms` from util/preprocess.py.
"""

import os
import sys
import glob
import pickle
import argparse
import time
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
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--sid', type=str, default=None,
                             help="Specify a session to process")
    cli_args = cli_parser.parse_args()

    main_path     = '/'.join(sys.path[0].split('/')[:-2]) + '/'
    resources_path = os.path.join(main_path, 'resources')
    sessions_path  = os.path.join(resources_path, 'sessions')
    out_path       = os.path.join(resources_path, '3ddfa_results')
    sid_paths      = sorted(glob.glob(sessions_path + '/*'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args   = build_args(device)

    # ── load models (once) ──────────────────────────────────────────────────
    recon_model      = face_model(args)
    fb               = face_box(args)
    facebox_detector = fb.detector_batch

    for sid_path in sid_paths:
        session_id = Path(sid_path).stem
        if cli_args.sid is not None and cli_args.sid not in session_id: continue

        for activity in activities:
            print(f'[3DDFA] {activity} — {session_id}')
            vid_paths = glob.glob(os.path.join(sid_path, activity) + '/*')
            vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]

            for vid_path in vid_paths:
                video_name = Path(vid_path).stem

                cap = cv.VideoCapture(vid_path)
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

                curr_out_path = os.path.join(out_path, session_id, activity)
                os.makedirs(curr_out_path, exist_ok=True)
                out_pkl = os.path.join(curr_out_path, f'{video_name}_3ddfa.pkl')

                frame_results = {}
                frames_buf    = []  # PIL images
                fidxs_buf     = []  # original frame indices

                t_read = t_detect = t_recon = t_postproc = 0.0

                def flush():
                    nonlocal t_detect, t_recon, t_postproc
                    if not frames_buf:
                        return

                    # ── batch face detection across all buffered frames ───────
                    _t0 = time.perf_counter()
                    det_results = facebox_detector(frames_buf)
                    t_detect += time.perf_counter() - _t0

                    # collect face crops from all frames into one sample list
                    samples      = []
                    sample_fidx  = []
                    sample_pid   = []
                    sample_trans = []
                    for fidx, (trans_results, im_results) in zip(fidxs_buf, det_results):
                        if trans_results is None:
                            continue
                        for pid in trans_results.keys():
                            samples.append(im_results[pid])
                            sample_fidx.append(fidx)
                            sample_pid.append(pid)
                            sample_trans.append(trans_results[pid])

                    if samples:
                        # ── single recon_model forward pass for all faces ─────
                        _t0 = time.perf_counter()
                        batch_tensor = torch.cat(samples, dim=0).to(args.device)
                        recon_model.input_img = batch_tensor
                        with torch.no_grad():
                            results = recon_model.forward()
                        t_recon += time.perf_counter() - _t0

                        _t0 = time.perf_counter()
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
                            }
                        t_postproc += time.perf_counter() - _t0

                    frames_buf.clear()
                    fidxs_buf.clear()

                def log_timing(n_frames):
                    elapsed = time.perf_counter() - t_video_start
                    t_other = elapsed - t_read - t_detect - t_recon - t_postproc
                    print(f'  [{n_frames} frames | {elapsed:.1f}s elapsed]'
                          f'  read {t_read:.1f}s ({100*t_read/elapsed:.0f}%)'
                          f'  detect {t_detect:.1f}s ({100*t_detect/elapsed:.0f}%)'
                          f'  recon {t_recon:.1f}s ({100*t_recon/elapsed:.0f}%)'
                          f'  post {t_postproc:.1f}s ({100*t_postproc/elapsed:.0f}%)'
                          f'  other {t_other:.1f}s ({100*t_other/elapsed:.0f}%)')

                t_video_start = time.perf_counter()
                for fidx in trange(total_frames, desc=video_name):
                    _t0 = time.perf_counter()
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    # frame_bgr = cv.resize(frame_bgr, (1280, 720))
                    frames_buf.append(Image.fromarray(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)))
                    fidxs_buf.append(fidx)
                    t_read += time.perf_counter() - _t0
                    if len(frames_buf) == FRAME_BATCH:
                        flush()
                    if (fidx + 1) % 500 == 0:
                        log_timing(fidx + 1)

                flush()  # process any remaining frames
                t_video_total = time.perf_counter() - t_video_start

                t_other = t_video_total - t_read - t_detect - t_recon - t_postproc
                n_frames = fidx + 1 if 'fidx' in dir() else total_frames
                print(f'  Timing over {n_frames} frames (total {t_video_total:.1f}s):')
                print(f'    read/decode : {t_read:.2f}s  ({100*t_read/t_video_total:.1f}%)')
                print(f'    face detect : {t_detect:.2f}s  ({100*t_detect/t_video_total:.1f}%)')
                print(f'    recon fwd   : {t_recon:.2f}s  ({100*t_recon/t_video_total:.1f}%)')
                print(f'    post-proc   : {t_postproc:.2f}s  ({100*t_postproc/t_video_total:.1f}%)')
                print(f'    other       : {t_other:.2f}s  ({100*t_other/t_video_total:.1f}%)')

                cap.release()

                with open(out_pkl, 'wb') as f:
                    pickle.dump(frame_results, f)
                print(f'  Saved {len(frame_results)} face detections → {out_pkl}')


if __name__ == '__main__':
    main()
    print('=== 3DDFA pipeline done')
