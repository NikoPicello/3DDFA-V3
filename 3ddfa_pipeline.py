"""
3DDFA-V3 pipeline — process videos and save per-frame face detections.

Reads pre-extracted frame images by default, or video files directly with
--use_video. In image mode (default), each `data_path` under
resources/sessions/<sid>/<activity>/ is a folder named like the camera (e.g.
FC1/) holding that camera's frames as 000000.jpeg, 000001.jpeg, ... -- exactly
what ../../scripts/extract_frames.py produces. Frame index 0 in that folder
must correspond to frame 0 of the source video, since results are keyed by
frame index and downstream pipelines correlate across modalities on it.

Output format (saved as .pkl per video):
    dict keyed by frame_index, then by face/person id, one entry per detected face:
    {
        frame_index: {
            pid: {
                'ldm68':   np.ndarray (68,  2) — 68 landmarks in original image space (pixels),
                'ldm106':  np.ndarray (106, 2) — 106 landmarks in original image space,
            },
            ...
        },
        ...
    }

Note: the dense reconstructed mesh (v2d/v3d, 35709 vertices each) is intentionally
not saved — at video scale (thousands of frames) it dominates file size by ~700x
over the sparse landmarks while being fully reconstructible from the model's
256-float coefficient vector. Only the landmarks are kept.

Note: landmarks are mapped back from the internal 224×224 crop to the original
image pixel coordinates using `back_resize_ldms` from util/preprocess.py.
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

FRAME_BATCH = 128 # frames accumulated per recon_model forward pass; tune to GPU memory


def _cuda_driver_ready(attempts=3, delay=5.0):
    """Probe the CUDA driver directly, retrying a cold/contended init.

    Why not just retry torch.cuda.is_available(): torch caches the device count on the
    FIRST query (c10::cuda::device_count() memoises it), so once it has answered 0 no
    later call in the same process can recover — a retry loop around it is a no-op. The
    driver API is not memoised, so probe libcuda BEFORE torch looks.

    This matters here because nvidia-persistenced is not running and persistence mode is
    off: the driver tears down per-GPU state when no client holds a device, so the first
    client onto a cold GPU pays a full init, which under load can take >15s or fail
    outright (observed: cuInit -> CUDA_ERROR_NOT_INITIALIZED after 16.4s). Retrying gives
    a colliding batch of jobs a chance to serialise instead of all falling back to CPU.
    """
    import ctypes
    try:
        lib = ctypes.CDLL('libcuda.so.1')
    except OSError as e:
        print(f"[cuda] libcuda.so.1 not loadable: {e}", flush=True)
        return False
    for i in range(attempts):
        rc = lib.cuInit(0)
        if rc == 0:
            return True
        if i < attempts - 1:
            print(f"[cuda] cuInit failed (CUDA error {rc}); retry {i + 1}/{attempts - 1} "
                  f"in {delay}s", flush=True)
            time.sleep(delay)
    print(f"[cuda] cuInit still failing (CUDA error {rc}) after {attempts} attempts", flush=True)
    return False


def select_device():
    """'cuda' when a GPU was requested and is usable — otherwise abort rather than
    silently running on CPU. A CPU fallback here is never what the caller wanted: it is
    ~100x slower, it looks like a healthy run (no GPU process in nvidia-smi/nvitop, so it
    is easy to miss for hours), and it burns ~7 cores per job on the shared node."""
    want = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    gpu_requested = want not in ('', '-1')
    if not gpu_requested:
        print("[cuda] CUDA_VISIBLE_DEVICES unset -> running on CPU deliberately", flush=True)
        return 'cpu'
    if not _cuda_driver_ready() or not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={want} was requested but the CUDA driver is "
            f"unusable, so this job would silently run on CPU (~100x slower). "
            f"Aborting instead.\n"
            f"  Most likely: persistence mode is off (nvidia-persistenced not running), "
            f"so a cold GPU's driver init fails under load.\n"
            f"  Ask an admin for `nvidia-smi -pm 1`, and/or raise --launch-stagger in "
            f"run_parallel_sessions.py so jobs don't all cold-init at once.\n"
            f"  To run on CPU on purpose, launch with CUDA_VISIBLE_DEVICES=''.")
    print(f"[cuda] driver ready; using GPU (CUDA_VISIBLE_DEVICES={want})", flush=True)
    return 'cuda'


def build_args(device='cuda'):
    """Build a minimal args namespace that face_model and face_box expect."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',       default=device)
    parser.add_argument('--iscrop',       default=True,         type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--detector',     default='retinaface')
    parser.add_argument('--ldm68',        default=True,         type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--ldm106',       default=True,         type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--ldm106_2d',    default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--ldm134',       default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--seg',          default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--seg_visible',  default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--useTex',       default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--extractTex',   default=False,        type=lambda x: x.lower() in ['true','1'])
    parser.add_argument('--batch_size',   default=8,            type=int)
    parser.add_argument('--sid',          default=None,         type=str)
    parser.add_argument('--activities',   default=['animals_task', 'gaze_task', 'ghost_task', 'lego_task', 'talk_task'], nargs='+')
    parser.add_argument('--use_video',    action='store_true')
    parser.add_argument('--max-frames',   default=-1,           type=int)
    parser.add_argument('--backbone',     default='resnet50')
    parser.add_argument('--inputpath',    default='')
    parser.add_argument('--savepath',     default='')
    return parser.parse_args()


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

    device = select_device()
    args   = build_args(device)

    # ── load models (once) ──────────────────────────────────────────────────
    recon_model      = face_model(args)
    fb               = face_box(args)
    facebox_detector = fb.detector_batch

    for sid_path in sid_paths:
        session_id = Path(sid_path).stem
        if args.sid is not None and args.sid not in session_id: continue

        for activity in args.activities:
            print(f'[3DDFA] {activity} — {session_id}')
            if args.use_video:
                # *.mp4 only -- an image-mode frame folder sitting alongside the
                # videos (same activity dir) must not be picked up as a video path
                data_paths = glob.glob(os.path.join(sid_path, activity, '*.mp4'))
                data_paths = [v for v in data_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
            else:
                # one sub-folder per camera, each holding that camera's frames
                data_paths = [p for p in glob.glob(os.path.join(sid_path, activity) + '/*') if os.path.isdir(p)]

            for data_path in data_paths:
                video_name = Path(data_path).stem

                if args.use_video:
                    cap = cv.VideoCapture(data_path)
                    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                else:
                    image_paths = sorted(glob.glob(os.path.join(data_path, '*.jpeg')))
                    total_frames = len(image_paths)
                if args.max_frames >= 0:
                    total_frames = min(total_frames, args.max_frames)

                curr_out_path = os.path.join(out_path, session_id, activity)
                os.makedirs(curr_out_path, exist_ok=True)
                out_pkl = os.path.join(curr_out_path, f'{video_name}_3ddfa.pkl')

                frame_results = {}
                frames_buf    = []  # PIL images
                fidxs_buf     = []  # original frame indices

                def flush():
                    if not frames_buf:
                        return

                    # ── batch face detection across all buffered frames ───────
                    det_results = facebox_detector(frames_buf)

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
                        batch_tensor = torch.cat(samples, dim=0).to(args.device)
                        recon_model.input_img = batch_tensor
                        with torch.no_grad():
                            results = recon_model.forward()

                        for n in range(len(samples)):
                            fidx         = sample_fidx[n]
                            pid          = sample_pid[n]
                            trans_params = sample_trans[n]

                            ldm68  = back_resize_pts(results['ldm68'][n],  trans_params)
                            ldm106 = back_resize_pts(results['ldm106'][n], trans_params)

                            if fidx not in frame_results:
                                frame_results[fidx] = {}
                            frame_results[fidx][pid] = {
                                'ldm68':  ldm68.astype(np.float32),
                                'ldm106': ldm106.astype(np.float32),
                            }

                    frames_buf.clear()
                    fidxs_buf.clear()

                t_video_start = time.perf_counter()
                for fidx in trange(total_frames, desc=video_name):
                    if args.use_video:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        # frame_bgr = cv.resize(frame_bgr, (1280, 720))
                    else:
                        frame_bgr = cv.imread(image_paths[fidx])
                        if frame_bgr is None:
                            print(f'  [warn] unreadable frame, skipping: {image_paths[fidx]}')
                            continue
                    frames_buf.append(Image.fromarray(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)))
                    fidxs_buf.append(fidx)
                    if len(frames_buf) == FRAME_BATCH:
                        flush()

                flush()  # process any remaining frames
                t_video_total = time.perf_counter() - t_video_start

                n_frames = fidx + 1 if 'fidx' in dir() else total_frames
                fps = n_frames / t_video_total if t_video_total > 0 else 0.0
                print(f'  {n_frames} frames in {t_video_total:.1f}s ({fps:.1f} fps)')

                if args.use_video:
                    cap.release()

                with open(out_pkl, 'wb') as f:
                    pickle.dump(frame_results, f)
                print(f'  Saved {len(frame_results)} face detections → {out_pkl}')


if __name__ == '__main__':
    main()
    print('=== 3DDFA pipeline done')
