"""raw16 <-> [0,1]/[-1,1] normalization and lens-distortion correction.

Trimmed down from thermal_restore/geometry/trnerf_io.py and
thermal_restore/nerf/distortion.py to just what train.py/infer.py need --
this pipeline never touches poses, IMU, or the stereo cameras, so all of
that plumbing (load_pose_sequence, transform_between_cams, ...) is left out
on purpose.
"""
import cv2
import numpy as np
import yaml


def normalize_16bit_to_unit(img16, low_pct=1.0, high_pct=99.0):
    """Percentile-stretch raw thermal counts into [-1, 1] float32."""
    lo, hi = np.percentile(img16, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img16.astype(np.float32) - lo) / (hi - lo), 0, 1) * 2 - 1


def match_timestamps(pose_ts, frame_ts, max_gap_ns=8_000_000):
    """For each pose timestamp, nearest frame index (or -1 if beyond max_gap_ns).
    Only needed for held-out evaluation against pseudo-GT sequences (see
    verify_pipeline.py) -- not needed for live inference."""
    frame_ts = np.asarray(frame_ts)
    idx = np.searchsorted(frame_ts, pose_ts)
    idx = np.clip(idx, 1, len(frame_ts) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(pose_ts - frame_ts[left]) <= np.abs(pose_ts - frame_ts[right])
    nearest = np.where(use_left, left, right)
    gap = np.abs(pose_ts - frame_ts[nearest])
    nearest[gap > max_gap_ns] = -1
    return nearest


# --- lens distortion (camchain.yaml, radtan model) ---

def load_camchain(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def cam_intrinsics_matrix(camchain, cam_name):
    fx, fy, cx, cy = camchain[cam_name]['intrinsics']
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])


def cam_distortion(camchain, cam_name):
    return np.array(camchain[cam_name]['distortion_coeffs'])


def build_undistort_remap(K, dist_coeffs, width, height):
    """(map1, map2) for cv2.remap -- undistorts an IMAGE into the ideal-
    pinhole grid the same K describes (newCameraMatrix=K, so the output
    stays in the same pixel grid as the input). Applied to the model's
    OUTPUT, never its input (see infer.py's RealtimeRestorer docstring)."""
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (width, height), cv2.CV_32FC1)
    return map1, map2
