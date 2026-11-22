#!/usr/bin/env python3
"""Optical-flow preprocessing and the four turbulence features.

The functions here mirror the pipeline used to produce data/features_4d.csv:
  RAFT/Farneback flow -> RANSAC global-motion removal -> Helmholtz solenoidal
  projection -> F_low, omega_max, max_temporal, p999_temporal.
"""
import numpy as np
import cv2


def load_frames_random(video_path, max_frames=20, target=(256, 256), seed=42):
    """Read a video and return ``max_frames`` randomly sampled frames (BGR)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, target))
    cap.release()
    if len(frames) > max_frames:
        idx = np.sort(
            np.random.RandomState(seed).choice(len(frames), max_frames, replace=False)
        )
        frames = [frames[i] for i in idx]
    return frames


def remove_global_motion(flow_seq, max_features=500):
    """Remove a RANSAC homography from a (T, H, W, 2) optical-flow sequence."""
    T, H, W, _ = flow_seq.shape
    residual = np.zeros_like(flow_seq)
    n = int(np.sqrt(max_features))
    ys = np.linspace(5, H - 5, n).astype(int)
    xs = np.linspace(5, W - 5, n).astype(int)
    base = np.array([[x, y] for y in ys for x in xs], dtype=np.int32)
    base_f = base.astype(np.float32)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    allp = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)

    for t in range(T):
        u, v = flow_seq[t, :, :, 0], flow_seq[t, :, :, 1]
        pts2 = base_f + np.stack(
            [u[base[:, 1], base[:, 0]], v[base[:, 1], base[:, 0]]], 1
        )
        H_mat, _ = cv2.findHomography(base_f, pts2, cv2.RANSAC, 3.0)
        if H_mat is None:
            residual[t] = flow_seq[t]
            continue
        warped = cv2.perspectiveTransform(allp.reshape(-1, 1, 2), H_mat).reshape(-1, 2)
        residual[t, :, :, 0] = u - (warped[:, 0] - allp[:, 0]).reshape(H, W)
        residual[t, :, :, 1] = v - (warped[:, 1] - allp[:, 1]).reshape(H, W)
    return residual


def helmholtz_solenoidal(u, v):
    """Return the divergence-free (solenoidal) component of a 2-D flow field."""
    H, W = u.shape
    U = np.fft.fft2(u)
    V = np.fft.fft2(v)
    kx = np.fft.fftfreq(W) * 2 * np.pi
    ky = np.fft.fftfreq(H) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1
    curl_k = 1j * (KX * V - KY * U)
    psi_k = -curl_k / K2
    return (
        np.real(np.fft.ifft2(-1j * KY * psi_k)),
        np.real(np.fft.ifft2(1j * KX * psi_k)),
    )


def compute_tf_features(flow, seed=42, remove_global=True):
    """Compute [F_low, omega_max, max_temporal, p999_temporal] from a flow sequence.

    If ``remove_global`` is False, the input is assumed to be already
    global-motion compensated; this is useful when testing scale equivariance
    without the fixed-pixel RANSAC threshold changing the residual.
    """
    T, H, W, _ = flow.shape
    if remove_global:
        flow = remove_global_motion(flow)
    sol = np.zeros_like(flow)
    for t in range(T):
        sol[t, :, :, 0], sol[t, :, :, 1] = helmholtz_solenoidal(
            flow[t, :, :, 0], flow[t, :, :, 1]
        )
    if T > 20:
        keep = np.sort(np.random.RandomState(seed).choice(T, 20, replace=False))
        sol = sol[keep]
    T = sol.shape[0]

    sp2, sp4 = [], []
    for ell in (1, 2, 3):
        d = np.concatenate(
            [
                (sol[:, :, ell:, 0] - sol[:, :, :-ell, 0]).ravel(),
                (sol[:, :, ell:, 1] - sol[:, :, :-ell, 1]).ravel(),
            ]
        )
        sp2.append(np.mean(d ** 2))
        sp4.append(np.mean(d ** 4))
    f_low = float(np.mean([s4 / (s2 ** 2 + 1e-30) for s2, s4 in zip(sp2, sp4)]))

    omega = [
        np.abs(
            np.gradient(sol[t, :, :, 1], axis=1)
            - np.gradient(sol[t, :, :, 0], axis=0)
        ).max()
        for t in range(T)
    ]
    omega_max = float(max(omega))

    dudt = sol[1:] - sol[:-1]
    mag = np.sqrt(dudt[:, :, :, 0] ** 2 + dudt[:, :, :, 1] ** 2)
    return np.array(
        [f_low, omega_max, float(mag.max()), float(np.percentile(mag, 99.9))],
        dtype=float,
    )


def extract_flow_farneback(frames):
    """Dense flow for consecutive frames using OpenCV's polynomial expansion."""
    flows = []
    for i in range(len(frames) - 1):
        f0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        f1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flows.append(
            cv2.calcOpticalFlowFarneback(
                f0, f1, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        )
    return np.stack(flows)
