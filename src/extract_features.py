#!/usr/bin/env python3
"""Optional: extract the 4-D turbulence features from a directory of videos.

Examples
--------
Farneback (no GPU):
  python src/extract_features.py --videos /data/crafter --label 0 \
      --generator Crafter --backend farneback --out crafter.csv

RAFT (needs RAFT code + raft-things.pth):
  python src/extract_features.py --videos /data/crafter --label 0 \
      --generator Crafter --backend raft \
      --raft-root /path/to/RAFT --weights /path/to/raft-things.pth --out crafter.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from feature_utils import (
    compute_tf_features,
    extract_flow_farneback,
    load_frames_random,
)


def build_raft(root, weights, device):
    import torch
    from argparse import Namespace

    sys.path.append(os.path.join(root, "core"))
    from raft import RAFT
    from utils.utils import InputPadder

    args = Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        corr_levels=4,
        corr_radius=4,
        dropout=0,
        num_heads=1,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.module.to(device).eval(), InputPadder


def get_raft_flow(frames, model, device, InputPadder):
    import torch

    flows = []
    for i in range(len(frames) - 1):
        img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).to(device).float()
        img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).unsqueeze(0).to(device).float()
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad():
            out = model(img1, img2, iters=20, test_mode=True)
        flow_up = out[-1] if isinstance(out, (tuple, list)) else out
        flows.append(padder.unpad(flow_up).squeeze().permute(1, 2, 0).cpu().numpy())
    return np.stack(flows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--label", type=int, choices=[0, 1], required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--backend", choices=["raft", "farneback"], default="farneback")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--raft-root", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.videos, "*.mp4")))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no .mp4 files under {args.videos}")

    if args.backend == "raft":
        import torch

        if not args.raft_root or not args.weights:
            raise SystemExit("--raft-root and --weights are required for RAFT")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, InputPadder = build_raft(args.raft_root, args.weights, device)

    rows = []
    for n, path in enumerate(paths, 1):
        frames = load_frames_random(
            path, max_frames=args.frames, seed=args.seed
        )
        if len(frames) < 2:
            print(f"skip (too short): {path}")
            continue
        if args.backend == "raft":
            flow = get_raft_flow(frames, model, device, InputPadder)
        else:
            flow = extract_flow_farneback(frames)
        feats = compute_tf_features(flow, seed=args.seed)
        rows.append(
            {
                "file": os.path.basename(path),
                "F_low": feats[0],
                "omega_max": feats[1],
                "max_temporal": feats[2],
                "p999_temporal": feats[3],
                "label": args.label,
                "generator": args.generator,
            }
        )
        print(f"[{n}/{len(paths)}] {os.path.basename(path)}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
