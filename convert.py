"""
preprocess_dataset.py
Run this ONCE before training.
Converts all MP4 videos to .npz files with 20 pre-extracted frames.
After this, training DataLoader reads .npz directly — no video decoding overhead.
"""

import os
import cv2
import numpy as np

DATASET_DIR = "dataset2/samples"
NPZ_DIR     = os.path.join(DATASET_DIR, "npz_samples")
NUM_FRAMES  = 20
IMG_SIZE    = 64

os.makedirs(NPZ_DIR, exist_ok=True)

samples_dir = os.path.join(DATASET_DIR)
video_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".mp4")])

print(f"Preprocessing {len(video_files)} videos...")

for i, fname in enumerate(video_files):
    sample_id = fname.replace(".mp4", "")
    npz_path  = os.path.join(NPZ_DIR, f"{sample_id}.npz")

    # Skip if already processed
    if os.path.exists(npz_path):
        continue

    video_path = os.path.join(samples_dir, fname)
    cap = cv2.VideoCapture(video_path)

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        all_frames.append(frame_resized)
    cap.release()

    if len(all_frames) == 0:
        print(f"[WARN] Empty video: {fname}")
        continue

    # Sample 20 evenly spaced frames
    total = len(all_frames)
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = np.stack([all_frames[idx] for idx in indices])  # (20, 64, 64, 3) uint8

    np.savez_compressed(npz_path, frames=frames)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(video_files)} done")

print("Preprocessing complete.")
print(f"NPZ files saved to: {NPZ_DIR}")