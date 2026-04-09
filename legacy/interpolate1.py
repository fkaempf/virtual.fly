import os, re, glob, shutil
import cv2
import numpy as np

# --------- paths ---------
SRC_FOLDER   = "/Users/fkampf/Downloads/aaa"   # change me
RENAMED_DIR  = "renamed_000_360"
INTERP_DIR   = "interp_3600"

# --------- helpers ---------
def find_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return paths

def extract_angle(path):
    # grab the last signed integer in the filename (covers fly-180 and fly180)
    base = os.path.basename(path)
    m = re.findall(r"[-+]?\d+", base)
    if not m:
        raise ValueError(f"No angle found in filename: {base}")
    return int(m[-1])

def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read: {path}")
    return img

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# --------- 1) collect and sort originals by angle ---------
src_paths = find_images(SRC_FOLDER)
if not src_paths:
    raise FileNotFoundError(f"No images found in {SRC_FOLDER}")

by_angle = []
for p in src_paths:
    try:
        ang = extract_angle(p)
        by_angle.append((ang, p))
    except Exception:
        pass
if not by_angle:
    raise RuntimeError("Could not parse any angles from filenames.")

by_angle.sort(key=lambda x: x[0])  # ascending angle

angles = [a for a,_ in by_angle]
print("Found angles min..max:", angles[0], "..", angles[-1], "count:", len(angles))

# Basic sanity: expect -180..180 inclusive -> 361 images
if angles[0] == -180 and angles[-1] == 180 and len(angles) == 361:
    pass  # looks right
else:
    print("Warning: expected -180..180 with 361 images. Proceeding anyway.")

# --------- 2) rename to 0..360 and normalize size ---------
ensure_dir(RENAMED_DIR)

# read first image to set canonical size
H0W0 = imread_rgb(by_angle[0][1]).shape[:2]
H0, W0 = H0W0

renamed_paths = []
for i, (_, p) in enumerate(by_angle):
    img = imread_rgb(p)
    if img.shape[:2] != (H0, W0):
        img = cv2.resize(img, (W0, H0), interpolation=cv2.INTER_AREA)
    outp = os.path.join(RENAMED_DIR, f"{i:03d}.png")  # 000..360
    cv2.imwrite(outp, img)
    renamed_paths.append(outp)

print(f"Renamed and normalized {len(renamed_paths)} images to {RENAMED_DIR}")

# --------- 3) interpolate to 3600 frames ---------
# We want 360 degrees * 10 frames per degree = 3600 frames
# Use pairs (0->1), (1->2), ..., (359->360). For each pair output 10 frames:
# k=0..9 with alpha=k/10. Do not add a final extra duplicate.
ensure_dir(INTERP_DIR)

# Preload to speed up
imgs = [imread_rgb(p) for p in renamed_paths]
H, W = imgs[0].shape[:2]

out_count = 0
for i in range(0, 360):  # 360 steps
    a = imgs[i]
    b = imgs[i+1]
    for k in range(10):  # 0..9 -> 10 frames per step
        alpha = k / 10.0
        # blend = (1-alpha)*a + alpha*b
        blend = cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)
        cv2.imwrite(os.path.join(INTERP_DIR, f"{out_count:04d}.png"), blend)
        out_count += 1

print(f"Saved {out_count} frames to {INTERP_DIR} (expected 3600)")
