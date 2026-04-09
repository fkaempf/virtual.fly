import os, re, glob, math
import numpy as np
import cv2

# ------------- config -------------
SRC_FOLDER    = "/Users/fkampf/Downloads/aaa"   # your input folder
RENAMED_DIR   = "renamed_000_360"          # normalized originals 000..360
OUT_DIR       = "turntable_interp_3600"    # final 0.1 deg steps
FRAMES_PER_DEGREE = 10                     # 10 gives 3600 output frames
FLOW_ALGO = "DIS"                          # "DIS" or "Farneback"
BG_COLOR = (255, 255, 255)                 # white background
THRESH = 250                               # silhouette threshold for optional masks
# -----------------------------------

def find_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return paths

def extract_angle(path):
    # pick the last signed integer in the name (covers fly-180 and fly180)
    base = os.path.basename(path)
    m = re.findall(r"[-+]?\d+", base)
    if not m:
        raise ValueError(f"No angle found in filename: {base}")
    return int(m[-1])

def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read {path}")
    return img

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def to_mask(img_bgr, thresh=THRESH):
    # treat near white as background, everything else as foreground
    return ((img_bgr[...,0] < thresh) | (img_bgr[...,1] < thresh) | (img_bgr[...,2] < thresh)).astype(np.uint8)

def build_flow(imgA, imgB, algo="DIS"):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    if algo.upper() == "DIS":
        # Fast and robust for small inter frame rotation
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        dis.setUseSpatialPropagation(True)
        flow = dis.calc(grayA, grayB, None)  # flow from A to B
    else:
        # Farneback as fallback
        flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None,
                                            pyr_scale=0.5, levels=4, winsize=21,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    return flow

def warp_with_flow(img, flow):
    H, W = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=BG_COLOR)

# 1) collect and sort originals by angle
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
    raise RuntimeError("Could not parse any angles from filenames")

by_angle.sort(key=lambda x: x[0])
angles = [a for a,_ in by_angle]
print("Angle range:", angles[0], "to", angles[-1], "count:", len(angles))

# 2) normalize size and rename to 000..360
ensure_dir(RENAMED_DIR)
first_img = imread_rgb(by_angle[0][1])
H0, W0 = first_img.shape[:2]

renamed_paths = []
for i, (_, p) in enumerate(by_angle):
    img = imread_rgb(p)
    if img.shape[:2] != (H0, W0):
        img = cv2.resize(img, (W0, H0), interpolation=cv2.INTER_AREA)
    outp = os.path.join(RENAMED_DIR, f"{i:03d}.png")
    cv2.imwrite(outp, img)
    renamed_paths.append(outp)

print(f"Saved normalized originals to {RENAMED_DIR}")

# Optional foreground masks to reduce background ghosting
orig_imgs = [imread_rgb(p) for p in renamed_paths]
masks = [to_mask(im) for im in orig_imgs]

# 3) flow based interpolation for real turntable effect
ensure_dir(OUT_DIR)
out_idx = 0

for i in range(0, len(orig_imgs) - 1):
    A = orig_imgs[i]
    B = orig_imgs[i+1]
    MA = masks[i]
    MB = masks[i+1]

    # forward and backward flows
    flowAB = build_flow(A, B, algo=FLOW_ALGO)
    flowBA = build_flow(B, A, algo=FLOW_ALGO)

    # write the original A at the start of the step
    cv2.imwrite(os.path.join(OUT_DIR, f"{out_idx:04d}.png"), A)
    out_idx += 1

    # generate FRAMES_PER_DEGREE-1 in-betweens
    for k in range(1, FRAMES_PER_DEGREE):
        t = k / FRAMES_PER_DEGREE

        # scale the flows
        flow_t_AB = flowAB * t
        flow_t_BA = flowBA * (1.0 - t)

        # warp both toward the middle time
        A_t = warp_with_flow(A, flow_t_AB)
        B_t = warp_with_flow(B, flow_t_BA)

        # simple symmetric mask blend
        wA = (MA.astype(np.float32) * (1.0 - t))[:, :, None]
        wB = (MB.astype(np.float32) * t)[:, :, None]
        denom = wA + wB + 1e-6
        blend = (A_t.astype(np.float32) * wA + B_t.astype(np.float32) * wB) / denom
        blend = np.clip(blend, 0, 255).astype(np.uint8)

        # for background, prefer clean white
        bg = (1 - ((MA | MB)[:, :, None])).astype(np.uint8) * 255
        out = np.where((MA | MB)[:, :, None].astype(bool), blend, bg)

        cv2.imwrite(os.path.join(OUT_DIR, f"{out_idx:04d}.png"), out)
        out_idx += 1

# add the very last original
cv2.imwrite(os.path.join(OUT_DIR, f"{out_idx:04d}.png"), orig_imgs[-1])
out_idx += 1

print(f"Saved {out_idx} frames to {OUT_DIR}")
