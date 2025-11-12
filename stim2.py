# Random walk stimulus from 361 images, with auto-cropping and masked rendering
# Keys: q or Esc quit, space pause, - or = change speed

import os, re, glob, time, math, random
import cv2
import numpy as np

# --------------- user settings ---------------
IMG_FOLDER   = "/Users/fkampf/Downloads/aaa"     # folder with fly-180.png ... fly180.png
PATTERN      = "fly*.png"
WINDOW_TITLE = "fly_stimulus"

FPS          = 120.0
WIN_W, WIN_H = 1280, 720
BG_COLOR     = (255, 255, 255)       # white background
NEAR_WHITE   = 245                   # pixels >= this are treated as background
SPRITE_MARGIN_PX = 2                 # pad after cropping
SPRITE_BOOST  = 1.8                  # extra scale to make the fly larger on screen

# Simple pinhole camera
FOV_X_DEG    = 90.0
BODY_LEN_MM  = 3.0
ARENA_X_MM   = 120.0
ARENA_Y_MM   = (120.0, 280.0)        # keep distance moderate so size is not tiny

# Random walk
DT           = 1.0 / FPS
SPEED_MM_S   = 80.0
TURN_SIGMA   = math.radians(20.0)
START_POS    = (0.0, 200.0)
START_HEADING_DEG = 180.0

# Scale clamps
MIN_SCALE = 0.5
MAX_SCALE = 6.0
# ---------------------------------------------

def extract_angle(path):
    m = re.findall(r"[-+]?\d+", os.path.basename(path))
    if not m:
        raise ValueError(f"No angle in {path}")
    return int(m[-1])

def load_and_crop_sorted(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)), key=lambda p: extract_angle(p))
    if not paths:
        raise FileNotFoundError(f"No images found for {os.path.join(folder, pattern)}")

    sprites = []
    masks   = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"Failed to read {p}")

        # Convert RGBA to BGR and take alpha if present
        if im.ndim == 3 and im.shape[2] == 4:
            b, g, r, a = cv2.split(im)
            # Treat alpha as foreground mask if provided
            fg_mask = (a > 0).astype(np.uint8)
            rgb = cv2.merge([b, g, r])
        else:
            rgb = im
            # Build mask from near-white background
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            fg_mask = (gray < NEAR_WHITE).astype(np.uint8)

        # Tight crop to foreground
        ys, xs = np.where(fg_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0 = max(0, int(xs.min()) - SPRITE_MARGIN_PX)
        x1 = min(rgb.shape[1], int(xs.max()) + 1 + SPRITE_MARGIN_PX)
        y0 = max(0, int(ys.min()) - SPRITE_MARGIN_PX)
        y1 = min(rgb.shape[0], int(ys.max()) + 1 + SPRITE_MARGIN_PX)

        crop = rgb[y0:y1, x0:x1].copy()
        cmask = fg_mask[y0:y1, x0:x1].copy()

        sprites.append(crop)
        masks.append(cmask)

    if not sprites:
        raise RuntimeError("No sprites after cropping. Check NEAR_WHITE or input images.")
    return sprites, masks

def angle_to_index(deg):
    # Map -180..180 to 0..360 inclusive
    deg = (deg + 180.0) % 360.0 - 180.0
    frac = (deg + 180.0) / 360.0
    idx = int(round(frac * 360.0))
    return max(0, min(idx, 360))

def step_random_walk(x, y, heading, dt, speed):
    dtheta = random.gauss(0.0, TURN_SIGMA * math.sqrt(dt))
    heading = (heading + dtheta) % (2 * math.pi)
    vx = speed * math.sin(heading)
    vy = speed * math.cos(heading)
    x += vx * dt
    y += vy * dt

    if x < -ARENA_X_MM:
        x = -ARENA_X_MM; heading = math.pi - heading
    if x > ARENA_X_MM:
        x = ARENA_X_MM; heading = math.pi - heading
    if y < ARENA_Y_MM[0]:
        y = ARENA_Y_MM[0]; heading = -heading
    if y > ARENA_Y_MM[1]:
        y = ARENA_Y_MM[1]; heading = -heading
    return x, y, heading

def project_to_screen(x, y, f_px):
    az = math.atan2(x, y)
    u = f_px * math.tan(az)
    cx = WIN_W // 2
    cy = WIN_H // 2
    return int(round(cx + u)), cy, az

def scale_from_distance(y_mm, f_px, ref_h_px):
    theta = 2.0 * math.atan(BODY_LEN_MM / (2.0 * max(1e-3, y_mm)))
    H_px = 2.0 * f_px * math.tan(theta / 2.0)
    scale = float(H_px) / max(1.0, ref_h_px)
    scale *= SPRITE_BOOST
    return max(MIN_SCALE, min(MAX_SCALE, scale))

def render_sprite_masked(canvas, sprite, mask, center_xy, scale):
    cx, cy = center_xy
    if scale != 1.0:
        new_w = max(1, int(round(sprite.shape[1] * scale)))
        new_h = max(1, int(round(sprite.shape[0] * scale)))
        spr = cv2.resize(sprite, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        msk = cv2.resize(mask,   (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        spr = sprite
        msk = mask
    h, w = spr.shape[:2]
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = max(0, x0); y1 = max(0, y0)
    x2 = min(WIN_W, x0 + w); y2 = min(WIN_H, y0 + h)
    if x2 <= x1 or y2 <= y1:
        return False
    sx1 = x1 - x0; sy1 = y1 - y0
    sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)

    roi = canvas[y1:y2, x1:x2]
    spr_c = spr[sy1:sy2, sx1:sx2]
    msk_c = msk[sy1:sy2, sx1:sx2].astype(bool)

    roi[msk_c] = spr_c[msk_c]
    canvas[y1:y2, x1:x2] = roi
    return True

# Load and crop
sprites, masks = load_and_crop_sorted(IMG_FOLDER, PATTERN)
n = len(sprites)
print(f"Cropped sprites: {n}")
ref_h = sprites[0].shape[0]

# Camera focal length in pixels
f_px = (WIN_W / 2.0) / math.tan(math.radians(FOV_X_DEG) / 2.0)

# State
x, y = START_POS
heading = math.radians(START_HEADING_DEG)
paused = False
speed = SPEED_MM_S

# Window
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_TITLE, WIN_W, WIN_H)

frame_interval = 1.0 / max(1e-6, FPS)
next_time = time.perf_counter()

while True:
    if not paused:
        x, y, heading = step_random_walk(x, y, heading, DT, speed)
        px, py, az = project_to_screen(x, y, f_px)

        # choose yaw so the fly faces the camera
        yaw_deg = math.degrees(-az) + 180.0
        idx = angle_to_index(yaw_deg)
        idx = min(idx, n - 1)

        scale = scale_from_distance(y, f_px, ref_h)

        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, np.uint8)
        render_sprite_masked(canvas, sprites[idx], masks[idx], (px, py), scale)

        # debug overlay
        cv2.putText(canvas, f"x={x:.1f} y={y:.1f} az={math.degrees(az):.1f} idx={idx} scale={scale:.2f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_TITLE, canvas)

    now = time.perf_counter()
    delay = max(0.0, next_time - now)
    key = cv2.waitKey(int(delay * 1000) if delay > 0 else 1) & 0xFF
    next_time = max(next_time + frame_interval, time.perf_counter())

    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        paused = not paused
    elif key in (ord('-'), ord('_')):
        speed = max(5.0, speed - 5.0); print(f"speed -> {speed:.1f} mm/s")
    elif key in (ord('='), ord('+')):
        speed = min(500.0, speed + 5.0); print(f"speed -> {speed:.1f} mm/s")

cv2.destroyAllWindows()
