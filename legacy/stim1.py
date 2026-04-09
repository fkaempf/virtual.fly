# Turntable stimulus player for 361 originals named like fly-180.png ... fly180.png
# Keys: q or Esc quit, space pause, r reverse, left or right arrows step when paused, - or = change fps

import os, re, glob, time
import cv2
import numpy as np

# ---------------- user settings ----------------
IMG_FOLDER   = "/Users/fkampf/Downloads/aaa"   # change this to your folder
PATTERN      = "fly*.png"          # matches fly-180.png ... fly180.png
FPS          = 60.0                # playback framerate
FULLSCREEN   = False               # True for fullscreen, False for window
BG_COLOR     = "white"             # "white" or "black"
SCALE        = 1.0                 # uniform scale factor
WINDOW_TITLE = "stimulus"
# -----------------------------------------------

def extract_angle(path):
    # last signed integer in filename handles fly-180 and fly180
    m = re.findall(r"[-+]?\d+", os.path.basename(path))
    if not m:
        raise ValueError(f"No angle in {path}")
    return int(m[-1])

def load_sorted_images(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)), key=lambda p: extract_angle(p))
    if not paths:
        raise FileNotFoundError(f"No images found for {os.path.join(folder, pattern)}")
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"Failed to read {p}")
        imgs.append(im)
    return imgs, paths

def make_canvas(img, target_h, target_w, bg="white", scale=1.0):
    H, W = img.shape[:2]
    if scale != 1.0:
        W = int(round(W * scale))
        H = int(round(H * scale))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if bg == "white":
        canvas = np.full((target_h, target_w, 3), 255, np.uint8)
    else:
        canvas = np.zeros((target_h, target_w, 3), np.uint8)

    x = (target_w - W) // 2
    y = (target_h - H) // 2
    x = max(0, x); y = max(0, y)
    Wc = min(W, target_w)
    Hc = min(H, target_h)
    canvas[y:y+Hc, x:x+Wc] = img[:Hc, :Wc]
    return canvas

# Load frames
frames, paths = load_sorted_images(IMG_FOLDER, PATTERN)
n = len(frames)
H0, W0 = frames[0].shape[:2]
print(f"Loaded {n} frames. First file: {os.path.basename(paths[0])}. Size: {W0}x{H0}")

# Create window
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

if FULLSCREEN:
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    rect = cv2.getWindowImageRect(WINDOW_TITLE)
    target_w = int(rect[2]) if rect[2] > 0 else W0
    target_h = int(rect[3]) if rect[3] > 0 else H0
else:
    target_w = int(W0 * SCALE)
    target_h = int(H0 * SCALE)
    cv2.resizeWindow(WINDOW_TITLE, target_w, target_h)

# Playback state
idx = 0
direction = 1
paused = False
fps = FPS
frame_interval = 1.0 / max(1e-6, fps)
next_time = time.perf_counter()

while True:
    if not paused:
        canvas = make_canvas(frames[idx], target_h, target_w, bg=BG_COLOR, scale=SCALE)
        cv2.imshow(WINDOW_TITLE, canvas)
        idx = (idx + direction) % n

    now = time.perf_counter()
    delay = max(0.0, next_time - now)
    key = cv2.waitKey(int(delay * 1000) if delay > 0 else 1) & 0xFF
    next_time = max(next_time + frame_interval, time.perf_counter())

    if key in (ord('q'), 27):        # q or Esc
        break
    elif key == ord(' '):            # pause or resume
        paused = not paused
    elif key == ord('r'):            # reverse
        direction *= -1
    elif key in (ord('-'), ord('_')):  # slower
        fps = max(1.0, fps - 1.0)
        frame_interval = 1.0 / fps
        print(f"fps -> {fps:.1f}")
    elif key in (ord('='), ord('+')):  # faster
        fps = min(240.0, fps + 1.0)
        frame_interval = 1.0 / fps
        print(f"fps -> {fps:.1f}")
    elif key == 81:  # left arrow
        if paused:
            idx = (idx - 1) % n
            canvas = make_canvas(frames[idx], target_h, target_w, bg=BG_COLOR, scale=SCALE)
            cv2.imshow(WINDOW_TITLE, canvas)
    elif key == 83:  # right arrow
        if paused:
            idx = (idx + 1) % n
            canvas = make_canvas(frames[idx], target_h, target_w, bg=BG_COLOR, scale=SCALE)
            cv2.imshow(WINDOW_TITLE, canvas)

cv2.destroyAllWindows()
