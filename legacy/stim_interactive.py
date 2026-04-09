# WASD fly with distance scaling, correct sprite front, fixed turning, minimap trail,
# faster turning while standing still, and tiny walk noise

import os, re, glob, math, time
import cv2
import numpy as np
import pygame
from collections import deque

# --------------- user settings ---------------
IMG_FOLDER   = '/Users/fkampf/Documents/one2one-mapping/mycode/red_pics'
PATTERN      = "fly*.png"

# Window layout
STIM_W, STIM_H = 1280, 720
MAP_W,  MAP_H  = 420,  420
PAD = 12
BG_COLOR = (255, 255, 255)


VSYNC = True          # try to use display vsync
VSYNC_FALLBACK_FPS = 120  # used only if vsync is unavailable

# Image preprocessing
NEAR_WHITE   = 245
SPRITE_MARGIN_PX = 2

# World and camera
FPS             = 120.0
ARENA_X_MM      = 200.0
ARENA_Y_MM      = (80.0, 420.0)

# Distance to size mapping
REF_DIST_MM       = 220.0
DESIRED_PX_AT_REF = 260.0
MIN_SCALE         = 0.10
MAX_SCALE         = 7.0

# Controls
SPEED_MM_S    = 140.0
BACK_MM_S     = 90.0
TURN_DEG_S    = 200             # base turn rate
STAND_TURN_MULT = 1.5           # multiplier when not pressing W or S
START_POS     = (0.0, 250.0)
START_HEADING_DEG = 180.0

# Walk noise (applied only while W or S is pressed)
WALK_TURN_NOISE_DEG_RMS = 20.0  # deg per sqrt(second)
WALK_TRANS_NOISE_MM_RMS = 5.0   # mm per sqrt(second) per axis

# Minimap trail
TRAIL_SECS    = 5.0
TRAIL_COLOR   = (255, 200, 0)   # BGR
TRAIL_THICK   = 2
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

    sprites, masks = [], []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"Failed to read {p}")

        if im.ndim == 3 and im.shape[2] == 4:
            b, g, r, a = cv2.split(im)
            fg = (a > 0).astype(np.uint8)
            rgb = cv2.merge([b, g, r])
        else:
            rgb = im
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            fg = (gray < NEAR_WHITE).astype(np.uint8)

        ys, xs = np.where(fg > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0 = max(0, int(xs.min()) - SPRITE_MARGIN_PX)
        x1 = min(rgb.shape[1], int(xs.max()) + 1 + SPRITE_MARGIN_PX)
        y0 = max(0, int(ys.min()) - SPRITE_MARGIN_PX)
        y1 = min(rgb.shape[0], int(ys.max()) + 1 + SPRITE_MARGIN_PX)

        sprites.append(rgb[y0:y1, x0:x1].copy())
        masks.append(fg[y0:y1, x0:x1].copy())

    if not sprites:
        raise RuntimeError("Cropping produced no sprites. Adjust NEAR_WHITE or check inputs.")
    return sprites, masks

def angle_to_index(deg):
    deg = (deg + 180.0) % 360.0 - 180.0
    frac = (deg + 180.0) / 360.0
    idx = int(round(frac * 360.0))
    return max(0, min(idx, 360))

def scale_from_distance(y_mm, ref_sprite_h_px):
    height_px = DESIRED_PX_AT_REF * (REF_DIST_MM / max(1.0, y_mm))
    scale = float(height_px) / max(1.0, ref_sprite_h_px)
    return max(MIN_SCALE, min(MAX_SCALE, scale))

def render_sprite_masked_bgr(canvas_bgr, sprite_bgr, mask, center_xy, scale):
    cx, cy = center_xy
    if scale != 1.0:
        new_w = max(1, int(round(sprite_bgr.shape[1] * scale)))
        new_h = max(1, int(round(sprite_bgr.shape[0] * scale)))
        spr = cv2.resize(sprite_bgr, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        msk = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        spr = sprite_bgr
        msk = mask
    h, w = spr.shape[:2]
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = max(0, x0); y1 = max(0, y0)
    x2 = min(STIM_W, x0 + w); y2 = min(STIM_H, y0 + h)
    if x2 <= x1 or y2 <= y1:
        return False
    sx1 = x1 - x0; sy1 = y1 - y0
    sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)
    roi = canvas_bgr[y1:y2, x1:x2]
    spr_c = spr[sy1:sy2, sx1:sx2]
    msk_c = msk[sy1:sy2, sx1:sx2].astype(bool)
    roi[msk_c] = spr_c[msk_c]
    canvas_bgr[y1:y2, x1:x2] = roi
    return True

# minimap helpers
MAP_PAD = 24
def world_to_map(x_mm, y_mm):
    x_min, x_max = -ARENA_X_MM, ARENA_X_MM
    y_min, y_max = ARENA_Y_MM[0], ARENA_Y_MM[1]
    u = int(round(MAP_PAD + (x_mm - x_min) * (MAP_W - 2*MAP_PAD) / (x_max - x_min)))
    v = int(round(MAP_H - MAP_PAD - (y_mm - y_min) * (MAP_H - 2*MAP_PAD) / (y_max - y_min)))
    return u, v

def draw_fly_triangle(img_bgr, center_u, center_v, heading_rad, size_px=18, color=(0,120,255)):
    angle_img = heading_rad   # 0 points up on the map
    tip  = np.array([0, -size_px], dtype=np.float32)
    left = np.array([-size_px*0.6, size_px*0.7], dtype=np.float32)
    right= np.array([ size_px*0.6, size_px*0.7], dtype=np.float32)
    R = np.array([[ math.cos(angle_img), -math.sin(angle_img)],
                  [ math.sin(angle_img),  math.cos(angle_img)]], dtype=np.float32)
    pts = np.stack([tip, left, right], axis=0) @ R.T
    pts[:, 0] += center_u
    pts[:, 1] += center_v
    cv2.fillConvexPoly(img_bgr, pts.astype(np.int32), color)
    cv2.polylines(img_bgr, [pts.astype(np.int32)], True, (0,0,0), 1, cv2.LINE_AA)

def draw_minimap_panel(x, y, heading, trail_pts_uv):
    img = np.full((MAP_H, MAP_W, 3), 255, np.uint8)
    cv2.rectangle(img, (MAP_PAD, MAP_PAD), (MAP_W - MAP_PAD, MAP_H - MAP_PAD), (0,0,0), 2)
    cam_u = MAP_W // 2
    cam_v = MAP_H - MAP_PAD
    cv2.circle(img, (cam_u, cam_v), 6, (0,0,255), -1)
    cv2.putText(img, "cam", (cam_u + 10, cam_v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    # trail for last TRAIL_SECS with fading
    if len(trail_pts_uv) >= 2:
        now_color = np.array(TRAIL_COLOR, dtype=np.float32)
        nseg = len(trail_pts_uv) - 1
        for i in range(nseg):
            p0 = trail_pts_uv[i]
            p1 = trail_pts_uv[i + 1]
            alpha = (i + 1) / nseg
            col = tuple((now_color * (0.3 + 0.7 * alpha)).astype(np.int32).tolist())
            cv2.line(img, p0, p1, col, TRAIL_THICK, cv2.LINE_AA)

    fu, fv = world_to_map(x, y)
    draw_fly_triangle(img, fu, fv, heading, size_px=18)
    return img
# ---------------------------

# load sprites
sprites, masks = load_and_crop_sorted(IMG_FOLDER, PATTERN)
n = len(sprites)
ref_h = sprites[0].shape[0]

# state
x, y = START_POS
heading = math.radians(START_HEADING_DEG)
speed_f = SPEED_MM_S
speed_b = BACK_MM_S
turn_rate_rad = math.radians(TURN_DEG_S)

# trail buffer stores (t, x, y)
trail = deque()

# pygame window
pygame.init()
total_w = STIM_W + PAD + MAP_W
total_h = max(STIM_H, MAP_H)
screen = pygame.display.set_mode((total_w, total_h))
pygame.display.set_caption("fly stimulus + minimap")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

def bgr_to_surface(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    now = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
        running = False

    # standing turn multiplier
    moving = keys[pygame.K_w] or keys[pygame.K_s]
    current_turn_rate = turn_rate_rad * (STAND_TURN_MULT if not moving else 1.0)

    # turn
    if keys[pygame.K_a]:
        heading -= current_turn_rate * dt
    if keys[pygame.K_d]:
        heading += current_turn_rate * dt

    # move
    if keys[pygame.K_w]:
        x += speed_f * math.sin(heading) * dt
        y += speed_f * math.cos(heading) * dt
    if keys[pygame.K_s]:
        x -= speed_b * math.sin(heading) * dt
        y -= speed_b * math.cos(heading) * dt

    # tiny noise while walking
    if moving:
        # rotational noise
        heading += math.radians(WALK_TURN_NOISE_DEG_RMS) * np.random.normal(0.0, math.sqrt(dt))
        # translational noise in world coords
        x += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))
        y += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))

    # clamp
    x = min(max(x, -ARENA_X_MM), ARENA_X_MM)
    y = min(max(y, ARENA_Y_MM[0]), ARENA_Y_MM[1])

    # add to trail and drop older than TRAIL_SECS
    trail.append((now, x, y))
    expire_before = now - TRAIL_SECS
    while trail and trail[0][0] < expire_before:
        trail.popleft()

    # stimulus panel
    stim = np.full((STIM_H, STIM_W, 3), BG_COLOR, np.uint8)
    cx = STIM_W // 2
    px_per_mm = STIM_W / (2 * ARENA_X_MM)
    px = int(round(cx + x * px_per_mm))
    py = STIM_H // 2

    yaw_deg = (math.degrees(heading)) % 360.0
    idx = angle_to_index(yaw_deg)
    idx = min(idx, n - 1)
    scale = scale_from_distance(y, ref_h)

    render_sprite_masked_bgr(stim, sprites[idx], masks[idx], (px, py), scale)
    cv2.putText(stim, f"y={y:.1f}mm scale={scale:.2f} yaw={yaw_deg:.1f} idx={idx}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

    # minimap panel
    trail_uv = [world_to_map(xi, yi) for _, xi, yi in trail]
    map_img = draw_minimap_panel(x, y, heading, trail_uv)

    # blit both panels
    screen.fill((30, 30, 30))
    screen.blit(bgr_to_surface(stim), (0, 0))
    screen.blit(bgr_to_surface(map_img), (STIM_W + PAD, 0))

    info = f"WASD move+turn   y={y:.1f}mm  scale={scale:.2f}"
    screen.blit(font.render(info, True, (220, 220, 220)), (10, total_h - 24))
    pygame.display.flip()

pygame.quit()
