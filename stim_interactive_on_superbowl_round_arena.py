# WASD fly with distance scaling, correct sprite front, fixed turning,
# faster turning while standing still, and tiny walk noise
# Main stimulus: projector mapping (camera -> projector via mapx/mapy),
# displayed on rightmost monitor using NOFRAME window like calibration.
# Minimap: separate OpenCV window on primary (main) screen.

import os, re, glob, math, time
import cv2
import numpy as np
import pygame
from collections import deque

# ----------------- helpers -----------------


def extract_angle(path):
    m = re.findall(r"[-+]?\d+", os.path.basename(path))
    if not m:
        raise ValueError(f"No angle in {path}")
    return int(m[-1])


def load_and_crop_sorted(folder, pattern, NEAR_WHITE, SPRITE_MARGIN_PX):
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


def scale_from_distance(
    dist_mm,
    ref_sprite_h_px,
    REF_DIST_MM,
    DESIRED_PX_AT_REF,
    MIN_SCALE,
    MAX_SCALE,
    VIRTUAL_CAMERA_OFFSET_MM,
    GLOBAL_SIZE_SCALE,
):
    d_eff = dist_mm + VIRTUAL_CAMERA_OFFSET_MM
    d_eff = max(1.0, d_eff)

    height_px = DESIRED_PX_AT_REF * (REF_DIST_MM / d_eff)
    height_px *= GLOBAL_SIZE_SCALE

    scale = float(height_px) / max(1.0, ref_sprite_h_px)
    return max(MIN_SCALE, min(MAX_SCALE, scale))


def render_sprite_masked_bgr(canvas_bgr, sprite_bgr, mask, center_xy, scale):
    H, W = canvas_bgr.shape[:2]
    cx, cy = center_xy
    if scale != 1.0:
        new_w = max(1, int(round(sprite_bgr.shape[1] * scale)))
        new_h = max(1, int(round(sprite_bgr.shape[0] * scale)))
        spr = cv2.resize(
            sprite_bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        msk = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        spr = sprite_bgr
        msk = mask
    h, w = spr.shape[:2]
    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(W, x0 + w)
    y2 = min(H, y0 + h)
    if x2 <= x1 or y2 <= y1:
        return False
    sx1 = x1 - x0
    sy1 = y1 - y0
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    roi = canvas_bgr[y1:y2, x1:x2]
    spr_c = spr[sy1:sy2, sx1:sx2]
    msk_c = msk[sy1:sy2, sx1:sx2].astype(bool)
    roi[msk_c] = spr_c[msk_c]
    canvas_bgr[y1:y2, x1:x2] = roi
    return True


def world_to_minimap(x_mm, y_mm, center_u, center_v, scale_px_per_mm):
    """Map world coords (mm) to minimap pixels with a given scale and center."""
    u = int(round(center_u + x_mm * scale_px_per_mm))
    v = int(round(center_v - y_mm * scale_px_per_mm))
    return u, v


def draw_fly_triangle(img_bgr, center_u, center_v, heading_rad, size_px=18, color=(0, 120, 255)):
    angle_img = heading_rad
    tip = np.array([0, -size_px], dtype=np.float32)
    left = np.array([-size_px * 0.6, size_px * 0.7], dtype=np.float32)
    right = np.array([size_px * 0.6, size_px * 0.7], dtype=np.float32)
    R = np.array(
        [
            [math.cos(angle_img), -math.sin(angle_img)],
            [math.sin(angle_img), math.cos(angle_img)],
        ],
        dtype=np.float32,
    )
    pts = np.stack([tip, left, right], axis=0) @ R.T
    pts[:, 0] += center_u
    pts[:, 1] += center_v
    cv2.fillConvexPoly(img_bgr, pts.astype(np.int32), color)
    cv2.polylines(img_bgr, [pts.astype(np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)


def draw_minimap_panel(
    x,
    y,
    heading,
    trail_pts_uv,
    ARENA_RADIUS_MM,
    CAMERA_X_MM,
    CAMERA_Y_MM,
    MAP_W,
    MAP_H,
    MAP_PAD,
    TRAIL_COLOR,
    TRAIL_THICK,
    center_u,
    center_v,
    scale_px_per_mm,
):
    img = np.full((MAP_H, MAP_W, 3), 255, np.uint8)

    # arena circle
    radius_px = int(round(ARENA_RADIUS_MM * scale_px_per_mm))
    cv2.circle(img, (center_u, center_v), radius_px, (0, 0, 0), 2)

    # camera at true position, not clamped to the arena
    cam_u, cam_v = world_to_minimap(CAMERA_X_MM, CAMERA_Y_MM, center_u, center_v, scale_px_per_mm)
    cv2.circle(img, (cam_u, cam_v), 6, (0, 0, 255), -1)
    cv2.putText(
        img,
        "cam",
        (cam_u + 10, cam_v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    # trail (already pre-mapped to minimap coords)
    if len(trail_pts_uv) >= 2:
        now_color = np.array(TRAIL_COLOR, dtype=np.float32)
        nseg = len(trail_pts_uv) - 1
        for i in range(nseg):
            p0 = trail_pts_uv[i]
            p1 = trail_pts_uv[i + 1]
            alpha = (i + 1) / nseg
            col = tuple((now_color * (0.3 + 0.7 * alpha)).astype(np.int32).tolist())
            cv2.line(img, p0, p1, col, TRAIL_THICK, cv2.LINE_AA)

    # fly at true position in arena
    fu, fv = world_to_minimap(x, y, center_u, center_v, scale_px_per_mm)
    draw_fly_triangle(img, fu, fv, heading, size_px=18)
    return img


def _frame_to_surface(img_u8, size_wh):
    W, H = size_wh
    if img_u8.dtype != np.uint8:
        img_u8 = cv2.convertScaleAbs(img_u8)
    if img_u8.ndim == 2:
        rgb = np.dstack([img_u8] * 3)
    else:
        rgb = img_u8[..., ::-1] if img_u8.shape[2] == 3 else img_u8
    if (rgb.shape[1], rgb.shape[0]) != (W, H):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None


def _pick_monitor_rightmost(default_w, default_h):
    if get_monitors:
        mons = get_monitors()
        if mons:
            return max(mons, key=lambda m: m.x)

    class M:
        pass

    m = M()
    m.x = 0
    m.y = 0
    m.width = default_w
    m.height = default_h
    return m


# ----------------- main driver -----------------


def start_stimulus(
    # sprite / image
    IMG_FOLDER,
    PATTERN,
    NEAR_WHITE,
    SPRITE_MARGIN_PX,
    # display
    BG_COLOR,
    FPS,
    WINDOW_W,
    WINDOW_H,
    # world / camera geometry
    ARENA_RADIUS_MM,
    CAMERA_X_MM,
    CAMERA_Y_MM,
    # distance / size mapping
    REF_DIST_MM,
    DESIRED_PX_AT_REF,
    MIN_SCALE,
    MAX_SCALE,
    VIRTUAL_CAMERA_OFFSET_MM,
    GLOBAL_SIZE_SCALE,
    # motion / controls
    SPEED_MM_S,
    BACK_MM_S,
    TURN_DEG_S,
    STAND_TURN_MULT,
    START_POS,
    START_HEADING_DEG,
    # noise
    WALK_TURN_NOISE_DEG_RMS,
    WALK_TRANS_NOISE_MM_RMS,
    # camera image vertical offset
    CAMERA_Y_OFFSET_PX,
    # minimap
    MAP_W,
    MAP_H,
    MAP_PAD,
    TRAIL_SECS,
    TRAIL_COLOR,
    TRAIL_THICK,
    # mapping paths
    MAPX_PATH,
    MAPY_PATH,
):
    sprites, masks = load_and_crop_sorted(IMG_FOLDER, PATTERN, NEAR_WHITE, SPRITE_MARGIN_PX)
    n = len(sprites)
    ref_h = sprites[0].shape[0]

    mapx = np.load(MAPX_PATH).astype(np.float32)
    mapy = np.load(MAPY_PATH).astype(np.float32)

    PROJ_H, PROJ_W = mapx.shape

    valid = np.isfinite(mapx) & np.isfinite(mapy) & (mapx >= 0) & (mapy >= 0)
    if not np.any(valid):
        raise RuntimeError("mapx/mapy contain no valid entries")

    cam_w = int(np.ceil(mapx[valid].max())) + 1
    cam_h = int(np.ceil(mapy[valid].max())) + 1

    # --- minimap transform: fit both arena and camera in view without clamping camera ---
    R = ARENA_RADIUS_MM
    r_cam = math.hypot(CAMERA_X_MM, CAMERA_Y_MM)

    # available half-height/half-width in pixels
    half_h = (MAP_H - 2 * MAP_PAD) / 2.0
    half_w = (MAP_W - 2 * MAP_PAD) / 2.0

    # vertical constraint: circle top and camera must fit
    # y in world: +R (top of arena), -r_cam (camera, possibly below)
    # span in mm = R + r_cam
    if r_cam <= 0:
        span_vert_mm = 2 * R
    else:
        span_vert_mm = R + r_cam
    scale_vert = (MAP_H - 2 * MAP_PAD) / span_vert_mm

    # horizontal constraint: arena must fit left/right
    scale_horiz = half_w / R

    scale_px_per_mm = min(scale_vert, scale_horiz)
    # center vertically so top of arena is at MAP_PAD and camera at bottom margin (approx)
    center_u = MAP_W // 2
    center_v = MAP_PAD + int(round(R * scale_px_per_mm))

    x, y = START_POS
    heading = math.radians(START_HEADING_DEG)
    speed_f = SPEED_MM_S
    speed_b = BACK_MM_S
    turn_rate_rad = math.radians(TURN_DEG_S)

    trail = deque()

    win_w = WINDOW_W
    win_h = WINDOW_H
    m = _pick_monitor_rightmost(win_w, win_h)
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{m.x},{m.y}"

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h), pygame.SWSURFACE | pygame.NOFRAME)
    pygame.display.set_caption("Probe projector mapping (fly)")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    cv2.namedWindow("minimap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("minimap", MAP_W, MAP_H)
    cv2.moveWindow("minimap", 50, 50)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        fps_now = clock.get_fps()
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

        moving = keys[pygame.K_w] or keys[pygame.K_s]
        current_turn_rate = turn_rate_rad * (STAND_TURN_MULT if not moving else 1.0)

        if keys[pygame.K_a]:
            heading -= current_turn_rate * dt
        if keys[pygame.K_d]:
            heading += current_turn_rate * dt

        if keys[pygame.K_w]:
            x += speed_f * math.sin(heading) * dt
            y += speed_f * math.cos(heading) * dt
        if keys[pygame.K_s]:
            x -= speed_b * math.sin(heading) * dt
            y -= speed_b * math.cos(heading) * dt

        if moving:
            heading += math.radians(WALK_TURN_NOISE_DEG_RMS) * np.random.normal(0.0, math.sqrt(dt))
            x += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))
            y += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))

        r_center = math.hypot(x, y)
        if r_center > ARENA_RADIUS_MM:
            scale_back = ARENA_RADIUS_MM / r_center
            x *= scale_back
            y *= scale_back

        trail.append((now, x, y))
        expire_before = now - TRAIL_SECS
        while trail and trail[0][0] < expire_before:
            trail.popleft()

        cam_img = np.full((cam_h, cam_w, 3), BG_COLOR, np.uint8)
        cx_cam = cam_w // 2
        px_per_mm = cam_w / (2 * ARENA_RADIUS_MM)
        px_cam = int(round(cx_cam + x * px_per_mm))
        py_cam = cam_h // 2 + CAMERA_Y_OFFSET_PX

        yaw_deg = (math.degrees(heading)) % 360.0
        idx = angle_to_index(yaw_deg)
        idx = min(idx, n - 1)

        dx_cam = x - CAMERA_X_MM
        dy_cam = y - CAMERA_Y_MM
        dist_mm = math.hypot(dx_cam, dy_cam)
        scale = scale_from_distance(
            dist_mm,
            ref_h,
            REF_DIST_MM,
            DESIRED_PX_AT_REF,
            MIN_SCALE,
            MAX_SCALE,
            VIRTUAL_CAMERA_OFFSET_MM,
            GLOBAL_SIZE_SCALE,
        )

        render_sprite_masked_bgr(cam_img, sprites[idx], masks[idx], (px_cam, py_cam), scale)
        cv2.putText(
            cam_img,
            f"dist={dist_mm:.1f}mm scale={scale:.2f} yaw={yaw_deg:.1f} idx={idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        proj_stim = cv2.remap(
            cam_img,
            mapx,
            mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        surf = _frame_to_surface(proj_stim, (win_w, win_h))
        screen.blit(surf, (0, 0))

        info = f"WASD move+turn   dist={dist_mm:.1f}mm  scale={scale:.2f}"
        text_surf = font.render(info, True, (220, 220, 220))
        screen.blit(text_surf, (10, win_h - 24))

        pygame.display.flip()

        # trail mapped using SAME minimap transform (so its geometry is consistent)
        trail_uv = [
            world_to_minimap(xi, yi, center_u, center_v, scale_px_per_mm)
            for _, xi, yi in trail
        ]
        map_img = draw_minimap_panel(
            x,
            y,
            heading,
            trail_uv,
            ARENA_RADIUS_MM,
            CAMERA_X_MM,
            CAMERA_Y_MM,
            MAP_W,
            MAP_H,
            MAP_PAD,
            TRAIL_COLOR,
            TRAIL_THICK,
            center_u,
            center_v,
            scale_px_per_mm,
        )

        # overlay current FPS on minimap
        cv2.putText(
            map_img,
            f"FPS: {fps_now:.1f}",
            (10, MAP_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("minimap", map_img)
        cv2.waitKey(1)

    pygame.quit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # sprite / image
    IMG_FOLDER = r"D:\virtual.fly\og_pics"
    PATTERN = "fly*.png"
    NEAR_WHITE = 245
    SPRITE_MARGIN_PX = 2

    # display
    BG_COLOR = (255, 255, 255)
    FPS = 0
    WINDOW_W, WINDOW_H = 1280, 800

    # world / camera geometry
    ARENA_RADIUS_MM = 200
    CAMERA_X_MM = 0.0
    CAMERA_Y_MM = -ARENA_RADIUS_MM * 1.3  # true camera position, outside arena

    # distance / size mapping
    REF_DIST_MM = 220.0
    DESIRED_PX_AT_REF = 260.0
    MIN_SCALE = 0.10
    MAX_SCALE = 7.0
    VIRTUAL_CAMERA_OFFSET_MM = 0.0
    GLOBAL_SIZE_SCALE = 1.0

    # motion / controls
    SPEED_MM_S = 140.0
    BACK_MM_S = 90.0
    TURN_DEG_S = 200
    STAND_TURN_MULT = 1.5
    START_POS = (0.0, 250.0)
    START_HEADING_DEG = 180.0

    # noise
    WALK_TURN_NOISE_DEG_RMS = 20.0
    WALK_TRANS_NOISE_MM_RMS = 5.0

    # camera image vertical offset
    CAMERA_Y_OFFSET_PX = 100

    # minimap
    MAP_W, MAP_H = 420, 420
    MAP_PAD = 24
    TRAIL_SECS = 5.0
    TRAIL_COLOR = (255, 200, 0)
    TRAIL_THICK = 2

    # mapping paths
    MAPX_PATH = r"D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy"
    MAPY_PATH = r"D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy"

    start_stimulus(
        IMG_FOLDER=IMG_FOLDER,
        PATTERN=PATTERN,
        NEAR_WHITE=NEAR_WHITE,
        SPRITE_MARGIN_PX=SPRITE_MARGIN_PX,
        BG_COLOR=BG_COLOR,
        FPS=FPS,
        WINDOW_W=WINDOW_W,
        WINDOW_H=WINDOW_H,
        ARENA_RADIUS_MM=ARENA_RADIUS_MM,
        CAMERA_X_MM=CAMERA_X_MM,
        CAMERA_Y_MM=CAMERA_Y_MM,
        REF_DIST_MM=REF_DIST_MM,
        DESIRED_PX_AT_REF=DESIRED_PX_AT_REF,
        MIN_SCALE=MIN_SCALE,
        MAX_SCALE=MAX_SCALE,
        VIRTUAL_CAMERA_OFFSET_MM=VIRTUAL_CAMERA_OFFSET_MM,
        GLOBAL_SIZE_SCALE=GLOBAL_SIZE_SCALE,
        SPEED_MM_S=SPEED_MM_S,
        BACK_MM_S=BACK_MM_S,
        TURN_DEG_S=TURN_DEG_S,
        STAND_TURN_MULT=STAND_TURN_MULT,
        START_POS=START_POS,
        START_HEADING_DEG=START_HEADING_DEG,
        WALK_TURN_NOISE_DEG_RMS=WALK_TURN_NOISE_DEG_RMS,
        WALK_TRANS_NOISE_MM_RMS=WALK_TRANS_NOISE_MM_RMS,
        CAMERA_Y_OFFSET_PX=CAMERA_Y_OFFSET_PX,
        MAP_W=MAP_W,
        MAP_H=MAP_H,
        MAP_PAD=MAP_PAD,
        TRAIL_SECS=TRAIL_SECS,
        TRAIL_COLOR=TRAIL_COLOR,
        TRAIL_THICK=TRAIL_THICK,
        MAPX_PATH=MAPX_PATH,
        MAPY_PATH=MAPY_PATH,
    )
