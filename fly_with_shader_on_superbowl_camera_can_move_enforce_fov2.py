import os, sys, math, time, re, glob, ctypes
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME, QUIT, KEYDOWN, K_ESCAPE
from OpenGL import GL

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# ----------------- CONFIG -----------------

# Turn this on to let the fly move by an internal behavioral algorithm (no WASD)
AUTOMATIC_FLY = True

# Path to precomputed projector→camera mapping (x-coordinates, float32 .npy)
MAPX_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy")
# Path to precomputed projector→camera mapping (y-coordinates, float32 .npy)
MAPY_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy")

# Folder containing the rendered fly sprite images used for the stimulus
IMG_FOLDER   = r"D:\virtual.fly\og_pics"
# Glob pattern to select the sprite frames inside IMG_FOLDER
PATTERN      = "fly*.png"
# Pixel intensity threshold for “white background” when auto-segmenting sprites (0–255)
NEAR_WHITE   = 245
# Extra pixels kept around the detected fly when cropping the sprite
SPRITE_MARGIN_PX = 2

# Background color for the camera image / stimulus canvas (B, G, R)
BG_COLOR = (255, 255, 255)
# Target main-loop rate in frames per second for the OpenGL / pygame loop
TARGET_FPS = 60

# Radius of the circular arena in millimetres (world coordinates)
ARENA_RADIUS_MM = 20
# Initial camera x-position in arena coordinates (mm)
CAMERA_X_MM     = 0.0
# Initial camera y-position in arena coordinates (mm)
CAMERA_Y_MM     = -ARENA_RADIUS_MM
# Vertical pixel offset of the rendered fly on the camera image (shifts the fly up/down)
CAMERA_Y_OFFSET_PX = 0

# Physical distance (mm) at which the fly sprite should appear with DESIRED_PX_AT_REF pixels height
REF_DIST_MM        = 220.0
# Desired sprite height in pixels when the fly is at REF_DIST_MM
DESIRED_PX_AT_REF  = 10
# Lower clamp on sprite scale factor (prevents vanishingly small sprites)
MIN_SCALE          = 0.0010
# Upper clamp on sprite scale factor (prevents absurdly large sprites)
MAX_SCALE          = 10
# Additional offset added to the distance before computing size (virtual camera position tweak)
VIRTUAL_CAMERA_OFFSET_MM = 0
# Global multiplicative factor on sprite size (quick overall size adjustment)
GLOBAL_SIZE_SCALE  = 1.0

# Forward walking speed of the fly in mm/s (W key or auto walk)
SPEED_MM_S       = 20
# Backward walking speed of the fly in mm/s (S key)
BACK_MM_S        = SPEED_MM_S * 0.64
# Baseline turning speed of the fly in degrees/s (A/D keys or auto turn)
TURN_DEG_S       = 200.0
# Multiplier applied to TURN_DEG_S when the fly is standing still (sharper turns on spot)
STAND_TURN_MULT  = 1.5
# Initial fly position in arena coordinates (x_mm, y_mm)
START_POS        = (0.0, 250.0)
# Initial fly heading in degrees (0° = facing +y in world coords, by our convention)
START_HEADING_DEG = 0

# camera motion (arrow keys)

# Forward/backward camera translation speed in mm/s (UP/DOWN)
CAMERA_SPEED_MM_S      = SPEED_MM_S
# Baseline camera turning speed in degrees/s (LEFT/RIGHT)
CAMERA_TURN_DEG_S      = TURN_DEG_S
# Turn multiplier when the camera is not translating (sharper rotation on spot)
CAMERA_STAND_TURN_MULT = 1.5

# Horizontal field of view of the virtual camera in degrees
CAMERA_FOV_X_DEG = 200

# RMS of random turning noise while the fly walks, in degrees per sqrt(second)
WALK_TURN_NOISE_DEG_RMS  = 20.0
# RMS of random translational jitter while the fly walks, in mm per sqrt(second)
WALK_TRANS_NOISE_MM_RMS  = 1

# Minimal forward distance in camera coordinates at which the fly is still considered visible (mm)
# If the fly is closer than this or has negative forward distance (behind camera), it is not drawn.
MIN_VIEW_Z_MM = 1.5  # behind or too-close to camera -> not visible

# minimap config

# Whether to show the separate OpenCV minimap window (True) or disable it (False)
SHOW_MINIMAP = True   # set to False to disable minimap window completely

# Size of the minimap window in pixels (width, height)
MAP_W, MAP_H = 420, 420
# Padding in pixels between the arena circle and the minimap window border
MAP_PAD      = 24
# Duration in seconds for which the fly trail is kept and drawn on the minimap
TRAIL_SECS   = 5.0
# Base color of the trajectory trail in the minimap (B, G, R)
TRAIL_COLOR  = (255, 200, 0)
# Line thickness of the trail in pixels
TRAIL_THICK  = 2
# Target refresh rate of the minimap in Hz (how often we redraw the minimap per second)
MINIMAP_HZ   = 60

# ----------------- SHADERS -----------------

VERT_SRC = r"""
#version 330 core

in vec2 in_pos;   // clip-space position (-1..1)
in vec2 in_uv;    // projector uv (0..1)
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAG_SRC = r"""
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_cam;   // camera image
uniform sampler2D u_warp;  // warp map (RG = camera UV)

void main() {
    vec2 cam_uv = texture(u_warp, v_uv).rg;

    // flip vertically; adjust if not needed
    cam_uv.y = 1.0 - cam_uv.y;

    if (cam_uv.x < 0.0 || cam_uv.y < 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = texture(u_cam, cam_uv).rgb;
    fragColor = vec4(col, 1.0);
}
"""

# ----------------- FLY / CAMERA HELPERS -----------------

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
    return True

def world_to_minimap(x_mm, y_mm, center_u, center_v, scale_px_per_mm):
    u = int(round(center_u + x_mm * scale_px_per_mm))
    v = int(round(center_v - y_mm * scale_px_per_mm))
    return u, v

def draw_arrow(img_bgr, center_u, center_v, heading_rad, size_px=18, color=(0, 120, 255)):
    angle_img = heading_rad
    tip = np.array([0, -size_px], dtype=np.float32)
    left = np.array([-size_px * 0.6, size_px * 0.7], dtype=np.float32)
    right = np.array([size_px * 0.6, size_px * 0.7], dtype=np.float32)
    R = np.array(
        [
            [math.cos(angle_img), -math.sin(angle_img)],
            [math.sin(angle_img),  math.cos(angle_img)],
        ],
        dtype=np.float32,
    )
    pts = np.stack([tip, left, right], axis=0) @ R.T
    pts[:, 0] += center_u
    pts[:, 1] += center_v
    pts_i = pts.astype(np.int32)
    cv2.fillConvexPoly(img_bgr, pts_i, color)
    cv2.polylines(img_bgr, [pts_i], True, (0, 0, 0), 1, cv2.LINE_AA)

def build_minimap_base(
    ARENA_RADIUS_MM,
    CAMERA_X_MM,  # unused now; camera drawn dynamically
    CAMERA_Y_MM,
    MAP_W,
    MAP_H,
    MAP_PAD,
    center_u,
    center_v,
    scale_px_per_mm,
):
    img = np.full((MAP_H, MAP_W, 3), 255, np.uint8)
    radius_px = int(round(ARENA_RADIUS_MM * scale_px_per_mm))
    cv2.circle(img, (center_u, center_v), radius_px, (0, 0, 0), 2)
    return img

def draw_minimap_dynamic(
    base_img,
    x,
    y,
    heading,
    trail_pts_uv,
    TRAIL_COLOR,
    TRAIL_THICK,
    center_u,
    center_v,
    scale_px_per_mm,
    fps_now,
    cam_x,
    cam_y,
    cam_heading,
):
    img = base_img.copy()

    # trail
    if len(trail_pts_uv) >= 2:
        now_color = np.array(TRAIL_COLOR, dtype=np.float32)
        nseg = len(trail_pts_uv) - 1
        for i in range(nseg):
            p0 = trail_pts_uv[i]
            p1 = trail_pts_uv[i + 1]
            alpha = (i + 1) / nseg
            col = tuple((now_color * (0.3 + 0.7 * alpha)).astype(np.int32).tolist())
            cv2.line(img, p0, p1, col, TRAIL_THICK, cv2.LINE_AA)

    # camera position in minimap coords
    cam_u, cam_v = world_to_minimap(cam_x, cam_y, center_u, center_v, scale_px_per_mm)

    # vision cone (use arena radius as max range)
    fov_rad = math.radians(CAMERA_FOV_X_DEG)
    half_fov = 0.5 * fov_rad
    cone_range_mm = ARENA_RADIUS_MM
    n_cone_samples = 24

    cone_pts = [(cam_u, cam_v)]
    for i in range(n_cone_samples + 1):
        a = cam_heading - half_fov + (2.0 * half_fov) * (i / n_cone_samples)
        wx = cam_x + cone_range_mm * math.sin(a)
        wy = cam_y + cone_range_mm * math.cos(a)
        u, v = world_to_minimap(wx, wy, center_u, center_v, scale_px_per_mm)
        cone_pts.append((u, v))

    cone_pts_np = np.array(cone_pts, dtype=np.int32)

    # transparent cone via overlay
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, cone_pts_np, (230, 230, 255))
    alpha = 0.35  # transparency
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.polylines(img, [cone_pts_np], True, (150, 150, 255), 1, cv2.LINE_AA)

    # fly arrow
    fu, fv = world_to_minimap(x, y, center_u, center_v, scale_px_per_mm)
    draw_arrow(img, fu, fv, heading, size_px=18, color=(0, 120, 255))

    # camera arrow on top of cone
    draw_arrow(img, cam_u, cam_v, cam_heading, size_px=16, color=(0, 0, 255))
    cv2.putText(
        img, "cam",
        (cam_u + 10, cam_v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 255), 1, cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"FPS: {fps_now:.1f}",
        (10, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img

# ----------------- SIMPLE BEHAVIORAL MODEL -----------------

def _sample_bout_duration(mode):
    # shorter bouts for more stop-go
    if mode == "rest":
        return np.random.exponential(0.8)      # short pauses
    if mode == "walk":
        return np.random.exponential(0.5)      # short walks
    if mode in ("turn_left", "turn_right"):
        return np.random.exponential(0.25)     # brief turns
    return 0.5

def _pick_next_mode(prev_mode):
    r = np.random.rand()
    if prev_mode == "rest":
        # often start a short walk, sometimes stay resting, sometimes turn
        if r < 0.55:
            return "walk"
        elif r < 0.75:
            return "rest"
        elif r < 0.87:
            return "turn_left"
        else:
            return "turn_right"
    elif prev_mode == "walk":
        # strong tendency to stop or reorient after a brief walk
        if r < 0.30:
            return "walk"
        elif r < 0.70:
            return "rest"
        elif r < 0.85:
            return "turn_left"
        else:
            return "turn_right"
    else:  # from a turn
        if r < 0.60:
            return "rest"
        else:
            return "walk"

def step_fly_auto(x, y, heading, state, dt):
    """
    Bout-based locomotor model in a circular arena.

    state: dict with keys:
      - mode: "rest", "walk", "turn_left", "turn_right"
      - t_in_state: time spent in current state
      - state_dur: duration of current state
    """
    if state is None or "mode" not in state:
        state = {
            "mode": "rest",
            "t_in_state": 0.0,
            "state_dur": _sample_bout_duration("rest"),
        }

    mode = state["mode"]
    t_in_state = state["t_in_state"] + dt
    state_dur = state["state_dur"]

    # state transitions
    if t_in_state >= state_dur:
        mode = _pick_next_mode(mode)
        t_in_state = 0.0
        state_dur = _sample_bout_duration(mode)

    # dynamics
    turn_rate_rad = math.radians(TURN_DEG_S)
    v_base = SPEED_MM_S

    if mode == "rest":
        # almost stationary, small jitter in heading only
        heading += math.radians(5.0) * np.random.normal(0.0, math.sqrt(dt))
    elif mode == "walk":
        # brief walks with some speed jitter
        v = v_base * np.clip(np.random.normal(1.0, 0.25), 0.2, 1.8)
        # occasional micro-stop inside a walk bout
        if np.random.rand() < 0.1 * dt:  # ~0.1 events per second
            v *= 0.2
        heading += math.radians(WALK_TURN_NOISE_DEG_RMS) * np.random.normal(0.0, math.sqrt(dt))
        x += v * math.sin(heading) * dt
        y += v * math.cos(heading) * dt
        x += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))
        y += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))
    elif mode == "turn_left":
        heading -= turn_rate_rad * dt
    elif mode == "turn_right":
        heading += turn_rate_rad * dt

    # soft wall avoidance: bias heading toward arena center near boundary
    r = math.hypot(x, y)
    if r > 0.7 * ARENA_RADIUS_MM:
        target = math.atan2(-x, -y)  # direction pointing inward
        delta = (target - heading + math.pi) % (2 * math.pi) - math.pi
        wall_gain = 2.0 * (r / ARENA_RADIUS_MM)
        heading += wall_gain * delta * dt

    # keep inside strict circle (hard clamp)
    r_center = math.hypot(x, y)
    if r_center > ARENA_RADIUS_MM:
        scale_back = ARENA_RADIUS_MM / r_center
        x *= scale_back
        y *= scale_back

    state["mode"] = mode
    state["t_in_state"] = t_in_state
    state["state_dur"] = state_dur
    return x, y, heading, state

# ----------------- GL HELPERS -----------------

def compile_shader(source, shader_type):
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not status:
        log = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader

def create_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    status = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not status:
        log = GL.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog

def create_texture_2d(width, height, internal_format, fmt, dtype,
                      min_filter=GL.GL_LINEAR, mag_filter=GL.GL_LINEAR):
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, min_filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, mag_filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        internal_format,
        width,
        height,
        0,
        fmt,
        dtype,
        None
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex

def pick_rightmost_monitor(default_w, default_h):
    if get_monitors:
        try:
            mons = get_monitors()
        except Exception:
            mons = None
        if mons:
            return max(mons, key=lambda m: m.x)
    class M: pass
    m = M()
    m.x = 0; m.y = 0
    m.width = default_w; m.height = default_h
    return m

def request_gl_context(window_w, window_h, borderless=True):
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    flags = DOUBLEBUF | OPENGL
    if borderless:
        flags |= NOFRAME
    pygame.display.set_mode((window_w, window_h), flags)
    major = GL.glGetIntegerv(GL.GL_MAJOR_VERSION)
    minor = GL.glGetIntegerv(GL.GL_MINOR_VERSION)
    if major < 3 or (major == 3 and minor < 3):
        raise RuntimeError(f"OpenGL {major}.{minor} too old (need >= 3.3)")

# ----------------- WARP LOADING -----------------

def load_warp(mapx_path: Path, mapy_path: Path, factor: float = 2.5):
    mapx = np.load(str(mapx_path)).astype(np.float32)
    mapy = np.load(str(mapy_path)).astype(np.float32)
    print(mapx.shape, mapy.shape)
    mapx = mapx / factor
    mapy = mapy / factor
    print(mapx.shape, mapy.shape)

    # if you flipped for CPU version, keep consistent:
    mapx = np.flipud(mapx)
    mapy = np.flipud(mapy)

    proj_h, proj_w = mapx.shape
    print(f"Loaded warp map: projector size = {proj_w} x {proj_h}")
    valid = np.isfinite(mapx) & np.isfinite(mapy) & (mapx >= 0) & (mapy >= 0)
    if not np.any(valid):
        raise RuntimeError("mapx/mapy contain no valid entries")

    cam_w = int(np.ceil(mapx[valid].max())) + 1
    cam_h = int(np.ceil(mapy[valid].max())) + 1
    print(f"Loaded warp map: projector size = {cam_w} x {cam_h}")

    warp = np.zeros((proj_h, proj_w, 2), dtype=np.float32)
    warp[..., 0] = mapx / float(cam_w)
    warp[..., 1] = mapy / float(cam_h)
    warp[~valid, 0] = -1.0
    warp[~valid, 1] = -1.0
    return warp, cam_w, cam_h, proj_w, proj_h

# ----------------- MAIN -----------------

def main():
    warp, cam_w, cam_h, proj_w, proj_h = load_warp(MAPX_PATH, MAPY_PATH)

    mon = pick_rightmost_monitor(proj_w, proj_h)
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon.x},{mon.y}"

    pygame.init()
    request_gl_context(proj_w, proj_h, borderless=True)
    pygame.display.set_caption("GPU-warped WASD/auto fly")

    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glViewport(0, 0, proj_w, proj_h)
    GL.glDisable(GL.GL_DEPTH_TEST)

    prog = create_program(VERT_SRC, FRAG_SRC)
    GL.glUseProgram(prog)

    quad_vertices = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype=np.float32)

    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL.GL_STATIC_DRAW)
    stride = 4 * quad_vertices.itemsize
    loc_pos = GL.glGetAttribLocation(prog, "in_pos")
    loc_uv  = GL.glGetAttribLocation(prog, "in_uv")
    GL.glEnableVertexAttribArray(loc_pos)
    GL.glVertexAttribPointer(loc_pos, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(loc_uv)
    GL.glVertexAttribPointer(loc_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * quad_vertices.itemsize))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    cam_tex = create_texture_2d(cam_w, cam_h, GL.GL_RGB8, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    warp_tex = create_texture_2d(proj_w, proj_h, GL.GL_RG32F, GL.GL_RG, GL.GL_FLOAT)

    u_cam_loc  = GL.glGetUniformLocation(prog, "u_cam")
    u_warp_loc = GL.glGetUniformLocation(prog, "u_warp")
    GL.glUseProgram(prog)
    GL.glUniform1i(u_cam_loc, 0)
    GL.glUniform1i(u_warp_loc, 1)

    warp_c = np.ascontiguousarray(warp.astype(np.float32))
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, warp_tex)
    warp_ptr = warp_c.ctypes.data_as(ctypes.c_void_p)
    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        proj_w,
        proj_h,
        GL.GL_RG,
        GL.GL_FLOAT,
        warp_ptr,
    )

    # horizontal FOV in radians for angle-based projection
    FOV_X_RAD = math.radians(CAMERA_FOV_X_DEG)

    # sprites
    sprites, masks = load_and_crop_sorted(IMG_FOLDER, PATTERN, NEAR_WHITE, SPRITE_MARGIN_PX)
    sprites = [np.flipud(x) for x in sprites]
    ref_h = sprites[0].shape[0]
    n_sprites = len(sprites)

    # fly state
    x, y = START_POS
    heading = math.radians(START_HEADING_DEG)
    behavior_state = None  # for AUTOMATIC_FLY

    # camera state (virtual, moves with arrows)
    camera_x = CAMERA_X_MM
    camera_y = CAMERA_Y_MM
    cam_heading = math.radians(START_HEADING_DEG)

    speed_f = SPEED_MM_S
    speed_b = BACK_MM_S
    turn_rate_rad = math.radians(TURN_DEG_S)
    cam_turn_rate_rad = math.radians(CAMERA_TURN_DEG_S)

    trail = deque()

    # minimap geometry
    R = ARENA_RADIUS_MM
    r_cam_init = math.hypot(CAMERA_X_MM, CAMERA_Y_MM)
    half_h = (MAP_H - 2 * MAP_PAD) / 2.0
    half_w = (MAP_W - 2 * MAP_PAD) / 2.0
    span_vert_mm = 2 * R if r_cam_init <= 0 else R + r_cam_init
    scale_vert = (MAP_H - 2 * MAP_PAD) / span_vert_mm
    scale_horiz = half_w / R
    scale_px_per_mm = min(scale_vert, scale_horiz)
    center_u = MAP_W // 2
    center_v = MAP_PAD + int(round(R * scale_px_per_mm))

    minimap_base = None
    if SHOW_MINIMAP:
        cv2.namedWindow("minimap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("minimap", MAP_W, MAP_H)
        cv2.moveWindow("minimap", 50, 50)
        minimap_base = build_minimap_base(
            ARENA_RADIUS_MM,
            CAMERA_X_MM,
            CAMERA_Y_MM,
            MAP_W,
            MAP_H,
            MAP_PAD,
            center_u,
            center_v,
            scale_px_per_mm,
        )

    cam_img = np.full((cam_h, cam_w, 3), BG_COLOR, np.uint8)
    cam_img = np.ascontiguousarray(cam_img, dtype=np.uint8)

    clock = pygame.time.Clock()
    running = True
    next_minimap_t = 0.0

    cx_cam = cam_w // 2

    # NEW: pause flag for automatic fly
    auto_paused = False

    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        fps_now = clock.get_fps()
        now = time.time()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # toggle pause for automatic fly
                elif event.key == pygame.K_p and AUTOMATIC_FLY:
                    auto_paused = not auto_paused

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

        # -------- fly controls --------
        if AUTOMATIC_FLY:
            if not auto_paused:
                x, y, heading, behavior_state = step_fly_auto(
                    x, y, heading, behavior_state, dt
                )
        else:
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

            # keep fly inside arena (manual control)
            r_center = math.hypot(x, y)
            if r_center > ARENA_RADIUS_MM:
                scale_back = ARENA_RADIUS_MM / r_center
                x *= scale_back
                y *= scale_back

        # -------- camera controls (arrow keys) --------
        cam_moving = keys[pygame.K_UP] or keys[pygame.K_DOWN]
        cam_current_turn_rate = cam_turn_rate_rad * (CAMERA_STAND_TURN_MULT if not cam_moving else 1.0)

        if keys[pygame.K_LEFT]:
            cam_heading -= cam_current_turn_rate * dt
        if keys[pygame.K_RIGHT]:
            cam_heading += cam_current_turn_rate * dt

        if keys[pygame.K_UP]:
            camera_x += CAMERA_SPEED_MM_S * math.sin(cam_heading) * dt
            camera_y += CAMERA_SPEED_MM_S * math.cos(cam_heading) * dt
        if keys[pygame.K_DOWN]:
            camera_x -= CAMERA_SPEED_MM_S * math.sin(cam_heading) * dt
            camera_y -= CAMERA_SPEED_MM_S * math.cos(cam_heading) * dt

        # keep camera inside arena
        r_cam_center = math.hypot(camera_x, camera_y)
        if r_cam_center > ARENA_RADIUS_MM:
            scale_back_cam = ARENA_RADIUS_MM / r_cam_center
            camera_x *= scale_back_cam
            camera_y *= scale_back_cam

        # trail for fly
        if SHOW_MINIMAP:
            trail.append((now, x, y))
            expire_before = now - TRAIL_SECS
            while trail and trail[0][0] < expire_before:
                trail.popleft()

        # -------- render fly into cam_img (camera-relative perspective) --------
        cam_img[:] = BG_COLOR

        # world -> camera local coords
        dx_world = x - camera_x
        dy_world = y - camera_y
        sin_ch = math.sin(cam_heading)
        cos_ch = math.cos(cam_heading)

        # right axis = (cos_ch, -sin_ch), forward axis = (sin_ch, cos_ch)
        x_cam_local = dx_world * cos_ch - dy_world * sin_ch   # right (+), left (-)
        z_cam_local = dx_world * sin_ch + dy_world * cos_ch   # forward (+)

        # horizontal angle of fly relative to camera forward
        theta = math.atan2(x_cam_local, z_cam_local)  # 0 = straight ahead, +right, -left
        half_fov = 0.5 * FOV_X_RAD

        if (z_cam_local > MIN_VIEW_Z_MM) and (abs(theta) <= half_fov):
            # map theta in [-half_fov, +half_fov] to screen x in [0, cam_w]
            norm = theta / half_fov    # -1..1
            proj_x = cx_cam + norm * (cam_w * 0.5)
            px_cam = int(round(proj_x))
            py_cam = cam_h // 2 + CAMERA_Y_OFFSET_PX

            # use true distance from camera, not forward component
            dist_mm = math.hypot(x_cam_local, z_cam_local)
            # avoid pathological scaling when very close
            dist_mm = max(dist_mm, 5.0)

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

            # sprite orientation: relative to camera heading
            rel_heading = heading - cam_heading
            yaw_deg = (math.degrees(rel_heading)) % 360.0
            idx = angle_to_index(yaw_deg)
            idx = min(idx, n_sprites - 1)

            # approximate half sprite width in pixels at this scale
            base_w = sprites[idx].shape[1]
            half_w_scaled = 0.5 * base_w * scale

            # quick cull: only skip if the whole sprite is outside horizontally
            if px_cam + half_w_scaled >= 0 and px_cam - half_w_scaled <= cam_w:
                render_sprite_masked_bgr(cam_img, sprites[idx], masks[idx], (px_cam, py_cam), scale)
        # else: fly behind / too close / outside FOV -> not rendered

        # upload cam_img to GPU
        shifted = np.ascontiguousarray(cam_img, dtype=np.uint8)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, cam_tex)
        shift_ptr = shifted.ctypes.data_as(ctypes.c_void_p)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            cam_w,
            cam_h,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            shift_ptr,
        )

        GL.glViewport(0, 0, proj_w, proj_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(prog)
        GL.glBindVertexArray(vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, cam_tex)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, warp_tex)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        pygame.display.flip()

        if SHOW_MINIMAP and now >= next_minimap_t:
            next_minimap_t = now + 1.0 / MINIMAP_HZ
            trail_uv = [
                world_to_minimap(xi, yi, center_u, center_v, scale_px_per_mm)
                for _, xi, yi in trail
            ]
            map_img = draw_minimap_dynamic(
                minimap_base,
                x,
                y,
                heading,
                trail_uv,
                TRAIL_COLOR,
                TRAIL_THICK,
                center_u,
                center_v,
                scale_px_per_mm,
                fps_now,
                camera_x,
                camera_y,
                cam_heading,
            )
            cv2.imshow("minimap", map_img)
            cv2.waitKey(1)

    GL.glDeleteTextures([cam_tex, warp_tex])
    GL.glDeleteBuffers(1, [vbo])
    GL.glDeleteVertexArrays(1, [vao])
    GL.glDeleteProgram(prog)
    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
