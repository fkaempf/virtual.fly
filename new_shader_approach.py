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

MAPX_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy")
MAPY_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy")

IMG_FOLDER   = r"D:\virtual.fly\og_pics"
PATTERN      = "fly*.png"
NEAR_WHITE   = 245
SPRITE_MARGIN_PX = 2

BG_COLOR = (255, 255, 255)
TARGET_FPS = 120.0

ARENA_RADIUS_MM = 200.0
CAMERA_X_MM     = 0.0
CAMERA_Y_MM     = -ARENA_RADIUS_MM * 1.3

REF_DIST_MM        = 220.0
DESIRED_PX_AT_REF  = 260.0
MIN_SCALE          = 0.10
MAX_SCALE          = 7.0
VIRTUAL_CAMERA_OFFSET_MM = 0.0
GLOBAL_SIZE_SCALE  = 1.0

SPEED_MM_S       = 140.0
BACK_MM_S        = 90.0
TURN_DEG_S       = 200.0
STAND_TURN_MULT  = 1.5
START_POS        = (0.0, 250.0)
START_HEADING_DEG = 180.0

WALK_TURN_NOISE_DEG_RMS  = 20.0
WALK_TRANS_NOISE_MM_RMS  = 5.0
CAMERA_Y_OFFSET_PX       = 100

SHOW_MINIMAP = True  # set False to disable minimap window + trail computations

MAP_W, MAP_H = 420, 420
MAP_PAD      = 24
TRAIL_SECS   = 5.0
TRAIL_COLOR  = (255, 200, 0)
TRAIL_THICK  = 2
MINIMAP_HZ   = 10.0

# ----------------- SHADERS -----------------

VERT_SRC = r"""
#version 330 core

layout(location = 0) in vec2 in_pos;   // clip-space position (-1..1)
layout(location = 1) in vec2 in_uv;    // projector uv (0..1)
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

    // keep consistent with earlier CPU version
    cam_uv.y = 1.0 - cam_uv.y;

    if (cam_uv.x < 0.0 || cam_uv.y < 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = texture(u_cam, cam_uv).rgb;
    fragColor = vec4(col, 1.0);
}
"""

SPRITE_VERT_SRC = r"""
#version 330 core

layout(location = 0) in vec2 in_pos;   // local quad coords in [-1, 1]
layout(location = 1) in vec2 in_uv;    // sprite UV in [0, 1]
out vec2 v_uv;

uniform vec2 u_center_ndc;     // center in NDC
uniform vec2 u_half_size_ndc;  // half-size in NDC
uniform float u_angle;         // rotation in radians, +CCW

void main() {
    vec2 local = in_pos * u_half_size_ndc;
    float c = cos(u_angle);
    float s = sin(u_angle);
    mat2 R = mat2(c, -s,
                  s,  c);
    vec2 rotated = R * local;
    vec2 pos = u_center_ndc + rotated;
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

SPRITE_FRAG_SRC = r"""
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_sprite;
uniform vec3 u_bg;

void main() {
    vec4 tex = texture(u_sprite, v_uv);
    vec3 col = mix(u_bg, tex.rgb, tex.a);
    fragColor = vec4(col, 1.0);
}
"""

# ----------------- FLY HELPERS -----------------

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

def upload_sprite_textures(sprites, masks):
    tex_ids = []
    for spr, m in zip(sprites, masks):
        # spr: BGR, top-left origin. Do NOT flip; rotation is encoded in sprite set.
        rgb = spr[..., ::-1]
        alpha = (m * 255).astype(np.uint8)
        rgba = np.dstack([rgb, alpha]).astype(np.uint8)
        rgba_c = np.ascontiguousarray(rgba)

        h, w = rgba_c.shape[:2]
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            w,
            h,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            rgba_c.ctypes.data_as(ctypes.c_void_p),
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        tex_ids.append((tex, w, h))
    return tex_ids

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

def world_to_minimap(x_mm, y_mm, center_u, center_v, scale_px_per_mm):
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
    pts_i = pts.astype(np.int32)
    cv2.fillConvexPoly(img_bgr, pts_i, color)
    cv2.polylines(img_bgr, [pts_i], True, (0, 0, 0), 1, cv2.LINE_AA)

def build_minimap_base(
    ARENA_RADIUS_MM,
    CAMERA_X_MM,
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
    cam_u, cam_v = world_to_minimap(CAMERA_X_MM, CAMERA_Y_MM, center_u, center_v, scale_px_per_mm)
    cv2.circle(img, (cam_u, cam_v), 6, (0, 0, 255), -1)
    cv2.putText(
        img, "cam",
        (cam_u + 10, cam_v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 255), 1, cv2.LINE_AA,
    )
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
):
    img = base_img.copy()
    if len(trail_pts_uv) >= 2:
        now_color = np.array(TRAIL_COLOR, dtype=np.float32)
        nseg = len(trail_pts_uv) - 1
        for i in range(nseg):
            p0 = trail_pts_uv[i]
            p1 = trail_pts_uv[i + 1]
            alpha = (i + 1) / nseg
            col = tuple((now_color * (0.3 + 0.7 * alpha)).astype(np.int32).tolist())
            cv2.line(img, p0, p1, col, TRAIL_THICK, cv2.LINE_AA)
    fu, fv = world_to_minimap(x, y, center_u, center_v, scale_px_per_mm)
    draw_fly_triangle(img, fu, fv, heading, size_px=18)
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

def load_warp(mapx_path: Path, mapy_path: Path):
    mapx = np.load(str(mapx_path)).astype(np.float32)
    mapy = np.load(str(mapy_path)).astype(np.float32)

    # keep consistent with earlier GPU/CPU mix
    mapx = np.flipud(mapx)
    mapy = np.flipud(mapy)

    proj_h, proj_w = mapx.shape
    valid = np.isfinite(mapx) & np.isfinite(mapy) & (mapx >= 0) & (mapy >= 0)
    if not np.any(valid):
        raise RuntimeError("mapx/mapy contain no valid entries")

    cam_w = int(np.ceil(mapx[valid].max())) + 1
    cam_h = int(np.ceil(mapy[valid].max())) + 1

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
    pygame.display.set_caption("GPU-warped WASD fly (GPU sprite + FBO)")

    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glDisable(GL.GL_DEPTH_TEST)

    # ---- warp program + fullscreen quad ----
    warp_prog = create_program(VERT_SRC, FRAG_SRC)
    quad_vertices = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype=np.float32)

    vao_warp = GL.glGenVertexArrays(1)
    vbo_warp = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao_warp)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_warp)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL.GL_STATIC_DRAW)
    stride = 4 * quad_vertices.itemsize
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * quad_vertices.itemsize))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    # ---- camera texture + FBO ----
    cam_tex = create_texture_2d(cam_w, cam_h, GL.GL_RGBA8, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    cam_fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cam_fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D,
        cam_tex,
        0
    )
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Camera FBO incomplete: {status}")
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ---- warp texture ----
    warp_tex = create_texture_2d(
        proj_w, proj_h, GL.GL_RG32F, GL.GL_RG, GL.GL_FLOAT,
        min_filter=GL.GL_NEAREST, mag_filter=GL.GL_NEAREST
    )
    warp_c = np.ascontiguousarray(warp.astype(np.float32))
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, warp_tex)
    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        proj_w,
        proj_h,
        GL.GL_RG,
        GL.GL_FLOAT,
        warp_c.ctypes.data_as(ctypes.c_void_p),
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    # ---- set sampler uniforms in warp shader ----
    GL.glUseProgram(warp_prog)
    u_cam_loc  = GL.glGetUniformLocation(warp_prog, "u_cam")
    u_warp_loc = GL.glGetUniformLocation(warp_prog, "u_warp")
    GL.glUniform1i(u_cam_loc, 0)
    GL.glUniform1i(u_warp_loc, 1)
    GL.glUseProgram(0)

    # ---- sprites -> GL textures ----
    sprites, masks = load_and_crop_sorted(IMG_FOLDER, PATTERN, NEAR_WHITE, SPRITE_MARGIN_PX)
    ref_h = sprites[0].shape[0]
    sprite_textures = upload_sprite_textures(sprites, masks)
    n_sprites = len(sprite_textures)

    # ---- sprite program + quad ----
    sprite_prog = create_program(SPRITE_VERT_SRC, SPRITE_FRAG_SRC)
    sprite_vertices = quad_vertices.copy()
    vao_sprite = GL.glGenVertexArrays(1)
    vbo_sprite = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao_sprite)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_sprite)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, sprite_vertices.nbytes, sprite_vertices, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * sprite_vertices.itemsize))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    u_center_loc    = GL.glGetUniformLocation(sprite_prog, "u_center_ndc")
    u_half_size_loc = GL.glGetUniformLocation(sprite_prog, "u_half_size_ndc")
    u_angle_loc     = GL.glGetUniformLocation(sprite_prog, "u_angle")
    u_bg_loc        = GL.glGetUniformLocation(sprite_prog, "u_bg")
    u_sprite_loc    = GL.glGetUniformLocation(sprite_prog, "u_sprite")

    bg_norm = np.array(BG_COLOR, dtype=np.float32) / 255.0
    GL.glUseProgram(sprite_prog)
    GL.glUniform1i(u_sprite_loc, 0)
    GL.glUniform3f(u_bg_loc, float(bg_norm[0]), float(bg_norm[1]), float(bg_norm[2]))
    GL.glUseProgram(0)

    # ---- minimap geometry ----
    R = ARENA_RADIUS_MM
    r_cam = math.hypot(CAMERA_X_MM, CAMERA_Y_MM)
    half_h = (MAP_H - 2 * MAP_PAD) / 2.0
    half_w = (MAP_W - 2 * MAP_PAD) / 2.0
    span_vert_mm = 2 * R if r_cam <= 0 else R + r_cam
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

    # ---- simulation state ----
    x, y = START_POS
    heading = math.radians(START_HEADING_DEG)
    speed_f = SPEED_MM_S
    speed_b = BACK_MM_S
    turn_rate_rad = math.radians(TURN_DEG_S)
    px_per_mm = cam_w / (2 * ARENA_RADIUS_MM)
    trail = deque() if SHOW_MINIMAP else None
    next_minimap_t = 0.0

    rng = np.random.default_rng()

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        fps_now = clock.get_fps()
        now = time.time()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN and event.key == K_ESCAPE:
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
            sigma = math.sqrt(dt)
            heading += math.radians(WALK_TURN_NOISE_DEG_RMS) * rng.normal(0.0, sigma)
            x += WALK_TRANS_NOISE_MM_RMS * rng.normal(0.0, sigma)
            y += WALK_TRANS_NOISE_MM_RMS * rng.normal(0.0, sigma)

        r_center = math.hypot(x, y)
        if r_center > ARENA_RADIUS_MM:
            scale_back = ARENA_RADIUS_MM / r_center
            x *= scale_back
            y *= scale_back

        if SHOW_MINIMAP:
            trail.append((now, x, y))
            expire_before = now - TRAIL_SECS
            while trail and trail[0][0] < expire_before:
                trail.popleft()

        # ---- camera-space position in pixels ----
        cx_cam = cam_w // 2
        px_cam = int(round(cx_cam + x * px_per_mm))
        py_cam = cam_h // 2 + CAMERA_Y_OFFSET_PX

        # orientation -> sprite index
        yaw_deg = (math.degrees(heading)) % 360.0
        idx = angle_to_index(yaw_deg)
        idx = min(idx, n_sprites - 1)
        tex_id, spr_w, spr_h = sprite_textures[idx]

        # distance for size scaling
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

        draw_w = max(1, int(round(spr_w * scale)))
        draw_h = max(1, int(round(spr_h * scale)))

        # pixel -> NDC mapping
        center_x_ndc = 2.0 * (px_cam / cam_w) - 1.0
        center_y_ndc = 1.0 - 2.0 * (py_cam / cam_h)
        half_w_ndc = draw_w / cam_w
        half_h_ndc = draw_h / cam_h

        # ---- camera pass: draw sprite into cam_tex via FBO ----
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cam_fbo)
        GL.glViewport(0, 0, cam_w, cam_h)
        GL.glClearColor(float(bg_norm[0]), float(bg_norm[1]), float(bg_norm[2]), 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(sprite_prog)
        GL.glUniform2f(u_center_loc, center_x_ndc, center_y_ndc)
        GL.glUniform2f(u_half_size_loc, half_w_ndc, half_h_ndc)
        # no extra rotation; heading already encoded by sprite index
        GL.glUniform1f(u_angle_loc, 0.0)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

        GL.glBindVertexArray(vao_sprite)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # ---- warp pass: projector ----
        GL.glViewport(0, 0, proj_w, proj_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(warp_prog)
        GL.glBindVertexArray(vao_warp)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, cam_tex)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, warp_tex)

        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        pygame.display.flip()

        # ---- minimap ----
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
            )
            cv2.imshow("minimap", map_img)
            cv2.waitKey(1)

    # ---- cleanup ----
    for tex_id, _, _ in sprite_textures:
        GL.glDeleteTextures([tex_id])
    GL.glDeleteTextures([cam_tex, warp_tex])
    GL.glDeleteFramebuffers(1, [cam_fbo])
    GL.glDeleteBuffers(1, [vbo_warp, vbo_sprite])
    GL.glDeleteVertexArrays(1, [vao_warp, vao_sprite])
    GL.glDeleteProgram(warp_prog)
    GL.glDeleteProgram(sprite_prog)

    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
