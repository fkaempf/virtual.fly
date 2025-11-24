"""
Configuration / input knobs (all module-level):
- is_mac: platform check used to pick default file paths for calibration and model assets.
- MAPX_PATH, MAPY_PATH: numpy float32 warp maps (projector->camera UV) that set the projection layout.
- FLY_MODEL_PATH: GLB model of the fly to render.
- FLY_PHYS_LENGTH_MM: physical target length (mm) to which the mesh longest dimension is scaled.
- FLY_BASE_SCALE: extra scalar multiplier applied after physical scaling to make the fly larger/smaller.
- FLY_MODEL_YAW_OFFSET_DEG: degrees to rotate the model so its nose faces forward in world space.
- BG_COLOR: background color for the offscreen fly render (BGR tuple 0-255).
- TARGET_FPS: Pygame tick target for timing and minimap text.
- ARENA_RADIUS_MM: radius of the circular arena; constrains fly/camera positions and minimap scale.
- CAMERA_X_MM, CAMERA_Y_MM, CAM_HEIGHT_MM: camera position and height in mm; camera looks along its heading on the XZ plane.
- SPEED_MM_S, BACK_MM_S: fly forward/back speeds.
- TURN_DEG_S, STAND_TURN_MULT: fly turn rate (deg/s) and a multiplier applied when the fly is stationary.
- START_POS, START_HEADING_DEG: initial fly position and heading (degrees).
- CAMERA_SPEED_MM_S, CAMERA_TURN_DEG_S, CAMERA_STAND_TURN_MULT: camera translation/turn rates and turn multiplier when camera is stationary.
- CAMERA_FOV_X_DEG: arena/minimap FOV for the 2D cone visualization (later mirrored to fly cam FOV X).
- FLY_CAM_FOV_Y_DEG, FLY_CAM_FOV_X_DEG: 3D fly camera vertical/horizontal FOV; if X is set, Y is derived from aspect.
- FLY_CAM_ALLOW_ULTRAWIDE: allows clamping near 180 deg without blowing up tan() in perspective.
- FLY_CAM_PROJECTION: projection for the fly's view; use 'equirect' or 'equidistant' for spherical sampling.
- FLY_CAM_FLIP_MODEL_FOR_ULTRAWIDE: flips mesh about X when using spherical maps; set False to avoid mirrored turns.
- WALK_TURN_NOISE_DEG_RMS, WALK_TRANS_NOISE_MM_RMS: per-axis motion noise applied while the fly moves.
- MIN_VIEW_DIST_MM: reserved minimum view distance (currently unused).
- SHOW_MINIMAP: toggle for drawing the 2D minimap window.
- MAP_W, MAP_H, MAP_PAD: minimap window size and padding.
- TRAIL_SECS, TRAIL_COLOR, TRAIL_THICK, MINIMAP_HZ: trail length, style, thickness, and refresh rate of the minimap.
- USE_AUTOMATIC_FLY: True enables autonomous walk/run state machine; False uses keyboard control (WASD).
- AUTO_MEAN_RUN_DUR, AUTO_MEAN_PAUSE_DUR: exponential mean durations for the autonomous run/pause states.
- AUTO_TURN_STD_DEG, AUTO_EDGE_TURN_DEG: autonomous turn noise std dev and stronger edge avoidance turn.
- AUTO_EDGE_THRESH: radius threshold (fraction of arena) to begin edge avoidance behavior.
"""

import os, sys, math, time, ctypes, base64
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME, QUIT, KEYDOWN, K_ESCAPE
from OpenGL import GL
from pygltflib import GLTF2  # pip install pygltflib

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# ----------------- CONFIG -----------------
is_mac = sys.platform == "darwin"
print("is_mac:", is_mac)

# Path to precomputed projector→camera mapping (x/y coords, float32 .npy)
if is_mac:
    MAPX_PATH = Path("/Users/fkampf/Documents/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy")
    MAPY_PATH = Path("/Users/fkampf/Documents/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy")
else:
    MAPX_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy")
    MAPY_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy")

# 3D fly model path (GLB from Blender)
if is_mac:
    FLY_MODEL_PATH = Path("/Users/fkampf/Documents/virtual.fly/femalefly.glb")
else:
    FLY_MODEL_PATH = Path(r"D:\virtual.fly\fly.glb")

# target physical length of fly (longest dimension of mesh) in arena mm
FLY_PHYS_LENGTH_MM = 3.0
FLY_BASE_SCALE     = 1  # extra multiplier if you want it bigger/smaller later
FLY_MODEL_YAW_OFFSET_DEG = 0  # rotate model so its nose faces motion direction

SCREEN_DISTANCE_MM       = 40.0  # fixed physical eye→screen distance (mm)
FLY_APPARENT_DISTANCE_MM = 10.0  # perceived distance target (mm) for angular size
DIST_SCALE_SMOOTH_HZ     = 8.0   # smoothing rate for scale to avoid pops

FLY_BODY_RADIUS_MM = 0.5 * FLY_PHYS_LENGTH_MM * FLY_BASE_SCALE
CAM_BODY_RADIUS_MM = FLY_BODY_RADIUS_MM  # treat camera as a second fly for overlap avoidance
MIN_CAM_FLY_DIST_MM = FLY_BODY_RADIUS_MM + CAM_BODY_RADIUS_MM
MIN_CAM_FLY_DIST_MM = 1.5
MIN_DIST_ADJ_STEP_MM = 0.5  # step for live min-distance tuning (keys -/=)

BG_COLOR = (255, 255, 255)  # BGR
TARGET_FPS = 60

# Lighting (four directional lights from above: N, E, S, W)
LIGHT_AMBIENT = 0.6              # base ambient multiplier
LIGHT_INTENSITIES = [2.0, 2.0, 2.0, 2.0]  # strengths for N, E, S, W
LIGHT_ELEVATION_DEG = 65.0       # elevation angle above horizon (0=horizon, 90=down)
LIGHT_MAX_GAIN = 4.0             # clamp on total light gain

ARENA_RADIUS_MM = 40
CAMERA_X_MM     = 0.0
CAMERA_Y_MM     = -ARENA_RADIUS_MM
CAM_HEIGHT_MM   = 0.89

SPEED_MM_S       = 20
BACK_MM_S        = SPEED_MM_S * 0.64
TURN_DEG_S       = 200
STAND_TURN_MULT  = 1.5

START_POS         = (0, 0)
START_HEADING_DEG = 0.0

CAMERA_SPEED_MM_S      = SPEED_MM_S
CAMERA_TURN_DEG_S      = TURN_DEG_S
CAMERA_STAND_TURN_MULT = STAND_TURN_MULT
CAMERA_Z_SPEED_MM_S    = SPEED_MM_S
YAW_ADJ_STEP_DEG       = 5.0  # step size for live yaw offset tuning (keys 9/0)
HEIGHT_ADJ_STEP_MM     = 0.5  # step size for live height tuning when tapping the height keys


# 3D fly camera FOV (sane perspective, independent of arena FOV)
# If FLY_CAM_FOV_X_DEG is not None, vertical FOV will be derived from it and the camera aspect.
FLY_CAM_FOV_Y_DEG = 60.0
FLY_CAM_FOV_X_DEG = 200
CAMERA_FOV_X_DEG = FLY_CAM_FOV_X_DEG
FLY_CAM_ALLOW_ULTRAWIDE = True  # clamp safely if FOV hits 180+
FLY_CAM_PROJECTION = "equirect"  # perspective | equidistant | equirect
FLY_CAM_FLIP_MODEL_FOR_ULTRAWIDE = True  # flip about X for spherical projection to keep fly upright

WALK_TURN_NOISE_DEG_RMS  = 20.0
WALK_TRANS_NOISE_MM_RMS  = 1.0



MIN_VIEW_DIST_MM = 2.5  # not explicitly used now

SHOW_MINIMAP = True
MAP_W, MAP_H = 420, 420
MAP_PAD      = 24
TRAIL_SECS   = 5.0
TRAIL_COLOR  = (255, 200, 0)
TRAIL_THICK  = 2
MINIMAP_HZ   = 60

USE_AUTOMATIC_FLY = True

AUTO_MEAN_RUN_DUR   = 1.0
AUTO_MEAN_PAUSE_DUR = 0.7
AUTO_TURN_STD_DEG   = 80.0
AUTO_EDGE_TURN_DEG  = 120.0
AUTO_EDGE_THRESH    = 0.8 * ARENA_RADIUS_MM

# ----------------- SHADERS -----------------

# Warp quad shaders, with toggle for using warp or not
VERT_SRC = r"""
#version 330 core

in vec2 in_pos;
in vec2 in_uv;
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

uniform sampler2D u_cam;   // offscreen camera image (3D fly)
uniform sampler2D u_warp;  // warp map (RG = camera UV)
uniform int u_useWarp;     // 1 = use warp, 0 = bypass (show u_cam directly)

void main() {
    vec2 cam_uv;

    if (u_useWarp == 0) {
        // no distortion: directly use v_uv as camera UV
        cam_uv = v_uv;
    } else {
        // warped mode: map projector UV -> camera UV via warp texture
        cam_uv = texture(u_warp, v_uv).rg;
        cam_uv.y = 1.0 - cam_uv.y;
    }

    // clamp/bounds check
    if (cam_uv.x < 0.0 || cam_uv.y < 0.0 || cam_uv.x > 1.0 || cam_uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = texture(u_cam, cam_uv).rgb;
    fragColor = vec4(col, 1.0);
}
"""

# 3D fly shaders, with stronger ambient so the model is clearly visible
FLY_VERT_SRC = r"""
#version 330 core

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_view;
uniform float u_far;
uniform float u_fovY;
uniform float u_fovX;
uniform int u_projMode; // 0 = perspective, 1 = equidistant fisheye (theta-proportional), 2 = equirectangular

out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(in_pos, 1.0);
    vec4 view_pos  = u_view * world_pos;
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    v_uv = in_uv;

    if (u_projMode == 1) {
        vec3 dir = normalize(-view_pos.xyz); // camera looks down -Z; use forward as +Z in this space
        if (dir.z <= 0.0) { // behind camera -> clip
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        float theta = acos(clamp(dir.z, -1.0, 1.0)); // polar angle from forward
        float max_theta = 0.5 * max(u_fovX, u_fovY);
        if (theta > max_theta) {
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0); // clip outside FOV to avoid artifacts
            return;
        }
        float r = theta / max_theta;

        float len_xy = length(dir.xy);
        vec2 dir_xy_norm = (len_xy > 1e-6) ? dir.xy / len_xy : vec2(0.0, 0.0);
        vec2 proj = r * vec2(dir_xy_norm.x, -dir_xy_norm.y); // flip Y to keep fly upright

        float depth = -view_pos.z / u_far; // simple linear depth
        gl_Position = vec4(proj.x, proj.y, depth, 1.0);
    } else if (u_projMode == 2) {
        vec3 dir = normalize(-view_pos.xyz);
        if (dir.z <= 0.0) { // behind camera -> clip
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        float az = atan(dir.x, dir.z);                      // [-pi, pi]
        float el = asin(clamp(dir.y, -1.0, 1.0));           // [-pi/2, pi/2]

        float half_fov_x = max(u_fovX * 0.5, 1e-6);
        float half_fov_y = max(u_fovY * 0.5, 1e-6);
        if (abs(az) > half_fov_x || abs(el) > half_fov_y) {
            gl_Position = vec4(0.0, 0.0, 0.0, 0.0); // clip outside FOV
            return;
        }
        vec2 ndc;
        ndc.x = az / half_fov_x;
        ndc.y = -el / half_fov_y; // keep +Y up in NDC

        float depth = -view_pos.z / u_far;
        gl_Position = vec4(ndc.x, ndc.y, depth, 1.0);
    } else {
        gl_Position = u_mvp * vec4(in_pos, 1.0);
    }
}
"""

FLY_FRAG_SRC = r"""
#version 330 core

in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;
out vec4 fragColor;

uniform vec4 u_baseColor;
uniform int  u_hasTexture;
uniform sampler2D u_baseColorTex;
uniform float u_ambient;
uniform vec4 u_lightIntensities;
uniform vec3 u_lightDirs[4];
uniform float u_lightMaxGain;

void main() {
    vec3 N = normalize(v_normal);

    vec3 base = (v_color.rgb * u_baseColor.rgb);
    if (u_hasTexture == 1) {
        base *= texture(u_baseColorTex, v_uv).rgb;
    }

    float gain = u_ambient;
    for (int i = 0; i < 4; ++i) {
        gain += u_lightIntensities[i] * max(dot(N, u_lightDirs[i]), 0.0);
    }
    gain = min(gain, u_lightMaxGain);

    vec3 col = base * gain;
    fragColor = vec4(col, 1.0);
}
"""

# ----------------- MINIMAP -----------------

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

    if len(trail_pts_uv) >= 2:
        now_color = np.array(TRAIL_COLOR, dtype=np.float32)
        nseg = len(trail_pts_uv) - 1
        for i in range(nseg):
            p0 = trail_pts_uv[i]
            p1 = trail_pts_uv[i + 1]
            alpha = (i + 1) / nseg
            col = tuple((now_color * (0.3 + 0.7 * alpha)).astype(np.int32).tolist())
            cv2.line(img, p0, p1, col, TRAIL_THICK, cv2.LINE_AA)

    cam_u, cam_v = world_to_minimap(cam_x, cam_y, center_u, center_v, scale_px_per_mm)

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

    overlay = img.copy()
    cv2.fillConvexPoly(overlay, cone_pts_np, (230, 230, 255))
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.polylines(img, [cone_pts_np], True, (150, 150, 255), 1, cv2.LINE_AA)

    fu, fv = world_to_minimap(x, y, center_u, center_v, scale_px_per_mm)
    draw_arrow(img, fu, fv, heading, size_px=18, color=(0, 120, 255))

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
    dist_mm = math.hypot(x - cam_x, y - cam_y)
    cv2.putText(
        img,
        f"cam-fly: {dist_mm:.1f} mm",
        (10, img.shape[0] - 32),
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

def create_texture_from_image(img_rgba):
    h, w, c = img_rgba.shape
    assert c == 4
    internal_fmt = getattr(GL, "GL_SRGB8_ALPHA8", GL.GL_RGBA8)
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_rgba)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex

def pick_monitor(default_w, default_h, which='right'):
    if get_monitors:
        try:
            mons = get_monitors()
        except Exception:
            mons = None

        if mons:
            if which == 'right':
                return max(mons, key=lambda m: m.x)
            if which == 'left':
                return min(mons, key=lambda m: m.x)

    class M: pass
    m = M()
    m.x = 0
    m.y = 0
    m.width = default_w
    m.height = default_h
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

# ----------------- GLB LOADER + MATH -----------------

def _gltf_access_data(gltf, accessor_index, blob):
    acc = gltf.accessors[accessor_index]
    bv = gltf.bufferViews[acc.bufferView]
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count

    if acc.componentType == 5126:      # FLOAT
        dt = np.float32
    elif acc.componentType == 5123:    # UNSIGNED_SHORT
        dt = np.uint16
    elif acc.componentType == 5125:    # UNSIGNED_INT
        dt = np.uint32
    else:
        raise RuntimeError(f"Unsupported componentType {acc.componentType}")

    ncomp = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[acc.type]

    arr = np.frombuffer(blob, dtype=dt, count=count * ncomp, offset=offset)
    if acc.normalized:
        arr = arr.astype(np.float32)
        if acc.componentType in (5121, 5123):      # UNSIGNED_BYTE/SHORT
            max_val = {5121: 255.0, 5123: 65535.0}[acc.componentType]
            arr /= max_val
        elif acc.componentType in (5120, 5122):    # BYTE/SHORT
            max_val = {5120: 127.0, 5122: 32767.0}[acc.componentType]
            arr = np.clip(arr / max_val, -1.0, 1.0)
    return arr.reshape(count, ncomp)

def _extract_image_bytes(gltf, image, blob, path: Path):
    if image.uri:
        if image.uri.startswith("data:"):
            try:
                b64_data = image.uri.split(",", 1)[1]
                return base64.b64decode(b64_data)
            except Exception:
                return None
        img_path = (path.parent / image.uri).expanduser()
        if img_path.exists():
            return img_path.read_bytes()
        return None

    if image.bufferView is not None:
        bv = gltf.bufferViews[image.bufferView]
        offset = bv.byteOffset or 0
        length = bv.byteLength or 0
        return blob[offset: offset + length]
    return None

def _decode_image_to_rgba(data: bytes):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        return None
    return np.ascontiguousarray(img)

def _quat_to_mat4(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),     0.0],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),     0.0],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy), 0.0],
        [0.0,             0.0,             0.0,             1.0],
    ], dtype=np.float32)

def _node_local_matrix(node):
    if node.matrix:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4).T
    t = np.array(node.translation or [0.0, 0.0, 0.0], dtype=np.float32)
    s = np.array(node.scale or [1.0, 1.0, 1.0], dtype=np.float32)
    q = np.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[0:3, 3] = t
    R = _quat_to_mat4(q)
    S = np.diag(np.concatenate([s, [1.0]])).astype(np.float32)
    return T @ R @ S

def _compute_world_matrices(gltf):
    world = {}

    def dfs(node_idx, parent_mat):
        node = gltf.nodes[node_idx]
        local = _node_local_matrix(node)
        wm = parent_mat @ local
        world[node_idx] = wm
        for child in getattr(node, "children", []) or []:
            dfs(child, wm)

    roots = []
    if gltf.scene is not None and gltf.scenes:
        roots = gltf.scenes[gltf.scene].nodes or []
    elif gltf.scenes:
        for sc in gltf.scenes:
            roots.extend(sc.nodes or [])
    else:
        roots = list(range(len(gltf.nodes)))

    for n in roots:
        dfs(n, np.eye(4, dtype=np.float32))
    return world

def load_gltf_triangles(path: Path):
    """
    Load a GLB (all meshes/primitives) and return:
        verts: float32 [N, 12]  (pos.xyz, normal.xyz, color.rgba, uv.xy)
        idx  : uint32 [M]
        draw_calls: list of dicts {base_index, count, base_color_factor, base_color_image}
    Collects all primitives so we don't miss parts of the model (wings, eyes, etc.).
    """
    gltf = GLTF2().load(str(path))
    blob = gltf.binary_blob()

    if not gltf.meshes:
        raise RuntimeError(f"GLB '{path}' has no meshes")

    world_mats = _compute_world_matrices(gltf)

    all_positions = []
    all_normals = []
    all_colors = []
    all_uvs = []
    all_indices = []
    idx_offset = 0

    draw_calls = []

    for node_idx, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        mesh = gltf.meshes[node.mesh]
        if not mesh.primitives:
            continue
        world = world_mats.get(node_idx, np.eye(4, dtype=np.float32))
        normal_mat = np.linalg.inv(world[:3, :3]).T
        for prim in mesh.primitives:
            attrs = prim.attributes  # pygltflib.Attributes

            pos_acc = attrs.POSITION
            if pos_acc is None:
                continue
            positions = _gltf_access_data(gltf, pos_acc, blob).astype(np.float32)
            pos_h = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float32)], axis=1)
            positions = (pos_h @ world.T)[:, :3]

            norm_acc = getattr(attrs, "NORMAL", None)
            if norm_acc is not None:
                normals = _gltf_access_data(gltf, norm_acc, blob).astype(np.float32)
            else:
                normals = np.zeros_like(positions, dtype=np.float32)
            normals = (normals @ normal_mat)
            nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
            normals = normals / nlen

            color_acc = getattr(attrs, "COLOR_0", None)
            colors = None
            if color_acc is not None:
                raw = _gltf_access_data(gltf, color_acc, blob)
                raw = raw.reshape(raw.shape[0], -1)
                if raw.shape[1] == 3:
                    alpha = np.ones((raw.shape[0], 1), dtype=raw.dtype)
                    raw = np.concatenate([raw, alpha], axis=1)
                colors = raw.astype(np.float32)

            uv_acc = getattr(attrs, "TEXCOORD_0", None)
            texcoords = None
            if uv_acc is not None:
                texcoords = _gltf_access_data(gltf, uv_acc, blob).astype(np.float32)

            n_verts = positions.shape[0]
            if normals.shape[0] != n_verts:
                normals = np.zeros_like(positions, dtype=np.float32)
            if colors is None or colors.shape[0] != n_verts:
                colors = np.ones((n_verts, 4), dtype=np.float32)
            if texcoords is None or texcoords.shape[0] != n_verts:
                texcoords = np.zeros((n_verts, 2), dtype=np.float32)

            all_positions.append(positions)
            all_normals.append(normals)
            all_colors.append(colors)
            all_uvs.append(texcoords)

            if prim.indices is not None:
                idx_local = _gltf_access_data(gltf, prim.indices, blob).astype(np.uint32).ravel()
            else:
                idx_local = np.arange(n_verts, dtype=np.uint32)
            all_indices.append(idx_local + idx_offset)

            base_color_factor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            base_color_image = None
            if prim.material is not None and gltf.materials:
                mat = gltf.materials[prim.material]
                pbr = mat.pbrMetallicRoughness
                if pbr is not None and pbr.baseColorFactor:
                    base_color_factor = np.array(pbr.baseColorFactor, dtype=np.float32)

                tex_info = pbr.baseColorTexture if pbr else None
                if tex_info and tex_info.index is not None and gltf.textures:
                    tex = gltf.textures[tex_info.index]
                    if tex.source is not None and 0 <= tex.source < len(gltf.images):
                        img = gltf.images[tex.source]
                        data = _extract_image_bytes(gltf, img, blob, path)
                        base_color_image = _decode_image_to_rgba(data)

            draw_calls.append({
                "base_index": None,  # to be filled after concatenating indices
                "count": idx_local.size,
                "base_color_factor": base_color_factor,
                "base_color_image": base_color_image,
            })
            idx_offset += n_verts

    if not all_positions:
        raise RuntimeError(f"GLB '{path}' contains no mesh primitives with POSITION")

    positions = np.concatenate(all_positions, axis=0)
    normals   = np.concatenate(all_normals, axis=0)
    colors    = np.concatenate(all_colors, axis=0)
    texcoords = np.concatenate(all_uvs, axis=0)
    idx       = np.concatenate(all_indices, axis=0)

    # recenter mesh so rotations happen about its geometric center
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    positions -= center

    # fill base_index for each draw_call based on concatenated indices
    base = 0
    for dc in draw_calls:
        dc["base_index"] = base
        base += dc["count"]

    verts = np.concatenate([positions, normals, colors, texcoords], axis=1).astype(np.float32)
    return verts, idx, draw_calls

def perspective(fov_y_rad, aspect, z_near, z_far):
    # guard against pathological FOV (>= 180 deg) that would blow up tan()
    max_fov = math.radians(179.0)
    if FLY_CAM_ALLOW_ULTRAWIDE:
        fov_y_rad = min(fov_y_rad, max_fov)
    else:
        fov_y_rad = min(fov_y_rad, math.radians(120.0))

    f = 1.0 / math.tan(0.5 * fov_y_rad)
    nf = 1.0 / (z_near - z_far)
    return np.array([
        [f / aspect, 0.0, 0.0,                               0.0],
        [0.0,        f,   0.0,                               0.0],
        [0.0,        0.0, (z_far + z_near) * nf,  2*z_far*z_near*nf],
        [0.0,        0.0, -1.0,                              0.0],
    ], dtype=np.float32)

def look_at(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] =  np.dot(f, eye)
    return m

def mat4_translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m

def mat4_rotate_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [ c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

def mat4_rotate_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c, -s, 0.0],
        [0.0,  s,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

def mat4_scale(s):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s
    m[1, 1] = s
    m[2, 2] = s
    return m

def compute_light_dirs(elev_deg):
    elev = math.radians(elev_deg)
    horiz = math.cos(elev)
    up = math.sin(elev)
    dirs = [
        np.array([0.0, up, -horiz], dtype=np.float32),   # north -> shining south
        np.array([-horiz, up, 0.0], dtype=np.float32),   # east -> shining west
        np.array([0.0, up, horiz], dtype=np.float32),    # south -> shining north
        np.array([horiz, up, 0.0], dtype=np.float32),    # west -> shining east
    ]
    # normalize to be safe
    dirs = [d / max(np.linalg.norm(d), 1e-6) for d in dirs]
    return np.stack(dirs, axis=0)

def enforce_min_distance(pos, other, min_dist):
    px, py = pos
    ox, oy = other
    dx = px - ox
    dy = py - oy
    dist = math.hypot(dx, dy)
    if dist < min_dist:
        if dist < 1e-6:
            px = ox + min_dist
            py = oy
        else:
            scale = min_dist / dist
            px = ox + dx * scale
            py = oy + dy * scale
    return px, py

def create_offscreen_cam(cam_w, cam_h):
    color_tex = create_texture_2d(cam_w, cam_h, GL.GL_RGB8, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

    depth_rb = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, cam_w, cam_h)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color_tex, 0)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb)

    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Offscreen framebuffer incomplete: 0x{status:04X}")

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return fbo, color_tex, depth_rb

# ----------------- WARP LOADING -----------------

def load_warp(mapx_path: Path, mapy_path: Path, factor: float = 1):
    mapx = np.load(str(mapx_path)).astype(np.float32)
    mapy = np.load(str(mapy_path)).astype(np.float32)
    print("raw map shapes:", mapx.shape, mapy.shape)
    mapx = mapx / factor
    mapy = mapy / factor

    mapx = np.flipud(mapx)
    mapy = np.flipud(mapy)

    proj_h, proj_w = mapx.shape
    print(f"Loaded warp map: projector size = {proj_w} x {proj_h}")
    valid = np.isfinite(mapx) & np.isfinite(mapy) & (mapx >= 0) & (mapy >= 0)
    if not np.any(valid):
        raise RuntimeError("mapx/mapy contain no valid entries")

    cam_w = int(np.ceil(mapx[valid].max())) + 1
    cam_h = int(np.ceil(mapy[valid].max())) + 1
    print(f"Loaded warp map: camera size = {cam_w} x {cam_h}")

    warp = np.zeros((proj_h, proj_w, 2), dtype=np.float32)
    warp[..., 0] = mapx / float(cam_w)
    warp[..., 1] = mapy / float(cam_h)
    warp[~valid, 0] = -1.0
    warp[~valid, 1] = -1.0
    return warp, cam_w, cam_h, proj_w, proj_h

# ----------------- MAIN -----------------

def main():
    warp, cam_w, cam_h, proj_w, proj_h = load_warp(MAPX_PATH, MAPY_PATH)

    mon = pick_monitor(proj_w, proj_h, which='right')

    if not is_mac:
        os.environ.setdefault("SDL_VIDEODRIVER", "windows")
        os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    else:
        os.environ.pop("SDL_VIDEODRIVER", None)

    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon.x},{mon.y}"

    pygame.init()
    request_gl_context(proj_w, proj_h, borderless=True)

    base_title = "GPU-warped 3D fly (GLB)"
    pygame.display.set_caption(base_title)

    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glViewport(0, 0, proj_w, proj_h)
    GL.glDisable(GL.GL_DEPTH_TEST)

    # warp quad program
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

    warp_tex = create_texture_2d(proj_w, proj_h, GL.GL_RG32F, GL.GL_RG, GL.GL_FLOAT)
    u_cam_loc     = GL.glGetUniformLocation(prog, "u_cam")
    u_warp_loc    = GL.glGetUniformLocation(prog, "u_warp")
    u_useWarp_loc = GL.glGetUniformLocation(prog, "u_useWarp")

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

    # offscreen fly camera
    fly_fbo, cam_tex, depth_rb = create_offscreen_cam(cam_w, cam_h)

    # 3D fly program + mesh
    fly_prog = create_program(FLY_VERT_SRC, FLY_FRAG_SRC)
    GL.glUseProgram(fly_prog)
    u_fly_mvp_loc        = GL.glGetUniformLocation(fly_prog, "u_mvp")
    u_fly_model_loc      = GL.glGetUniformLocation(fly_prog, "u_model")
    u_fly_view_loc       = GL.glGetUniformLocation(fly_prog, "u_view")
    u_fly_far_loc        = GL.glGetUniformLocation(fly_prog, "u_far")
    u_fly_fovy_loc       = GL.glGetUniformLocation(fly_prog, "u_fovY")
    u_fly_fovx_loc       = GL.glGetUniformLocation(fly_prog, "u_fovX")
    u_fly_proj_mode_loc  = GL.glGetUniformLocation(fly_prog, "u_projMode")
    u_fly_base_color_loc = GL.glGetUniformLocation(fly_prog, "u_baseColor")
    u_fly_has_tex_loc    = GL.glGetUniformLocation(fly_prog, "u_hasTexture")
    u_fly_tex_loc        = GL.glGetUniformLocation(fly_prog, "u_baseColorTex")
    u_fly_ambient_loc    = GL.glGetUniformLocation(fly_prog, "u_ambient")
    u_fly_light_int_loc  = GL.glGetUniformLocation(fly_prog, "u_lightIntensities")
    u_fly_light_dirs_loc = GL.glGetUniformLocation(fly_prog, "u_lightDirs")
    u_fly_light_max_loc  = GL.glGetUniformLocation(fly_prog, "u_lightMaxGain")

    fly_verts, fly_indices, fly_draws = load_gltf_triangles(FLY_MODEL_PATH)

    # If FOV > 180°, invert model on its Y axis (positions + normals)
    if FLY_CAM_FOV_X_DEG is not None and FLY_CAM_FOV_X_DEG > 180.0:
        # pos.y is column 1, normal.y is column 4 in [pos.xyz, normal.xyz, color.rgba, uv.xy]
        fly_verts[:, 1] *= -1.0  # invert position Y
        fly_verts[:, 4] *= -1.0  # invert normal Y

    # create textures per draw (if needed)
    draw_textures = []
    for dc in fly_draws:
        tex = None
        if dc.get("base_color_image") is not None:
            tex = create_texture_from_image(dc["base_color_image"])
        draw_textures.append(tex)

    GL.glUniform1i(u_fly_tex_loc, 0)

    # auto-scale GLB so its longest dimension = FLY_PHYS_LENGTH_MM
    pos = fly_verts[:, 0:3]
    extents = pos.max(axis=0) - pos.min(axis=0)
    longest = float(extents.max()) if extents.max() > 0 else 1.0
    fly_base_scale = FLY_BASE_SCALE * (FLY_PHYS_LENGTH_MM / longest)
    print(
        "fly extents:", extents,
        "longest:", longest,
        "base_scale:", fly_base_scale,
        "bio_len_mm:", FLY_PHYS_LENGTH_MM * FLY_BASE_SCALE,
    )

    fly_vao = GL.glGenVertexArrays(1)
    fly_vbo = GL.glGenBuffers(1)
    fly_ebo = GL.glGenBuffers(1)

    GL.glBindVertexArray(fly_vao)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, fly_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, fly_verts.nbytes, fly_verts, GL.GL_STATIC_DRAW)

    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, fly_ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, fly_indices.nbytes, fly_indices, GL.GL_STATIC_DRAW)

    stride_f = 12 * 4
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride_f, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride_f, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(2)
    GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, stride_f, ctypes.c_void_p(6 * 4))
    GL.glEnableVertexAttribArray(3)
    GL.glVertexAttribPointer(3, 2, GL.GL_FLOAT, GL.GL_FALSE, stride_f, ctypes.c_void_p(10 * 4))

    GL.glBindVertexArray(0)

    # set static lighting uniforms
    light_dirs = compute_light_dirs(LIGHT_ELEVATION_DEG)
    light_int = np.array(LIGHT_INTENSITIES, dtype=np.float32)
    if light_int.shape[0] != 4:
        light_int = np.resize(light_int, 4)
    GL.glUseProgram(fly_prog)
    GL.glUniform1f(u_fly_ambient_loc, float(LIGHT_AMBIENT))
    GL.glUniform4fv(u_fly_light_int_loc, 1, light_int.astype(np.float32))
    GL.glUniform3fv(u_fly_light_dirs_loc, 4, light_dirs.astype(np.float32))
    GL.glUniform1f(u_fly_light_max_loc, float(LIGHT_MAX_GAIN))
    # projection for fly camera (independent FOV)
    aspect = cam_w / float(cam_h)
    projection_mode = (FLY_CAM_PROJECTION or "equirect").strip().lower()
    if FLY_CAM_FOV_X_DEG is not None:
        fov_x_rad_raw = math.radians(FLY_CAM_FOV_X_DEG)
        if projection_mode == "equirect":
            fov_y_rad_raw = fov_x_rad_raw * (cam_h / cam_w)
        else:
            fov_y_rad_raw = 2.0 * math.atan(math.tan(0.5 * fov_x_rad_raw) / max(aspect, 1e-6))
    else:
        fov_y_rad_raw = math.radians(FLY_CAM_FOV_Y_DEG)
        if projection_mode == "equirect":
            fov_x_rad_raw = fov_y_rad_raw * (cam_w / cam_h)
        else:
            fov_x_rad_raw = 2.0 * math.atan(math.tan(0.5 * fov_y_rad_raw) * aspect)

    # clamp if ultrawide (>=180) to keep projection stable
    max_fov_y_rad = math.radians(179.0)
    max_fov_x_rad = math.radians(359.0)
    fov_y_rad = min(fov_y_rad_raw, max_fov_y_rad) if FLY_CAM_ALLOW_ULTRAWIDE else min(fov_y_rad_raw, math.radians(120.0))
    fov_x_rad = min(fov_x_rad_raw, max_fov_x_rad) if FLY_CAM_ALLOW_ULTRAWIDE else min(fov_x_rad_raw, math.radians(160.0))

    fov_y_deg_effective = math.degrees(fov_y_rad)
    fov_x_deg_effective = math.degrees(fov_x_rad)
    print(f"Fly camera FOV used: {fov_y_deg_effective:.2f} deg vertical, {fov_x_deg_effective:.2f} deg horizontal (aspect {aspect:.3f}, mode {projection_mode})")
    z_near = 1.0
    z_far = 10.0 * ARENA_RADIUS_MM
    proj_mode = 0
    if projection_mode == "equidistant":
        proj_mode = 1
    elif projection_mode == "equirect":
        proj_mode = 2
    proj_mat = perspective(fov_y_rad, aspect, z_near=z_near, z_far=z_far) if proj_mode == 0 else np.eye(4, dtype=np.float32)
    flip_model_for_ultrawide = (
        FLY_CAM_FLIP_MODEL_FOR_ULTRAWIDE and proj_mode != 0 and fov_x_deg_effective > 180.0
    )

    # state
    x, y = START_POS
    heading = math.radians(START_HEADING_DEG)

    auto_state = "pause"
    auto_state_t_remaining = np.random.exponential(AUTO_MEAN_PAUSE_DUR)
    auto_pending_turn = 0.0
    artificial_paused = False

    yaw_offset_deg = float(FLY_MODEL_YAW_OFFSET_DEG)
    min_cam_fly_dist = float(MIN_CAM_FLY_DIST_MM)
    apparent_distance_mm = float(FLY_APPARENT_DISTANCE_MM)
    screen_distance_mm = float(SCREEN_DISTANCE_MM)
    fly_scale_target = fly_base_scale * (screen_distance_mm / max(apparent_distance_mm, 1e-6))
    fly_scale_current = fly_scale_target

    camera_x = CAMERA_X_MM
    camera_y = CAMERA_Y_MM
    cam_height = CAM_HEIGHT_MM
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

    clock = pygame.time.Clock()
    running = True
    next_minimap_t = 0.0

    # warp toggle: True = warped, False = raw camera texture
    use_warp = True

    last_caption_state = (None, None, None)

    def update_caption():
        nonlocal last_caption_state
        state = (round(yaw_offset_deg, 2), round(cam_height, 3), round(min_cam_fly_dist, 3))
        if state != last_caption_state:
            pygame.display.set_caption(
                f"{base_title} | FLY_MODEL_YAW_OFFSET_DEG={yaw_offset_deg:.1f} | cam_height_mm={cam_height:.2f} | MIN_CAM_FLY_DIST_MM={min_cam_fly_dist:.2f}"
            )
            last_caption_state = state

    update_caption()

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
                elif event.key == pygame.K_p and USE_AUTOMATIC_FLY:
                    artificial_paused = not artificial_paused
                elif event.key == pygame.K_u:
                    use_warp = not use_warp  # toggle warp
                elif event.key == pygame.K_9:
                    yaw_offset_deg -= YAW_ADJ_STEP_DEG
                    update_caption()
                    print(f"FLY_MODEL_YAW_OFFSET_DEG={yaw_offset_deg:.1f}")
                elif event.key == pygame.K_0:
                    yaw_offset_deg += YAW_ADJ_STEP_DEG
                    update_caption()
                    print(f"FLY_MODEL_YAW_OFFSET_DEG={yaw_offset_deg:.1f}")
                elif event.key == pygame.K_MINUS:
                    min_cam_fly_dist = max(0.0, min_cam_fly_dist - MIN_DIST_ADJ_STEP_MM)
                    update_caption()
                    print(f"MIN_CAM_FLY_DIST_MM={min_cam_fly_dist:.2f}")
                elif event.key == pygame.K_EQUALS:
                    min_cam_fly_dist += MIN_DIST_ADJ_STEP_MM
                    update_caption()
                    print(f"MIN_CAM_FLY_DIST_MM={min_cam_fly_dist:.2f}")
                elif event.key == pygame.K_PERIOD:
                    cam_height += HEIGHT_ADJ_STEP_MM
                    update_caption()
                    print(f"cam_height_mm={cam_height:.2f}")
                elif event.key == pygame.K_COMMA:
                    cam_height -= HEIGHT_ADJ_STEP_MM
                    update_caption()
                    print(f"cam_height_mm={cam_height:.2f}")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

        prev_x, prev_y = x, y
        prev_cam_x, prev_cam_y = camera_x, camera_y

        # Smooth distance-based scale to avoid pops when apparent distance changes.
        fly_scale_target = fly_base_scale * (screen_distance_mm / max(apparent_distance_mm, 1e-6))
        if DIST_SCALE_SMOOTH_HZ > 0:
            alpha = 1.0 - math.exp(-DIST_SCALE_SMOOTH_HZ * dt)
        else:
            alpha = 1.0
        fly_scale_current = fly_scale_current + alpha * (fly_scale_target - fly_scale_current)

        # fly behaviour
        if USE_AUTOMATIC_FLY:
            moving = False
            if not artificial_paused:
                auto_state_t_remaining -= dt

                if auto_state == "run":
                    max_auto_turn = turn_rate_rad * STAND_TURN_MULT * dt
                    if abs(auto_pending_turn) > 1e-4:
                        turn_step = max(-max_auto_turn, min(max_auto_turn, auto_pending_turn))
                        heading += turn_step
                        auto_pending_turn -= turn_step
                    else:
                        x += speed_f * math.sin(heading) * dt
                        y += speed_f * math.cos(heading) * dt
                        moving = True

                        heading += math.radians(WALK_TURN_NOISE_DEG_RMS) * np.random.normal(0.0, math.sqrt(dt))
                        x += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))
                        y += WALK_TRANS_NOISE_MM_RMS * np.random.normal(0.0, math.sqrt(dt))

                r_center = math.hypot(x, y)
                if moving and r_center > AUTO_EDGE_THRESH:
                    angle_to_center = math.atan2(-x, -y)
                    delta = angle_to_center - heading
                    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
                    max_turn = math.radians(AUTO_EDGE_TURN_DEG) * dt
                    delta_clamped = max(-max_turn, min(max_turn, delta))
                    heading += delta_clamped

                if auto_state_t_remaining <= 0.0:
                    if auto_state == "pause":
                        auto_state = "run"
                        auto_state_t_remaining = np.random.exponential(AUTO_MEAN_RUN_DUR)
                        auto_pending_turn += math.radians(AUTO_TURN_STD_DEG) * np.random.normal()
                    else:
                        auto_state = "pause"
                        auto_state_t_remaining = np.random.exponential(AUTO_MEAN_PAUSE_DUR)

            r_center = math.hypot(x, y)
            if r_center > ARENA_RADIUS_MM:
                scale_back = ARENA_RADIUS_MM / r_center
                x *= scale_back
                y *= scale_back

        else:
            moving = keys[pygame.K_w] or keys[pygame.K_s]
            current_turn_rate = turn_rate_rad * (STAND_TURN_MULT if not moving else 1.0)

            if keys[pygame.K_a]:
                heading -= current_turn_rate * dt  # left key turns left
            if keys[pygame.K_d]:
                heading += current_turn_rate * dt  # right key turns right

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
            x, y = enforce_min_distance((x, y), (camera_x, camera_y), min_cam_fly_dist)
            r_center = math.hypot(x, y)
            if r_center > ARENA_RADIUS_MM:
                scale_back = ARENA_RADIUS_MM / r_center
                x *= scale_back
                y *= scale_back
            if math.hypot(x - camera_x, y - camera_y) < min_cam_fly_dist:
                x, y = prev_x, prev_y

        # camera controls
        cam_moving = keys[pygame.K_UP] or keys[pygame.K_DOWN]
        cam_current_turn_rate = cam_turn_rate_rad * (CAMERA_STAND_TURN_MULT if not cam_moving else 1.0)

        if keys[pygame.K_LEFT]:
            cam_heading -= cam_current_turn_rate * dt  # left arrow turns view left
        if keys[pygame.K_RIGHT]:
            cam_heading += cam_current_turn_rate * dt  # right arrow turns view right

        if keys[pygame.K_UP]:
            camera_x += CAMERA_SPEED_MM_S * math.sin(cam_heading) * dt
            camera_y += CAMERA_SPEED_MM_S * math.cos(cam_heading) * dt
        if keys[pygame.K_DOWN]:
            camera_x -= CAMERA_SPEED_MM_S * math.sin(cam_heading) * dt
            camera_y -= CAMERA_SPEED_MM_S * math.cos(cam_heading) * dt
        if keys[pygame.K_PERIOD]:
            cam_height += CAMERA_Z_SPEED_MM_S * dt
        if keys[pygame.K_COMMA]:
            cam_height -= CAMERA_Z_SPEED_MM_S * dt

        r_cam_center = math.hypot(camera_x, camera_y)
        if r_cam_center > ARENA_RADIUS_MM:
            scale_back_cam = ARENA_RADIUS_MM / r_cam_center
            camera_x *= scale_back_cam
            camera_y *= scale_back_cam
        if math.hypot(camera_x - x, camera_y - y) < min_cam_fly_dist:
            camera_x, camera_y = prev_cam_x, prev_cam_y
        # If camera moves into fly, push the fly away (do not move camera)
        x, y = enforce_min_distance((x, y), (camera_x, camera_y), min_cam_fly_dist)
        r_center = math.hypot(x, y)
        if r_center > ARENA_RADIUS_MM:
            scale_back = ARENA_RADIUS_MM / r_center
            x *= scale_back
            y *= scale_back
        camera_x, camera_y = enforce_min_distance((camera_x, camera_y), (x, y), min_cam_fly_dist)
        r_cam_center = math.hypot(camera_x, camera_y)
        if r_cam_center > ARENA_RADIUS_MM:
            scale_back_cam = ARENA_RADIUS_MM / r_cam_center
            camera_x *= scale_back_cam
            camera_y *= scale_back_cam

        if SHOW_MINIMAP:
            trail.append((now, x, y))
            expire_before = now - TRAIL_SECS
            while trail and trail[0][0] < expire_before:
                trail.popleft()

        update_caption()

        # offscreen render of 3D fly
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fly_fbo)
        GL.glViewport(0, 0, cam_w, cam_h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(BG_COLOR[2] / 255.0, BG_COLOR[1] / 255.0, BG_COLOR[0] / 255.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        eye = [camera_x, cam_height, camera_y]
        forward = [math.sin(cam_heading), 0.0, math.cos(cam_heading)]
        target = [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]]
        up = [0.0, 1.0, 0.0]

        view_mat = look_at(eye, target, up)
        # Mirror the heading for the model so screen rotation matches minimap controls.
        yaw = -heading + math.radians(yaw_offset_deg)
        base_rot = mat4_rotate_y(yaw)
        if flip_model_for_ultrawide:
            base_rot = mat4_rotate_x(math.pi) @ base_rot  # flip upside-down when ultrawide
        model_mat = mat4_translate(x, 0.0, y) @ base_rot @ mat4_scale(fly_scale_current)
        mvp = proj_mat @ view_mat @ model_mat

        GL.glUseProgram(fly_prog)
        GL.glBindVertexArray(fly_vao)
        GL.glUniformMatrix4fv(u_fly_view_loc, 1, GL.GL_FALSE, view_mat.T.astype(np.float32))
        GL.glUniformMatrix4fv(u_fly_mvp_loc, 1, GL.GL_FALSE, mvp.T.astype(np.float32))
        GL.glUniformMatrix4fv(u_fly_model_loc, 1, GL.GL_FALSE, model_mat.T.astype(np.float32))
        GL.glUniform1f(u_fly_far_loc, z_far)
        GL.glUniform1f(u_fly_fovy_loc, fov_y_rad)
        GL.glUniform1f(u_fly_fovx_loc, fov_x_rad)
        GL.glUniform1i(u_fly_proj_mode_loc, proj_mode)

        index_stride = ctypes.sizeof(ctypes.c_uint32)
        for dc, tex in zip(fly_draws, draw_textures):
            GL.glUniform4fv(u_fly_base_color_loc, 1, dc["base_color_factor"].astype(np.float32))
            GL.glUniform1i(u_fly_has_tex_loc, 1 if tex is not None else 0)
            if tex is not None:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                dc["count"],
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(dc["base_index"] * index_stride),
            )
        GL.glBindVertexArray(0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDisable(GL.GL_DEPTH_TEST)

        # warp / unwarp pass
        GL.glViewport(0, 0, proj_w, proj_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(prog)
        GL.glUniform1i(u_useWarp_loc, 1 if use_warp else 0)

        GL.glBindVertexArray(vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, cam_tex)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, warp_tex)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        pygame.display.flip()

        # minimap
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

    tex_list = [warp_tex, cam_tex]
    for tex in draw_textures:
        if tex is not None:
            tex_list.append(tex)
    GL.glDeleteTextures(tex_list)
    GL.glDeleteRenderbuffers(1, [depth_rb])
    GL.glDeleteFramebuffers(1, [fly_fbo])
    GL.glDeleteBuffers(3, [vbo, fly_vbo, fly_ebo])
    GL.glDeleteVertexArrays(2, [vao, fly_vao])
    GL.glDeleteProgram(prog)
    GL.glDeleteProgram(fly_prog)
    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
