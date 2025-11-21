import argparse
import os
import sys
import ctypes
from pathlib import Path

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME, QUIT, KEYDOWN, K_ESCAPE
from OpenGL import GL

try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DEFAULT_MAPX_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapx.experimental.npy")
DEFAULT_MAPY_PATH = Path(r"D:/screen.calibration/configs/camera.projector.mapping/mapy.experimental.npy")
DEFAULT_WINDOW_W = 1280
DEFAULT_WINDOW_H = 800


# -------------------------------------------------------------
# SHADERS
# -------------------------------------------------------------

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

    // flip vertically (image was upside down)
    cam_uv.y = 1.0 - cam_uv.y;

    // invalid mapping marker: u < 0
    if (cam_uv.x < 0.0 || cam_uv.y < 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = texture(u_cam, cam_uv).rgb;
    fragColor = vec4(col, 1.0);
}
"""


# -------------------------------------------------------------
# GL HELPERS
# -------------------------------------------------------------

def compile_shader(source, shader_type):
    type_name = {
        GL.GL_VERTEX_SHADER: "vertex",
        GL.GL_FRAGMENT_SHADER: "fragment",
    }.get(shader_type, str(shader_type))
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not status:
        log = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error ({type_name}):\n{log}")
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

    class M:
        pass

    m = M()
    m.x = 0
    m.y = 0
    m.width = default_w
    m.height = default_h
    return m


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preview projector warp using a GPU shader.")
    parser.add_argument(
        "--mapx",
        type=Path,
        default=DEFAULT_MAPX_PATH,
        help="Path to the numpy file with camera u coordinates (default: %(default)s)",
    )
    parser.add_argument(
        "--mapy",
        type=Path,
        default=DEFAULT_MAPY_PATH,
        help="Path to the numpy file with camera v coordinates (default: %(default)s)",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=DEFAULT_WINDOW_W,
        help="Output window width in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=DEFAULT_WINDOW_H,
        help="Output window height in pixels (default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_warp(mapx_path: Path, mapy_path: Path):
    mapx_path = mapx_path.expanduser()
    mapy_path = mapy_path.expanduser()
    if not mapx_path.exists():
        raise FileNotFoundError(f"mapx file does not exist: {mapx_path}")
    if not mapy_path.exists():
        raise FileNotFoundError(f"mapy file does not exist: {mapy_path}")

    mapx = np.load(mapx_path).astype(np.float32)
    mapy = np.load(mapy_path).astype(np.float32)

    # flip vertically if needed
    mapx = np.flipud(mapx)
    mapy = np.flipud(mapy)

    if mapx.shape != mapy.shape:
        raise ValueError(f"mapx/mapy shape mismatch: {mapx.shape} vs {mapy.shape}")

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
        raise RuntimeError(
            f"OpenGL {major}.{minor} context is too old. "
            "Ensure your GPU driver supports OpenGL 3.3 or higher."
        )
    version = GL.glGetString(GL.GL_VERSION)
    print(f"OpenGL context: {major}.{minor} ({version.decode() if version else 'unknown'})")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    warp, cam_w, cam_h, proj_w, proj_h = load_warp(args.mapx, args.mapy)
    window_w = max(1, args.window_width)
    window_h = max(1, args.window_height)
    mon = pick_rightmost_monitor(window_w, window_h)
    os.environ.setdefault("SDL_VIDEODRIVER", "windows")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "1")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon.x},{mon.y}"

    pygame.init()
    request_gl_context(window_w, window_h, borderless=True)
    pygame.display.set_caption("GPU warp demo")

    # allow tightly packed RGB rows (3-byte alignment)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

    GL.glViewport(0, 0, window_w, window_h)
    GL.glDisable(GL.GL_DEPTH_TEST)

    prog = create_program(VERT_SRC, FRAG_SRC)
    GL.glUseProgram(prog)

    quad_vertices = np.array([
        # x,    y,    u,   v
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
    loc_uv = GL.glGetAttribLocation(prog, "in_uv")

    GL.glEnableVertexAttribArray(loc_pos)
    GL.glVertexAttribPointer(
        loc_pos,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(0)
    )

    GL.glEnableVertexAttribArray(loc_uv)
    GL.glVertexAttribPointer(
        loc_uv,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(2 * quad_vertices.itemsize)
    )

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    cam_tex = create_texture_2d(
        cam_w,
        cam_h,
        GL.GL_RGB8,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        min_filter=GL.GL_LINEAR,
        mag_filter=GL.GL_LINEAR,
    )

    warp_tex = create_texture_2d(
        proj_w,
        proj_h,
        GL.GL_RG32F,
        GL.GL_RG,
        GL.GL_FLOAT,
        min_filter=GL.GL_NEAREST,
        mag_filter=GL.GL_NEAREST,
    )

    u_cam_loc = GL.glGetUniformLocation(prog, "u_cam")
    u_warp_loc = GL.glGetUniformLocation(prog, "u_warp")
    GL.glUseProgram(prog)
    GL.glUniform1i(u_cam_loc, 0)
    GL.glUniform1i(u_warp_loc, 1)

    # test camera image
    yy, xx = np.indices((cam_h, cam_w))
    cam_img = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    cam_img[..., 0] = (xx / cam_w * 255).astype(np.uint8)
    cam_img[..., 1] = (yy / cam_h * 255).astype(np.uint8)
    cam_img[..., 2] = 128
    cam_img[yy % 50 == 0] = (255, 255, 255)
    cam_img[xx % 50 == 0] = (255, 255, 255)
    cam_img = np.ascontiguousarray(cam_img, dtype=np.uint8)

    # upload warp once
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

    # initial cam upload
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, cam_tex)
    cam_ptr = cam_img.ctypes.data_as(ctypes.c_void_p)
    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        cam_w,
        cam_h,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        cam_ptr,
    )

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        t = pygame.time.get_ticks() / 1000.0
        shift = int((np.sin(t) * 0.5 + 0.5) * 50)  # 0..50 px
        shifted = np.roll(cam_img, shift, axis=1)
        shifted = np.ascontiguousarray(shifted, dtype=np.uint8)

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

        GL.glViewport(0, 0, window_w, window_h)
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
        clock.tick(60)

    GL.glDeleteTextures([cam_tex, warp_tex])
    GL.glDeleteBuffers(1, [vbo])
    GL.glDeleteVertexArrays(1, [vao])
    GL.glDeleteProgram(prog)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
