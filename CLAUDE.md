# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **virtual fly stimulus system** for neuroscience experiments. It renders a 3D fly model (GLB format) in a circular arena and projects the output through a calibrated projector warp map. The system is used to present realistic visual stimuli of a walking fly to real flies in a laboratory setup.

## Architecture

The project evolved through several iterations, each building on the last:

### Pipeline Stages
1. **Image capture & interpolation** (`interpolate1.py`, `interpolate2.py`): Take turntable photos of a real fly at 1-degree intervals, rename/sort by angle, and interpolate to 3600 frames for smooth playback.
2. **3D reconstruction** (`recover3dmode.py`): Visual hull reconstruction from silhouettes into a voxel grid, exported as `.ply` mesh.
3. **2D sprite stimulus** (`stim1.py`, `stim2.py`, `stim_interactive*.py`): Plays back turntable images as 2D sprites with keyboard/autonomous control. Progressively adds arena minimap, vsync, and projector warp.
4. **Shader-based rendering** (`create_shader.py`, `new_shader_approach.py`, `fly_with_shader_on_superbowl*.py`): OpenGL shader pipeline that warps 2D sprite output through calibrated projector-to-camera mapping (mapx/mapy .npy files).
5. **Full 3D GLB rendering** (`3d_object_fly*.py`): The current main scripts. Loads a `.glb` fly model via `pygltflib`, renders it with OpenGL shaders, applies physically-correct scaling (mm-to-pixel), perspective/equirectangular projection, and projector warp correction. `3d_object_fly4.py` is the latest version.

### Key Concepts
- **Projector warp maps**: `mapx.npy`/`mapy.npy` files define the pixel mapping from projector space to camera space, loaded as OpenGL textures for real-time correction.
- **Physically-scaled rendering**: The fly model is scaled to a target physical size in mm (`FLY_PHYS_LENGTH_MM`), and apparent size on screen is computed from camera-to-fly distance and a known screen distance (`SCREEN_DISTANCE_MM`).
- **Dual control modes**: Fly movement is either keyboard-controlled (WASD) or autonomous (run/pause state machine with edge avoidance). Controlled by `USE_AUTOMATIC_FLY`.
- **Minimap**: Optional 2D top-down view showing arena, fly position/heading, camera FOV cone, and movement trail.
- **Platform-aware paths**: Scripts use `sys.platform == "darwin"` to switch between Mac and Windows file paths for calibration data and models.

## Running

All scripts are standalone Python files. Run directly:
```
python 3d_object_fly4.py    # latest 3D renderer (main script)
python stim_interactive.py  # 2D sprite version with keyboard control
```

## Dependencies

- `numpy`, `opencv-python` (`cv2`), `pygame`, `PyOpenGL`, `pygltflib`
- Optional: `screeninfo` (auto-detect monitor), `Pillow` (`PIL`), `scikit-image` (for 3D reconstruction)

## Key Configuration

All config is at the top of each script as module-level constants. The most important in `3d_object_fly4.py`:
- `FLY_MODEL_PATH`: Path to the `.glb` model file
- `MAPX_PATH`/`MAPY_PATH`: Projector calibration warp maps
- `FLY_PHYS_LENGTH_MM`: Target physical fly size in mm
- `SCREEN_DISTANCE_MM`: Physical eye-to-screen distance for size calculations
- `FLY_CAM_FOV_X_DEG`/`FLY_CAM_PROJECTION`: Camera field of view and projection mode (`perspective`, `equirect`, `equidistant`)
- `ARENA_RADIUS_MM`: Size of the circular arena
- `USE_AUTOMATIC_FLY`: Toggle between keyboard and autonomous fly movement

## Sibling Repository: `D:\screen.calibration`

The calibration system lives in a separate repo at `D:\screen.calibration`. The virtual.fly scripts depend on its output files. Key structure:

- **`configs/camera.projector.mapping/`**: Contains the warp maps (`mapx.npy`, `mapy.npy`, `mapx.experimental.npy`, `mapy.experimental.npy`, `valid.mask.npy`) consumed by the virtual.fly rendering scripts.
- **`configs/fisheye.config/`**: Fisheye camera intrinsics (`fisheye.K.npy`, `fisheye.D.npy`, `fisheye.xi.npy`) from omni/fisheye calibration.
- **`configs/pinhole.config/`**: Pinhole camera intrinsics (`pinhole.K.npy`, `pinhole.D.npy`).
- **`fisheye.pipeline.py`**: Top-level orchestrator that chains the full calibration pipeline: chessboard detection → fisheye K/D/xi calibration → structured-light projector-camera mapping → sparse refinement → warp stimulus preview.
- **`generate_projector_camera_mapping/`**: Structured-light mapping pipeline (`mapping_pipeline.py`) that projects sine/gray patterns via pygame, captures with an Alvium or FLIR camera, and decodes phase to build `mapx`/`mapy`.
- **`cameras/`**: Camera drivers — `CamAlvium.py` (Allied Vision Alvium) and `CamRotPy.py` (FLIR/Spinnaker via RotPy).
- **`stimulus/`**: Warp test scripts (`warp_circle.py`, `warp_moving_circle.py`, etc.) for verifying calibration by displaying warped stimuli.

### Calibration Workflow
1. Capture chessboard images → compute fisheye K, D, xi (`camera_internals/fisheye_KDxi.py`)
2. Run structured-light mapping: project sine patterns, decode phase → raw `mapx.npy`/`mapy.npy` (`generate_projector_camera_mapping/mapping_pipeline.py`)
3. Refine maps with sparse dot-grid verification (`refine_mapx_mapy.py`) → `mapx.experimental.npy`/`mapy.experimental.npy`
4. Verify with warp stimulus scripts (`stimulus/warp_moving_circle_fish.py`)

The virtual.fly scripts reference the `*.experimental.npy` maps by default.

## File Naming Convention

Versioned scripts use incrementing suffixes (e.g., `3d_object_fly.py` → `3d_object_fly4.py`). The highest number is the latest/active version. Similarly for `fly_with_shader_on_superbowl*.py` variants which add features progressively (camera movement, FOV enforcement).
