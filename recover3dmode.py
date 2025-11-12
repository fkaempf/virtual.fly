import os, glob, math
import numpy as np
from PIL import Image

# Optional: surface reconstruction
try:
    from skimage.measure import marching_cubes
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# ---------------- Config ----------------
IMG_FOLDER = "/Users/fkampf/Downloads/aaa"   # change to your folder path
PATTERN = "fly*.png"             # will catch fly-180.png ... fly180.png
RES = 128                        # voxel grid resolution
Z_SCALE = 0.5                    # relative extent in Z (flatten if <1.0)
THRESH = 250                     # background threshold

# ----------------------------------------

# Load images
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, PATTERN)))
if not img_paths:
    raise FileNotFoundError("No images found, check IMG_FOLDER and PATTERN")

imgs = [Image.open(p).convert("RGB") for p in img_paths]
W, H = imgs[0].size

def to_mask(im):
    arr = np.array(im)
    mask = (arr[...,0] < THRESH) | (arr[...,1] < THRESH) | (arr[...,2] < THRESH)
    return mask.astype(np.uint8)

masks = [to_mask(im) for im in imgs]
n_views = len(masks)
angles = [2*math.pi*i/n_views for i in range(n_views)]  # assume even spacing

# Build voxel grid
xs = np.linspace(-1, 1, RES, dtype=np.float32)
ys = np.linspace(-1, 1, RES, dtype=np.float32)
zs = np.linspace(-Z_SCALE, Z_SCALE, RES, dtype=np.float32)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")

def project_inside(mask, theta):
    c, s = math.cos(theta), math.sin(theta)
    Xr = c*X - s*Y
    Yr = s*X + c*Y
    u = (Xr + 1)*0.5*(W-1)
    v = (1 - (Yr + 1)*0.5)*(H-1)
    ui = np.clip(np.rint(u).astype(np.int32), 0, W-1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, H-1)
    return mask[vi, ui] == 1

# Visual hull carving
occ = np.ones((RES, RES, RES), dtype=bool)
for mask, theta in zip(masks, angles):
    occ &= project_inside(mask, theta)

# Export results
np.save("visual_hull_occ.npy", occ)

# Point cloud
pts = np.stack([X[occ], Y[occ], Z[occ]], axis=1)
with open("visual_hull_points.ply", "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {pts.shape[0]}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("end_header\n")
    for x,y,z in pts:
        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

# Mesh if possible
if HAVE_SKIMAGE and occ.any():
    vol = occ.astype(np.float32)
    dx, dy, dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
    verts, faces, _, _ = marching_cubes(vol, 0.5, spacing=(dx,dy,dz))
    verts[:,0] += xs[0]
    verts[:,1] += ys[0]
    verts[:,2] += zs[0]
    with open("visual_hull_mesh.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

print(f"Processed {n_views} images")
print("Saved visual_hull_occ.npy, visual_hull_points.ply, and visual_hull_mesh.ply (if available).")
