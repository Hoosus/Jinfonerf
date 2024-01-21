import os
from pathlib import Path
from tqdm import tqdm
import jittor as jt
import imageio
import numpy as np

from render import get_rays, render, get_coords

def save_rgb(rgb, dir):
  rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
  imageio.imwrite(dir, rgb)

def save_depth(depth, dir):
  print("max:", np.nanmax(depth))
  depth = (depth / 5 * 255).astype(np.uint8)
  imageio.imwrite(dir, depth)

def evaluate(iter, log_path, model, model_fine, dataset, render_settings):
  model.eval()
  model_fine.eval()

  split = dataset["split"]
  log_dir = os.path.join(log_path, f"{split}_iter{iter}")
  Path(log_dir).mkdir(parents=True, exist_ok=True)
  
  images = dataset["imgs"]
  poses = dataset["poses"]
  camera = dataset["camera_setting"]

  N = images.shape[0]

  for i in tqdm(range(N)):
    image_gt = images[i]
    pose_gt = poses[i]
    image_gt = image_gt[..., :3] * image_gt[..., -1:] + (1.0 - image_gt[..., -1:])
    H, W = image_gt.shape[:2]
    H, W = 256, 256

    with jt.no_grad():
      coords = get_coords(H, W)
      rays_o, rays_d = get_rays(coords, pose_gt, camera)
      render_results = render(model, model_fine, rays_o, rays_d, camera, render_settings)
      rgb_dir = os.path.join(log_dir, f"{i}_rgb.jpg")
      depth_dir = os.path.join(log_dir, f'{i}_depth.png')
      save_rgb(render_results["rgb"].reshape(H, W, -1).cpu().numpy(), rgb_dir)
      save_depth(render_results["depth"].reshape(H, W).cpu().numpy(), depth_dir)

  model.train()
  model_fine.train()
