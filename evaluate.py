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
  depth = (depth / 5 * 255).astype(np.uint8)
  imageio.imwrite(dir, depth)

def psnr_from_mse(mse):
  return -10. * np.log10(mse)

def psnr_from_img(img, gt_img):
  mse = np.mean(np.square(img - gt_img))
  return psnr_from_mse(mse)

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

  psnr = 0
  psnr0 = 0

  for i in tqdm(range(N)):
    image_gt = images[i]
    pose_gt = poses[i]
    image_gt = image_gt[..., :3] * image_gt[..., -1:] + (1.0 - image_gt[..., -1:])
    H, W = image_gt.shape[:2]

    with jt.no_grad():
      coords = get_coords(H, W)
      rays_o, rays_d = get_rays(coords, pose_gt, camera)
      rgb, depth, rgb0, depth0 = render(model, model_fine, rays_o, rays_d, camera, render_settings, cpu=True)
      rgb_dir = os.path.join(log_dir, f"{i}_rgb.jpg")
      depth_dir = os.path.join(log_dir, f'{i}_depth.png')
      rgb0_dir = os.path.join(log_dir, f"{i}_rgb0.jpg")
      depth0_dir = os.path.join(log_dir, f'{i}_depth0.png')
      rgb = rgb.reshape(H, W, -1)
      rgb0 = rgb0.reshape(H, W, -1)
      depth = depth.reshape(H, W)
      depth0 = depth0.reshape(H, W)
      save_rgb(rgb, rgb_dir)
      save_depth(depth, depth_dir)
      save_rgb(rgb0, rgb0_dir)
      save_depth(depth, depth0_dir)
      psnr += psnr_from_img(rgb, image_gt)
      psnr0 += psnr_from_img(rgb0, image_gt)

  psnr /= N
  psnr0 /= N

  print(f"split {split} coarse psnr = {psnr0} fine psnr = {psnr}")

  model.train()
  model_fine.train()
