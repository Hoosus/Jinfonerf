import jittor as jt
import numpy as np
import cv2
import os
from tqdm import tqdm
import imageio

from data import load_lego
from render import render, get_coords, get_rays
from model import NeRF
from evaluate import psnr_from_img

def lookat(position, target, up): 
  forward = np.subtract(target, position)
  forward = np.divide(forward, np.linalg.norm(forward))

  right = np.cross(forward, up)
  
  # if forward and up vectors are parallel, right vector is zero; 
  #   fix by perturbing up vector a bit
  if np.linalg.norm(right) < 0.001:
      epsilon = np.array([0.001, 0, 0])
      right = np.cross(forward, up + epsilon)
      
  right = np.divide(right, np.linalg.norm(right))
  
  up = np.cross(right, forward)
  up = np.divide(up, np.linalg.norm(up))
  
  return np.array([[right[0], up[0], -forward[0], position[0]], 
                   [right[1], up[1], -forward[1], position[1]], 
                   [right[2], up[2], -forward[2], position[2]]]) 

def render_perspective(model, model_fine, dataset, name):
  model.eval()
  model_fine.eval()

  split = dataset["split"]
  images = dataset["imgs"]
  poses = dataset["poses"]
  camera = dataset["camera_setting"]

  N = images.shape[0]
  H, W = images.shape[1:3]

  render_settings = {
    "N_uniform": 64,
    "N_importance": 128,
    "perturb": False
  }

  total_frame = 120
  # fps = 1
  # out = cv2.VideoWriter(f"./results/{name}_perspective.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (H, W), False)
  for i in tqdm(range(total_frame)):
    phi = i / total_frame * 2 * (2 * np.pi)
    theta = i / total_frame * 60 / 180 * np.pi
    camera_pos = 4.0 * np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])
    print(i, camera_pos)
    pose = lookat(camera_pos, np.zeros(3), np.array([0., 0., 1.]))
    with jt.no_grad():
      coords = get_coords(H, W)
      rays_o, rays_d = get_rays(coords, pose, camera)
      rgb, depth, rgb0, depth0 = render(model, model_fine, rays_o, rays_d, camera, render_settings, cpu=True)

      rgb = rgb.reshape(H, W, -1)
      rgb0 = rgb0.reshape(H, W, -1)
      depth = depth.reshape(H, W)
      depth0 = depth0.reshape(H, W)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    # out.write(rgb)
    imageio.imwrite(f"./results/{name}_perspective_{i}.png", rgb)
  # out.release()

def render_training(log_dir, name, iters, viewid):
  total = 120
  fps = 10
  out = cv2.VideoWriter(f"./results/{name}_training_{viewid}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)
  for iter in iters:
    array = imageio.imread(os.path.join(log_dir, f"test_iter{iter}/{viewid}_rgb.jpg"))
    out.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
  out.release()

def getpsnr(dataset, dir):
  split = dataset["split"]
  images = dataset["imgs"]

  dir = os.path.join(dir, f"{split}_iter50000")

  psnr = 0
  N = images.shape[0]

  for i in range(N):
    image = images[i]
    image = image[..., :3] * image[..., -1:] + (1.0 - image[..., -1:])
    rgb = imageio.imread(os.path.join(dir, f"{i}_rgb.jpg"))
    rgb = rgb.astype(np.float32) / 255.0
    psnr += psnr_from_img(rgb, image)

  psnr /= N  
  return psnr

if __name__ == "__main__":
  jt.flags.use_cuda = 2

  model = NeRF()
  model_fine = NeRF()

  data_path = "./data/lego"
  dataset = {
    'train' : load_lego(data_path, 'train', samples=None),#[26, 86, 2, 55]),
    'val' : load_lego(data_path, 'val', skip=8),
    'test' : load_lego(data_path, 'test', skip=8)
  }

  name = "final"
  log_dir = f"./logs/lego_{name}"

  model.load_state_dict(jt.load(os.path.join(log_dir, "ckpt_50000.pkl")))
  model_fine.load_state_dict(jt.load(os.path.join(log_dir, "ckpt_fine_50000.pkl")))

  val_psnr = getpsnr(dataset["val"], log_dir)
  test_psnr = getpsnr(dataset["test"], log_dir)
  print(f"val psnr = {val_psnr} test psnr = {test_psnr}")
  for i in range(10):
    if name == "final":
      render_training(log_dir, name, range(2500, 50001, 2500), i)
    else:
      render_training(log_dir, name, range(5000, 50001, 5000), i)
  render_perspective(model, model_fine, dataset["val"], name)
  
  




