import numpy as np
import jittor as jt
import yaml
from tqdm import tqdm
import copy
import os
from pathlib import Path

from data import load_lego
from utils import get_datetime_str, parser_parse_args
from model import NeRF
from render import get_rays, render, get_near_pose, get_coords
from evaluate import evaluate
from loss import KL_loss, EntropyLoss

def train(config, model, model_fine, dataset, log_path):
  train_dataset = dataset["train"]
  train_dataset_size = train_dataset["imgs"].shape[0]

  optimizer = jt.optim.Adam(model.parameters() + model_fine.parameters(), lr=config["lr"], betas=(0.9, 0.999))
  render_settings = {
    "N_uniform": config["N_uniform"],
    "N_importance": config["N_importance"],
    "perturb": config["perturb"]
  }
  render_settings_test = copy.deepcopy(render_settings)
  render_settings_test["perturb"] = False

  model.train()
  N_iter = config["iteration"]
  for i in tqdm(range(N_iter)):
    N_seen = config["N_seen"]
    if N_seen > 0:  
      # ------- Select One Image -------
      if config["fewshot"]:
        img_id = np.random_choice(config["fewshot_train"])
      else:
        img_id = np.random.choice(train_dataset_size)

      # img_id = 0

      image_gt = train_dataset["imgs"][img_id]
      image_gt = image_gt[..., :3] * image_gt[...,-1:] + (1.0 - image_gt[..., -1:])
      pose_gt = train_dataset["poses"][img_id, :3, :4]
      H, W = image_gt.shape[:2]

      # TODO
      # print("Center Cropping Not Implemented!")
      pass

      coords = get_coords(H, W)

      # ------- Select Random Rays -------
      samples = np.random.choice(coords.shape[0], N_seen, replace=False)

      # samples = np.arange(N_seen)

      coords = coords[samples] # (N, 2)
      # print(coords)
      target = image_gt[coords[:, 0], coords[:, 1]] # (N, 3)
      
      rays_o, rays_d = get_rays(coords, pose_gt, train_dataset["camera_setting"])

      # print(rays_o, rays_d)

      if config["smoothing"]:
        near_pose_seen = get_near_pose(pose_gt)
        rays_o_near_seen, rays_d_near_seen = get_rays(coords, near_pose_seen, train_dataset["camera_setting"])

      # from evaluate import save_rgb
      # save_rgb(image_gt, os.path.join(log_path, "train_gt.jpg"))

    N_unseen = config["N_unseen"]
    if N_unseen > 0:
      # TODO
      img_id_un = np.random.choice(train_dataset_size)
      
      image_gt_un = train_dataset["imgs"][img_id_un]
      image_gt_un = image_gt_un[..., :3] * image_gt_un[...,-1:] + (1.0 - image_gt_un[..., -1:])
      pose_gt_un = train_dataset["poses"][img_id_un, :3, :4]
      H, W = image_gt_un.shape[:2]

      # center cropping not Implemented
      # TODO

      coords_un = get_coords(H, W)

      # ------- Select Random Rays -------
      samples = np.random.choice(coords_un.shape[0], N_unseen, replace=False)
      coords_un = coords_un[samples]
      rays_o_un, rays_d_un = get_rays(coords_un, pose_gt_un, train_dataset["camera_setting"])

      if config["smoothing"]:
        near_pose_un = get_near_pose(pose_gt_un)
        rays_o_near_un, rays_d_near_un = get_rays(coords_un, near_pose_un, train_dataset["camera_setting"])

    render_results = render(model, model_fine, rays_o, rays_d, train_dataset["camera_setting"], render_settings)
    render_rgb = render_results["rgb"]

    if config["smoothing"]:
      render_results_near_seen = render(model, model_fine, rays_o_near_seen, rays_d_near_seen, train_dataset["camera_setting"], render_settings)

    if config["N_unseen"] > 0:
      render_results_un = render(model, model_fine, rays_o_un, rays_d_un, train_dataset["camera_setting"], render_settings)
      if config["smoothing"]:
        render_results_near_un = render(model, model_fine, rays_o_near_un, rays_d_near_un, train_dataset["camera_setting"], render_settings)

    optimizer.zero_grad()
    loss = jt.mean(jt.sqr(render_rgb - target))
    print(f"iteration {i}, MSELoss = {loss.item()}")
    if config["N_importance"] > 0:
      mse_loss0 = jt.mean(jt.sqr(render_results["rgb0"] - target))
      print(f"iteration {i}, Coarse MSELoss = {mse_loss0.item()}")
      loss += mse_loss0
    
    # entropy loss
    if config["N_unseen"] > 0:
      entropy_lambda = config["entropy_lambda"]
      # occ_raw = render_results["occupancy"]
      occ_raw_un = render_results_un["occupancy"]
      # alpha_raw = render_results["alpha"]
      alpha_raw_un = render_results_un["alpha"]
      entropy_loss = EntropyLoss(alpha_raw_un, occ_raw_un)
      print(f"iteration {i}, Entropy Loss = {entropy_loss.item()}")
      loss += entropy_lambda * entropy_loss

    smoothing_lambda = 0.00002 * 0.5 ** (int(i / 5000))
    if config["smoothing"]:
      smoothing_loss = KL_loss(render_results["alpha"], render_results_near_seen["alpha"])
      print(f"iteration {i}, Seen Smoothing Loss = {smoothing_loss.item()}")
      loss += smoothing_lambda * smoothing_loss

    if config["N_unseen"] > 0 and config["smoothing"]:
      smoothing_loss_un = KL_loss(render_results_un["alpha"], render_results_near_un["alpha"])
      print(f"iteration {i}, Unseen Smoothing Loss = {smoothing_loss_un.item()}")
      loss += smoothing_lambda * smoothing_loss_un

    optimizer.step(loss)

    # ------- Update Learning Rate -------
    decay_rate = 0.1
    decay_it = config["lr_decay_it"]
    new_lrate = config["lr"] * (decay_rate ** (i / decay_it))
    # print("new learning rate is", new_lrate)
    for param_group in optimizer.param_groups:
      param_group['lr'] = new_lrate

    if i % config["evaluate_iter"] == 0 and i > 0:
      evaluate(i, log_path, model, model_fine, dataset["val"], render_settings_test)
      evaluate(i, log_path, model, model_fine, dataset["test"], render_settings_test)

if __name__ == "__main__":
  args = parser_parse_args()
  config_path = args.config
  if isinstance(config_path, list):
    config_path = config_path[0]
  config = yaml.safe_load(open(config_path))
  print(config)
  name = config["name"]
  assert name in ["lego"]
  data_path = config["data_path"]
  exp_name = config["exp_name"]
  log_path = f"./logs/{exp_name}_{get_datetime_str()}"
  Path(log_path).mkdir(parents=True, exist_ok=False)

  if name == "lego":
    dataset = {
      'train' : load_lego(data_path, 'train', samples=None),#[26, 86, 2, 55]),
      'val' : load_lego(data_path, 'val', skip=8),
      'test' : load_lego(data_path, 'test', skip=8)
    }
  
  jt.flags.use_cuda = 2

  model = NeRF()
  model_fine = NeRF()
  
  train(config, model, model_fine, dataset, log_path)
  
  
  
