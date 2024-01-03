import numpy as np
import jittor as jt
import yaml
from data import load_lego
from utils import get_datetime_str, parser_parse_args
from model import NeRF
from render import get_rays, render

def train(config, model, dataset, log_path):
  train_dataset = dataset["train"]
  train_dataset_size = train_dataset["imgs"].shape[0]

  optimizer = jt.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
  render_settings = {
    "N_uniform": config["N_uniform"],
    "N_importance": config["N_importance"],
    "perturb": config["perturb"]
  }

  model.train()
  N_iter = config["iteration"]
  for i in range(N_iter):
    N_seen = config["N_seen"]
    if N_seen > 0:  
      # ------- Select One Image -------
      img_id = np.random.choice(train_dataset_size)
      image_gt = train_dataset["imgs"][img_id]
      pose_gt = train_dataset["poses"][img_id]
      H, W = image_gt.shape[:2]

      # TODO
      print("Center Cropping Not Implemented!")

      coords_i, coords_j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
      coords_i = coords_i.reshape(-1, 1) # (H*W, 1)
      coords_j = coords_j.reshape(-1, 1) # (H*W, 1)
      coords = np.concatenate((coords_i, coords_j), axis=-1)

      # ------- Select Random Rays -------
      samples = np.random.choice(coords.shape[0], N_seen, replace=False)
      coords = coords[samples] # (N, 3)
      target = image_gt[coords[:, 0], coords[:, 1]] # (N, 3)
      
      rays_o, rays_d = get_rays(coords, pose_gt, train_dataset["camera_setting"])

      if config["smoothing"]:
        # TODO
        print("Smoothing Not Implemented!")
        # See GetNearC2W in utils/generate_near_c2w.py



    N_unseen = config["N_unseen"]
    if N_unseen > 0:
      # TODO
      print("Sampling Unseen Rays Not Implemented")

    # TODO
    all_rays_o = rays_o
    all_rays_d = rays_d

    render_rgb = render(model, all_rays_o, all_rays_d, train_dataset["camera_setting"], render_settings)

    optimizer.zero_grad()
    mse_loss = jt.mean(jt.sqr((render_rgb - target)))
    # TODO: add more losses 

    loss = mse_loss
    print(f"iteration {i}, MSELoss = {mse_loss.item()}")

    loss.backward()
    optimizer.step()


    # ------- Update Learning Rate -------
    decay_rate = 0.1
    decay_it = config["lr_decay_it"]
    new_lrate = args["lr"] * (decay_rate ** (i / decay_it))
    print("new learning rate is", new_lrate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    # TODO: delete this
    break

if __name__ == "__main__":
  args = parser_parse_args()
  config_path = args.config
  config = yaml.safe_load(open(config_path))
  print(config)
  name = config["name"]
  assert name in ["lego"]
  data_path = config["data_path"]
  exp_name = config["exp_name"]
  log_path = f"./logs/{exp_name}_{get_datetime_str()}"

  if name == "lego":
    dataset = {
      'train' : load_lego(data_path, 'train', samples=[26, 86, 2, 55]),
      'val' : load_lego(data_path, 'val', skip=8),
      'test' : load_lego(data_path, 'test', skip=8)
    }
  
  model = NeRF()
  
  train(config, model, dataset, log_path)
  
  
  
