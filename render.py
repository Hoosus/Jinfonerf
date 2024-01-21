import numpy as np
import jittor as jt

def anaylze_output(output, t_vals, rays_d):
  d = t_vals[..., 1:] - t_vals[..., :-1] # (N, N_uniform - 1)
  d = jt.concat([d, jt.ones_like(d[..., :1]) * 1e10], dim=-1) # (N, N_uniform)
  d = d * (jt.norm(rays_d, dim=-1)[:, None]) # (N, N_uniform)

  raw_rgb = output[..., :3]
  raw_alpha = output[..., 3]

  rgb = jt.sigmoid(raw_rgb) # (N, N_uniform, 3)
  sigma = jt.nn.relu(raw_alpha)  # (N, N_uniform)
  alpha = 1.0 - jt.exp(-sigma * d) # (N, N_uniform)
  cum_alpha = jt.cumprod(alpha, dim=1) # (N, N_uniform)
  cum_alpha = jt.concat([jt.ones_like(cum_alpha[:, :1]), cum_alpha[:, :-1]], dim=-1) # (N, N_uniform)
  weights = alpha * cum_alpha # (N, N_uniform)

  # print("weights", weights.min(), weights.max())
  # print("rgb", rgb.min(), rgb.max())

  rgb_map = jt.sum(weights[..., None] * rgb, dim=1) # (N, 3)
  depth_map = jt.sum(weights * t_vals, dim=-1) # (N)
  occupancy_map = jt.sum(weights, -1) # (N)
  
  # print("occ", occupancy_map.min(), occupancy_map.max())

  disp_map = 1.0 / jt.maximum(depth_map / occupancy_map, 1e-10 * jt.ones(depth_map.shape[0]))

  rgb_map = rgb_map + (1.0 - occupancy_map)[..., None]
  
  # print("rgb_map", rgb_map.min(), rgb_map.max())

  return rgb_map, depth_map, occupancy_map, disp_map, weights, alpha


def query_network(model, x, v):
  shape = x.shape
  x = x.reshape(-1, shape[-1])
  v = v.reshape(-1, shape[-1])
  output = model(x, v)
  output = output.reshape(shape[:2] + [-1])
  return output

def get_rays(coords, pose, camera):
  H, W = camera['resolution']
  f = camera["focal"]
  rays_o = np.tile(pose[:3, -1].reshape(1, -1), (coords.shape[0], 1)) # (N, 3)
  coords_i = coords[:, 0]
  coords_j = coords[:, 1]
  rays_d = np.stack([(coords_i - 0.5 * W) / f, -(coords_j - 0.5 * H) / f, -np.ones(coords.shape[0])], -1) # (N, 3)
  return jt.Var(rays_o), jt.Var(rays_d)

def sample_pdf(mids, weights, N_importance, perturb):
  weights = weights + 1e-5 # (N, N_uniform - 2)
  pdf = weights / jt.sum(weights, -1, keepdims=True) # (N, N_uniform - 1)
  cdf = jt.cumsum(pdf, -1)
  cdf = jt.concat([jt.zeros(cdf.shape[:-1] + [1]), cdf], -1) # (N, N_uniform - 1)

  rn_shape = cdf.shape[:-1] + [N_importance]
  if perturb:
    rn = jt.linspace(0., 1., N_importance).expand(rn_shape).contiguous()
  else:
    rn = jt.rand(rn_shape).contiguous()
  
  inds = jt.searchsorted(cdf, rn, right=True)
  low = jt.maximum(jt.zeros_like(inds), inds - 1)
  high = jt.minimum(jt.ones_like(inds) * (cdf.shape[-1] - 1), inds) # (N, N_importance)
  inds = jt.stack([low, high], -1)

  matched_shape = inds.shape[:2] + [cdf.shape[-1]] # (N, N_importance, N_uniform - 1)
  cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds) # (N, N_importance, 2)
  bins_g = jt.gather(mids.unsqueeze(1).expand(matched_shape), 2, inds) 

  denom = cdf_g[...,1] - cdf_g[...,0]
  denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
  t = (rn - cdf_g[...,0]) / denom
  samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])

  return samples

def render(model, rays_o, rays_d, camera, args):
  H, W = camera['resolution']
  f = camera['focal']
  N = rays_o.shape[0]
  v = jt.normalize(rays_d, dim=-1) # (N, 3)

  N_uniform = args["N_uniform"]
  N_importance = args["N_importance"]
  perturb = args["perturb"]

  t_vals = jt.linspace(0., 1., N_uniform).repeat(N, 1) # (N, N_uniform)
  if perturb:
    mids = (t_vals[...,1:] + t_vals[...,:-1]) / 2.0
    upper = jt.concat([mids, t_vals[...,-1:]], -1)
    lower = jt.concat([t_vals[...,:1], mids], -1)
    t_rand = jt.rand(N, N_uniform)
    t_vals = lower + (upper - lower) * t_rand

  x = rays_o[:, None, :] + t_vals[:, :, None] * rays_d[:, None, :] # (N, N_uniform, 3)
  v = v.repeat(1, N_uniform, 1) # (N, N_uniform, 3)

  output = query_network(model, x, v)
  rgb_map, depth_map, occupancy_map, disp_map, weights, alpha = anaylze_output(output, t_vals, rays_d)

  if N_importance > 0:
    rgb_map_0, disp_map_0, occupancy_map_0 = rgb_map, disp_map, occupancy_map
    mids = (t_vals[..., 1:] + t_vals[..., :-1]) / 2.0 # (N, N_uniform - 1)
    t_samples = sample_pdf(mids, weights[..., 1:-1], N_importance, perturb)
    print("not finished!")
  
  results = {
    "rgb": rgb_map,
    "depth": depth_map,
    "occupancy": occupancy_map,
    "disp": disp_map,
    "alpha": alpha
  }
  
  return results

def get_near_pose(poses):
  near_trans = 0.1
  phi = 5 * (np.pi / 180.)

  X = np.random.rand(3) * 2 * phi - phi # (-phi, phi)
  x, y, z = X

  rot_x = jt.array([[1, 0, 0],
                    [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)]])
  rot_y = jt.array([[np.cos(y), 0, -np.sin(y)],
                    [0, 1, 0],
                    [np.sin(y), 0, np.cos(y)]])
  rot_z = jt.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]])
  rot_mat = jt.matmul(rot_x, jt.matmul(rot_y, rot_z))
  return rot_mat







