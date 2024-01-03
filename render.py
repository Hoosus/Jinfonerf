import numpy as np
import jittor as jt

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

def render(model, rays_o, rays_d, camera, args):
  H, W = camera['resolution']
  f = camera['focal']
  N = rays_o.shape[0]
  v = jt.normalize(rays_d, dim=-1) # (N, 3)

  N_uniform = args["N_uniform"]
  N_importance = args["N_importance"]
  perturb = args["perturb"]

  t_vals = jt.linspace(0., 1., N_uniform).repeat(N, 1) # (N, N_uniform)
  if perturb > 0:
    mids = (t_vals[...,1:] + t_vals[...,:-1]) / 2.0
    upper = jt.concat([mids, t_vals[...,-1:]], -1)
    lower = jt.concat([t_vals[...,:1], mids], -1)
    t_rand = jt.rand(N, N_uniform)
    t_vals = lower + (upper - lower) * t_rand

  print(rays_o.shape, t_vals.shape, rays_d.shape)

  x = rays_o[:, None, :] + t_vals[:, :, None] * rays_d[:, None, :] # (N, N_uniform, 3)
  v = v.repeat(1, N_uniform, 1) # (N, N_uniform, 3)

  output = query_network(model, x, v)

  print(output.shape)






