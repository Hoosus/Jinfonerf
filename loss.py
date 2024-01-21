import jittor as jt
import jittor.nn as nn

def KL_loss(x, y):
  x = x / (jt.sum(x, -1, keepdims=True) + 1e-10) + 1e-10
  y = y / (jt.sum(y, -1, keepdims=True) + 1e-10) + 1e-10
  return jt.nn.KLDivLoss(reduction='batchmean')(x.log(), y)

def entropy_log2(prob):
  return -prob * jt.log2(prob+1e-10)

def EntropyLoss(sigma, acc):
  ray_prob = sigma / (jt.sum(sigma,-1).unsqueeze(-1)+1e-10)
  entropy_ray = entropy_log2(ray_prob)
  entropy_ray_loss = jt.sum(entropy_ray, -1)
  
  mask = (acc > 0.1).detach()
  entropy_ray_loss *= mask
  return jt.mean(entropy_ray_loss)