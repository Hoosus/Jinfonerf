import jittor as jt

def KL_loss(x, y):
  x = x / (jt.sum(x, -1, keepdims=True) + 1e-10) + 1e-10
  y = y / (jt.sum(y, -1, keepdims=True) + 1e-10) + 1e-10
  return jt.nn.KLDivLoss(reduction='batchmean')(x.log(), y)