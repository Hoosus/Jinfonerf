import jittor as jt
from jittor import nn, Module

class Embedder:
  def __init__(self, multires):
    self.multires = multires
    self.freq_bands = 2.**jt.linspace(0., multires - 1, steps=multires)
    self.out_dim = (multires * 2 + 1) * 3 # multires (cos, sin), plus identity

  def embed(self, x):
    results = [x]
    for f in self.freq_bands:
      results.append(jt.cos(x * f))
      results.append(jt.sin(x * f))
    return jt.concat(results, dim=-1)

class NeRF(Module):
  def __init__ (self, depth=8, width=256, output_channel=3, skips=[4]):
    super(NeRF, self).__init__()
    self.embedder_pos = Embedder(10)
    self.embedder_dir = Embedder(4)
    input_channel = [self.embedder_pos.out_dim, self.embedder_dir.out_dim]
    self.depth = depth
    self.width = width
    self.input_channel = input_channel
    self.output_channel = output_channel
    self.skips = skips
    self.layers = [nn.Linear(input_channel[0], width)]
    for i in range(depth - 1):
      self.layers.append(nn.Linear(width + input_channel[0] if i in skips else width, width))
    self.alpha_layer = nn.Linear(width, 1)
    self.feature_layer = nn.Linear(width, width)
    self.view_fusion = nn.Linear(width + input_channel[1], width // 2)
    self.color_layer = nn.Linear(width // 2, output_channel)

  def execute(self, position, view):
    position, view = self.embedder_pos.embed(position), self.embedder_dir.embed(view)
    x, v = position, view
    for i, layer in enumerate(self.layers):
      x = nn.relu(layer(x))
      if i in self.skips:
        x = jt.concat([x, position], -1)
    alpha = self.alpha_layer(x)
    feature = self.feature_layer(x)
    x = nn.relu(self.view_fusion(jt.concat([feature, view], -1)))
    color = self.color_layer(x)
    return jt.concat([color, alpha], -1)

if __name__ == "__main__":
  a = NeRF()
  import numpy as np
  position = np.random.randn(1000, 3)
  views = np.random.randn(1000, 3)
  x = jt.float32(position)
  v = jt.float32(views)
  out = a(x, v)
  print(out)


      
