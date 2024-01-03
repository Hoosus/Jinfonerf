import jittor as jt
from jittor import nn, Module

class NeRF(Module):
  def __init__ (self, depth=8, width=256, input_channel=[3, 3], output_channel=4, skips=[4]):
    super(NeRF, self).__init__()
    self.depth = depth
    self.width = width
    self.input_channel = input_channel
    self.output_channel = output_channel
    self.skips = skips
    self.layers = [nn.Linear(input_channel[0], width)]
    for i in range(depth - 1):
      self.layers.append(nn.Linear(width + input_channel[1] if i in skips else width, width))
    self.alpha_layer = nn.Linear(width, 1)
    self.feature_layer = nn.Linear(width, width)
    self.view_fusion = nn.Linear(width + input_channel[1], width // 2)
    self.color_layer = nn.Linear(width // 2, output_channel)

  def execute(self, position, view):
    x, v = position, view
    for i, layer in enumerate(self.layers):
      x = nn.relu(layer(x))
      if i in self.skips:
        x = jt.concat([x, position], -1)
    alpha = self.alpha_layer(x)
    feature = self.feature_layer(x)
    x = self.view_fusion(jt.concat([feature, v], -1))
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


      
