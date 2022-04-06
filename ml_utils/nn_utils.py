import torch
from torch import nn

class Mean(nn.Module):
  def __init__(self, *axes):
    super(Mean, self).__init__()
    self.axes = axes

  def forward(self, x):
    return x.mean(self.axes)

class Reshape(nn.Module):
  def __init__(self, *shape):
    super(Reshape, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.reshape(self.shape)

class BatchReshape(nn.Module):
  def __init__(self, *shape):
    super(BatchReshape, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.reshape((x.shape[0],) + self.shape)

class ResConv2d(nn.Module):
  def __init__(self, cin, cout, stride=1):
    super(ResConv2d, self).__init__()
    self.seq = nn.Sequential(
      conv3(cin, cout, stride),
      gate(),
      conv3(cout, cout, 1),
    )
    if cin == cout and stride == 1:
      self.shortcut = None
    elif cin == cout:
      self.shortcut == nn.AvgPool2d(stride)
    elif stride == 1:
      self.shortcut = nn.Sequential(
        nn.Conv2d(cin, cout, 1, 1, 0),
        BatchNorm(cout),
      )
    else:
      self.shortcut = nn.Sequential(
        nn.AvgPool2d(stride),
        nn.Conv2d(cin, cout, 1, 1, 0),
        BatchNorm(cout),
      )
    self.gate = gate()

  def forward(self, x):
    out = self.seq(x)
    if self.shortcut is not None:
      out += self.shortcut(x)
    else:
      out += x
    return self.gate(out)
