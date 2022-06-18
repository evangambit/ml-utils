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

class SwapAxes(nn.Module):
  def __init__(self, a, b):
    super(SwapAxes, self).__init__()
    self.a = a
    self.b = b

  def forward(self, x):
    return x.transpose(self.a, self.b)

class GaussianBlur2d(nn.Module):
  def __init__(self, c, kernel_size=5, stdev=1.0):
    super(GaussianBlur2d, self).__init__()
    if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)
    if type(stdev) is not tuple:
      stdev = (stdev, stdev)
    assert kernel_size[0] % 2 == 1
    assert kernel_size[1] % 2 == 1
    assert min(kernel_size) > 1
    assert min(stdev) >= 0.0
    assert max(stdev) > 0.0
    self.w = torch.zeros((c, 1) + kernel_size)
    for y in range(kernel_size[0]):
      for x in range(kernel_size[1]):
        d = 0.0
        if stdev[0] > 0.0:
          d += ((y - kernel_size[0] // 2) / stdev[0])**2
        else:
          d += 0.0 if abs(y - kernel_size[0] // 2) < 1e-3 else float('inf')
        if stdev[1] > 0.0:
          d += ((x - kernel_size[1] // 2) / stdev[1])**2
        else:
          d += 0.0 if abs(x - kernel_size[1] // 2) < 1e-3 else float('inf')
        self.w[:,:,y,x] = math.exp(d / -2.0)
    self.w /= self.w[0,0].sum()
    self.w = nn.Parameter(self.w, requires_grad=False)
    self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

  def forward(self, x):
    return nn.functional.conv2d(x, self.w, padding=self.padding, groups=self.w.shape[0])

class ResConv2d(nn.Module):
  def __init__(self, cin, cout, stride=1):
    super(ResConv2d, self).__init__()
    self.seq = nn.Sequential(
      nn.Conv2d(cin, cout, 3, padding=1, stride=stride),
      nn.BatchNorm2d(cout),
      nn.LeakyReLU(),
      nn.Conv2d(cout, cout, 3, padding=1),
      nn.BatchNorm2d(cout),
    )
    if cin == cout and stride == 1:
      self.shortcut = None
    elif cin == cout:
      self.shortcut = nn.AvgPool2d(stride)
    elif stride == 1:
      self.shortcut = nn.Sequential(
        nn.Conv2d(cin, cout, 1, 1, 0),
        nn.BatchNorm2d(cout),
      )
    else:
      self.shortcut = nn.Sequential(
        nn.AvgPool2d(stride),
        nn.Conv2d(cin, cout, 1, 1, 0),
        nn.BatchNorm2d(cout),
      )
    self.gate = nn.LeakyReLU()

  def forward(self, x):
    out = self.seq(x)
    if self.shortcut is not None:
      out += self.shortcut(x)
    else:
      out += x
    return self.gate(out)
