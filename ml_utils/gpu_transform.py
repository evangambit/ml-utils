import torch
from torch import nn

class GridAugment(nn.Module):
  def __init__(self, batch_size, size,
               horizontal_flip=0.5,
               rotate_stdev=[1.0, 0.0, 0.0],
               scale_stdev=0.3,
               warp_std=0.1
               ):
    super(GridAugment, self).__init__()

    self.batch_size = batch_size
    self.size = list(size)

    self.horizontal_flip = horizontal_flip
    self.rotate_stdev = rotate_stdev
    self.scale_stdev = scale_stdev
    self.warp_std = warp_std

    self.Zs = torch.ones([batch_size, 1] + size)
    self.Ys = torch.zeros([batch_size, 1] + size)
    self.Xs = torch.zeros([batch_size, 1] + size)
    with torch.no_grad():
      for y in range(size[0]):
        for x in range(size[1]):
          self.Ys[:,0,y,x] = y / (size[0] - 1) * 2 - 1
          self.Xs[:,0,y,x] = x / (size[1] - 1) * 2 - 1
    self.Zs = nn.Parameter(self.Zs, requires_grad=False)
    self.Ys = nn.Parameter(self.Ys, requires_grad=False)
    self.Xs = nn.Parameter(self.Xs, requires_grad=False)

    self.lastParams = nn.ParameterList([
      # rotating
      nn.Parameter(torch.zeros((batch_size, 1, 1, 1)), requires_grad=False),
      nn.Parameter(torch.zeros((batch_size, 1, 1, 1)), requires_grad=False),
      nn.Parameter(torch.zeros((batch_size, 1, 1, 1)), requires_grad=False),

      # scaling
      nn.Parameter(torch.zeros((batch_size, 1, 1, 1)), requires_grad=False),

      # warping
      nn.Parameter(torch.zeros((batch_size, 1, 5, 5)), requires_grad=False),
      nn.Parameter(torch.zeros((batch_size, 1, 5, 5)), requires_grad=False),

      # horizontal flip
      nn.Parameter(torch.zeros((batch_size, 1, 1, 1)), requires_grad=False),
    ])

  @staticmethod
  def rotate(x, y, theta):
    tmp = torch.cos(theta) * x - torch.sin(theta) * y
    return tmp, torch.cos(theta) * y + torch.sin(theta) * x

  def forward(self, input_):
    x, y, z = self.Xs, self.Ys, self.Zs

    # theta1, theta2, theta3, scale, warpx, warpy
    with torch.no_grad():
      self.lastParams[0].normal_(mean=0.0, std=self.rotate_stdev[0])
      self.lastParams[1].normal_(mean=0.0, std=self.rotate_stdev[1])
      self.lastParams[2].normal_(mean=0.0, std=self.rotate_stdev[2])
      self.lastParams[3].normal_(mean=0.0, std=self.scale_stdev)
      self.lastParams[4].normal_(mean=0.0, std=self.warp_std)
      self.lastParams[5].normal_(mean=0.0, std=self.warp_std)
      self.lastParams[6].uniform_()

    if self.horizontal_flip > 0.0:
      x = x * (1 - 2 * (self.lastParams[6] > 1.0 - self.horizontal_flip))

    if self.rotate_stdev[0] > 0.0:
      x, y = GridAugment.rotate(x, y, self.lastParams[0] / 4)
    if self.rotate_stdev[1] > 0.0:
      y, z = GridAugment.rotate(y, z, self.lastParams[1] / 8)
    if self.rotate_stdev[2] > 0.0:
      z, x = GridAugment.rotate(z, x, self.lastParams[2] / 8)

    if self.scale_stdev > 0.0:
      z = z * torch.exp(self.lastParams[3])

    if self.warp_std > 0.0:
      x = x + nn.functional.upsample_bilinear(self.lastParams[4], size=self.size)
      y = y + nn.functional.upsample_bilinear(self.lastParams[5], size=self.size)

    x = (x / z).reshape([self.batch_size] + self.size)
    y = (y / z).reshape([self.batch_size] + self.size)
    grid = torch.stack([x, y], 3)

    return nn.functional.grid_sample(input_, grid)

class RandomColorChange(nn.Module):
  def __init__(self, batch_size, num_channels, brightness=0.1):
    super(RandomColorChange, self).__init__()
    self.brightness = brightness
    self.params = nn.ParameterList([
      nn.Parameter(torch.zeros((batch_size, num_channels, 1, 1)), requires_grad=False),
    ])
  def forward(self, x):
    with torch.no_grad():
      self.params[0].normal_(mean=1.0, std=self.brightness)    
    return (x * self.params[0]).clip(0, 1)

class AddAlpha(nn.Module):
  def __init__(self, batch_size, image_size):
    super(AddAlpha, self).__init__()
    self.params = nn.ParameterList([
      nn.Parameter(torch.ones([batch_size, 1] + image_size), requires_grad=False),
    ])
  def forward(self, x):
    return torch.cat([x, self.params[0]], 1)

class CenterCropper(nn.Module):
  def __init__(self, k):
    super(CenterCropper, self).__init__()
    self.k = k
  def forward(self, x):
    return x[:,:,self.k:-self.k,self.k:-self.k]

