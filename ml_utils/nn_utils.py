import torch
from torch import nn
from torchvision import models

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
  def __init__(self, cin, cout, stride=1, gate1=nn.LeakyReLU, gate2=nn.LeakyReLU):
    super(ResConv2d, self).__init__()
    self.seq = nn.Sequential(
      nn.Conv2d(cin, cout, 3, padding=1, stride=stride),
      nn.BatchNorm2d(cout),
      gate1(),
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
    self.gate = gate2()

  def forward(self, x):
    out = self.seq(x)
    if self.shortcut is not None:
      out += self.shortcut(x)
    else:
      out += x
    return self.gate(out)

class Resnet(models.ResNet):
  """
  model = Resnet(block = models.resnet.Bottleneck, layers = [3, 4, 6, 3], groups=32, width_per_group=4)
  model.load_state_dict(models.resnet.resnext50_32x4d(weights=models.resnet.ResNeXt50_32X4D_Weights).state_dict())
  model.init(redset.tasks)
  """
  def __init__(self, *args, **kwargs):
    super(Resnet, self).__init__(*args, **kwargs)

  def init(self, tasks):
    emb_dim = model.fc.in_features
    delattr(self, 'fc')
    self._heads = nn.ModuleDict()
    for task in tasks:
      task_heads = task.create_heads(emb_dim)
      for head_name in task_heads:
        self._heads[head_name] = task_heads[head_name]

  def embed(self, x):
    assert x.shape[1:] == (3, 224, 224), f"{x.shape}"
    x = (x - 0.449) / 0.226
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x
  
  def head(self, x):
    r = {}
    for headName in self._heads:
      r[headName] = self._heads[headName](x)
    return r

  def forward(self, x):
    return self.head(self.embed(x))
