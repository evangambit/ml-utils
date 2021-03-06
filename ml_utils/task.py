import torch
from torch import nn

import numpy as np

import math

def logit(x):
  if isinstance(x, np.ndarray):
    return np.log(x / (1.0 - x))
  elif isinstance(x, torch.Tensor):
    return torch.log(x / (1.0 - x))
  else:
    return math.log(x / (1.0 - x))

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


class Task:
  def __init__(self, name):
    self.name = name
    self.default = None

  def loss(self, predictions, batch):
    """
    Returns a 0-D Pytorch tensor.
    """
    pass

  def metrics(self, predictions, batch, it, prefix = ''):
    """
    Returns a dictionary where keys are strings
    and values are tuples of the form

    (int sample_size, float first_moment, float second_moment)
    """
    pass

class ClassificationTask(Task):
  def __init__(self, name, classes):
    Task.__init__(self, name)
    self.classes = classes
    self._loss = nn.CrossEntropyLoss(reduction='none')
    self.default = torch.tensor(0, dtype=torch.int64)

  def create_heads(self, din):
    head = nn.Linear(din, len(self.classes))
    with torch.no_grad():
      b = logit(1.0 / len(self.classes))
      head.bias.zero_()
      head.bias += b
      head.weight.zero_()
    return {
      self.name: head
    }

  def loss(self, predictions, batch):
    yhat = predictions[self.name]
    y = batch[self.name]
    mask = batch[self.name + '?']
    mask_sum = float(mask.sum())

    l = self._loss(yhat, y)

    if mask_sum == 0.0:
      return 0.0
    return (l * mask).sum() / mask_sum

  def metrics(self, predictions, batch, it, prefix = ''):
    yhat = predictions[self.name]
    y = batch[self.name]
    mask = batch[self.name + '?']
    mask_sum = int(mask.sum())
    if mask_sum == 0:
      return {}

    l = self._loss(yhat, y)

    incorrect = (yhat.argmax(1) != y).to(torch.float32)

    return {
      f"{prefix}:{self.name}:loss": (
        it,
        mask_sum,
        float((l * mask).sum()) / mask_sum,
        float((l * l * mask).sum()) / mask_sum,
      ),
      f"{prefix}:{self.name}:error": (
        it,
        mask_sum,
        float((incorrect * mask).sum()) / mask_sum,
        float((incorrect * mask).sum()) / mask_sum,
      ),
    }

class RegressionTask(Task):
  """
  Represents a 1D regression task.

  Neural network outputs are expected to have a mean of 0 and
  a standard deviation of 1. This task scales the labels to match
  this. The reasoning is that if we don't normalize the labels,
  then the gradients between different tasks could have very different
  magnitudes.
  """
  def __init__(self, name, avg = 0.0, std = 1.0):
    Task.__init__(self, name)
    self.avg = avg
    self.std = std
    self.default = torch.tensor(0.0, dtype=torch.float32)
    self._loss = nn.MSELoss(reduction='none')

  def create_heads(self, din):
    head = nn.Linear(din, 1)
    with torch.no_grad():
      head.bias.zero_()
      head.weight.zero_()
    return {
      self.name: nn.Sequential(
        head,
        Reshape(-1),
      )
    }

  def loss(self, predictions, batch):
    yhat = predictions[self.name]
    y = (batch[self.name] - self.avg) / self.std
    mask = batch[self.name + '?']
    mask_sum = float(mask.sum())

    l = self._loss(yhat, y)

    if mask_sum == 0.0:
      return 0.0
    return (l * mask).sum() / mask_sum

  def metrics(self, predictions, batch, it, prefix = ''):
    yhat = predictions[self.name]
    y = (batch[self.name] - self.avg) / self.std
    mask = batch[self.name + '?']
    mask_sum = int(mask.sum())
    if mask_sum == 0:
      return {}

    l = self._loss(yhat, y)

    incorrect = (torch.abs(yhat - y) > 1.0).to(torch.float32)

    return {
      f"{prefix}:{self.name}:loss": (
        it,
        mask_sum,
        float((l * mask).sum()) / mask_sum,
        float((l * l * mask).sum()) / mask_sum,
      ),
      f"{prefix}:{self.name}:wrong": (
        it,
        mask_sum,
        float((incorrect * mask).sum()) / mask_sum,
        float((incorrect * mask).sum()) / mask_sum,
      ),
    }

class HydraHeads(nn.Module):
  def __init__(self, backbone, dim, tasks):
    super().__init__()
    self.backbone = backbone
    self.heads = nn.ModuleDict()
    for task in tasks:
      heads = task.create_heads(dim)
      for k in heads:
        self.heads[k] = heads[k]

  def forward(self, x):
    z = self.backbone((x - 0.449) / .226)
    r = {}
    for k in self.heads:
      r[k] = self.heads[k](z)
    return r

def hydra_resnet(resnet, tasks):
  """
  Given a torchvision resnet model and a list of tasks, creates
  a hydranet from the list of tasks.
  """
  seq = nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1,
    resnet.layer2,
    resnet.layer3,
    resnet.layer4,
    resnet.avgpool,
    BatchReshape(-1),
  )
  dim = resnet.layer4[-1].conv2.weight.shape[0]
  return HydraHeads(seq, dim, tasks)
