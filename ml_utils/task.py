import torch
from torch import nn
import torch.utils.data as tdata
import numpy as np
from scipy.ndimage import uniform_filter1d

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
    self.gTaskId = Task.taskId
    Task.taskId += 1

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

  @staticmethod
  def plot_helper(A, smooth):
    time = np.array([a[0] for a in A])
    samplesize = [a[1] for a in A]
    mom1 = np.array([a[2] for a in A]) * samplesize
    mom2 = np.array([a[2] for a in A]) * samplesize

    N = uniform_filter1d(samplesize, smooth)
    avg = uniform_filter1d(mom1, smooth) / N
    var = np.maximum(uniform_filter1d(mom2, smooth) - uniform_filter1d(mom1**2, smooth), 0.0) / N

    return (time, samplesize, avg, var)

Task.taskId = 0

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

class DetectionTasks(ml_utils.task.Task):
  def __init__(self, name, classes):
    ml_utils.task.Task.__init__(self, name)
    assert isinstance(self.classes, list)
    assert isinstance(self.classes[0], str)
    self.classes = classes
    self.default = torch.tensor([0.5] * len(classes), dtype=torch.float32)
    self.logSigmoid = nn.LogSigmoid()

  def create_heads(self, din):
    head = nn.Linear(din, len(self.classes))
    with torch.no_grad():
      head.bias.zero_()
      head.weight.zero_()
    return {
      self.name: head
    }

  def _loss(self, predictions, batch):
    yhat = predictions[self.name]
    y = batch[self.name]
    assert yhat.shape == y.shape
    return y * self.logSigmoid(yhat) + (1 - y) * self.logSigmoid(-yhat)

  def loss(self, predictions, batch):
    l = self._loss(predictions, batch)
    mask = batch[self.name + '?']
    loss_per_class = (l * mask).sum(0) * torch.nan_to_num(1.0 / mask.sum(0), posinf=0.0)
    return loss_per_class.sum()

  def metrics(self, predictions, batch, it, prefix = ''):
    yhat = predictions[self.name]
    y = batch[self.name]
    mask = batch[self.name + '?']
    if mask_sum == 0:
      return {}

    # Compute two (batch_size x num_classes) matrices.
    l = self._loss(yhat, y)
    incorrect = (torch.round(torch.sigmoid(yhat)).to(torch.int64) != y).to(torch.float32)

    r = {}
    for i, klass in enumerate(self.classes):
      n = int(mask[:,i].sum())
      inv_n = 0.0 if n == 0 else 1.0 / n
      r[f"{prefix}:{klass}:loss"] = (
        it,
        n,
        float((l[:,i] * mask).sum()) * inv_n,
        float(((l[:,i]**2) * mask).sum()) * inv_n,
      )
      r[f"{prefix}:{klass}:error"] = (
        it,
        n,
        float((l[:,i] * incorrect).sum()) * inv_n,
        float(((l[:,i]**2) * incorrect).sum()) * inv_n,
      )

    return r

class ConcatedDataset(tdata.Dataset):
  def __init__(self, *datasets):
    super().__init__()
    assert len(datasets) < 20, 'This class is not designed for large numbers of datasets'
    self.datasets = datasets
    self.tasks = {}
    for dataset in datasets:
      for task in dataset.tasks:
        if task.name in self.tasks:
          assert task.id == self.tasks[task.name].id, 'Duplicate task name found'
        self.tasks[task.name] = task
    self.tasks = list(self.task.values())

  def __len__(self):
    return sum(len(x) for x in self.datasets)
  
  def __getitem__(self, idx):
    i = 0
    while idx >= len(self.datasets[i]):
      idx -= len(self.datasets[i])
      i += 1
    batch = self.datasets[i][idx]
    for task in self.tasks:
      if task.name not in batch:
        batch[task.name] = task.default
        batch[task.name + '?'] = torch.tensor(0.0)
    return batch


