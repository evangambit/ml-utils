import torch
from torch import nn
import torch.utils.data as tdata
import numpy as np

import math

from .encodable import Encodable

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


class Task(Encodable):
  def __init__(self, name : str):
    self.name = name
    self.default = None
    self.missing = None
    self.taskId = Task.taskId
    Task.taskId += 1

  def loss(self, predictions, batch):
    """
    Returns a 0-D Pytorch tensor.
    """
    pass

  def metrics(self, predictions : dict, batch : dict, it : int, run : str):
    raise NotImplementedError('')
    return {}

  def log_metrics(self, predictions : dict, batch : dict, it : int, run : str, logger):
    metrics = self.metrics(predictions, batch, it, run)
    for k in metrics:
      value, n = metrics[k]
      logger.log(run, k, it, value, n)


Task.taskId = 0


class ClassificationTask(Task):
  def __init__(self, name : str, classes : list):
    Task.__init__(self, name)
    self.classes = classes
    self._loss = nn.CrossEntropyLoss(reduction='none')
    self.default = torch.tensor(0, dtype=torch.int64)
    self.missing = torch.tensor(0, dtype=torch.int64)

  def create_heads(self, din : int):
    head = nn.Linear(din, len(self.classes))
    with torch.no_grad():
      b = logit(1.0 / len(self.classes))
      head.bias.zero_()
      head.bias += b
      head.weight.zero_()
    return {
      self.name: head
    }

  def loss(self, predictions : dict, batch : dict):
    yhat = predictions[self.name]
    y = batch[self.name]
    if self.name + '?' in batch:
      mask = batch[self.name + '?']
      mask_sum = float(mask.sum())
    else:
      mask = 1.0
      mask_sum = y.shape[0]

    l = self._loss(yhat, y)

    if mask_sum == 0.0:
      return 0.0
    return (l * mask).sum() / mask_sum

  def metrics(self, predictions : dict, batch : dict, it : int, run : str):
    yhat = predictions[self.name]
    y = batch[self.name]

    if self.name + '?' in batch:
      mask = batch[self.name + '?']
      mask_sum = int(mask.sum())
    else:
      mask = 1.0
      mask_sum = y.shape[0]

    if mask_sum == 0:
      return {}

    l = self._loss(yhat, y)
    incorrect = (yhat.argmax(1) != y).to(torch.float32)

    return {
      f"{self.name}:loss": (float((l * mask).sum()) / mask_sum, mask_sum),
      f"{self.name}:error": (float((incorrect * mask).sum()) / mask_sum, mask_sum),
    }

  def encode(self):
    return {
      "$type": "ml_utils.task.ClassificationTask",
      "kwargs": {
        "name": self.name,
        "classes": self.classes,
      },
    }
  
  @classmethod
  def decode(klass, state):
    return ClassificationTask(**state["kwargs"])

Encodable.add_type("ml_utils.task.ClassificationTask", ClassificationTask)


class RegressionTask(Task):
  """
  Represents a 1D regression task.

  Neural network outputs are expected to have a mean of 0 and
  a standard deviation of 1. This task scales the labels to match
  this. The reasoning is that if we don't normalize the labels,
  then the gradients between different tasks could have very different
  magnitudes.
  """
  def __init__(self, name : str, avg : float = 0.0, std : float = 1.0):
    Task.__init__(self, name)
    self.avg = avg
    self.std = std
    self.default = torch.tensor(0.0, dtype=torch.float32)
    self.missing = torch.tensor(0.0, dtype=torch.float32)
    self._loss = nn.MSELoss(reduction='none')

  def create_heads(self, din : int):
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

  def loss(self, predictions : dict, batch : dict):
    yhat = predictions[self.name]
    y = (batch[self.name] - self.avg) / self.std
    mask = batch[self.name + '?']
    mask_sum = np.float32(mask.sum())

    l = self._loss(yhat, y)

    if mask_sum == 0.0:
      return 0.0
    return (l * mask).sum() / mask_sum

  def metrics(self, predictions : dict, batch : dict, it : int, run : str):
    yhat = predictions[self.name]
    y = (batch[self.name] - self.avg) / self.std
    mask = batch[self.name + '?']
    mask_sum = int(mask.sum())
    if mask_sum == 0:
      return {}

    l = self._loss(yhat, y)
    incorrect = (torch.abs(yhat - y) > 1.0).to(torch.float32)

    return {
      f"{self.name}:loss": (float((l * mask).sum()) / mask_sum, mask_sum),
      f"{self.name}:error": (float((incorrect * mask).sum()) / mask_sum, mask_sum),
    }

  def encode(self):
    return {
      "$type": "ml_utils.task.RegressionTask",
      "kwargs": {
        "name": self.name,
        "avg": self.avg,
        "std": self.std,
      },
    }
  
  @classmethod
  def decode(klass, state):
    return RegressionTask(**state["kwargs"])

Encodable.add_type("ml_utils.task.RegressionTask", RegressionTask)

class DetectionTask(Task):
  def __init__(self, name : str, classes : list):
    Task.__init__(self, name)
    assert isinstance(classes, list)
    assert isinstance(classes[0], str)
    self.classes = classes
    self.default = torch.tensor(0.5, dtype=torch.float32)
    self.missing = torch.tensor([0.5] * len(classes), dtype=torch.float32)
    self._logSigmoid = nn.LogSigmoid()

  def create_heads(self, din : int):
    head = nn.Linear(din, len(self.classes))
    with torch.no_grad():
      head.bias.zero_()
      head.weight.zero_()
    return {
      self.name: head
    }

  def loss(self, predictions : dict, batch : dict):
    yhat = predictions[self.name]
    y = batch[self.name]
    assert yhat.shape == y.shape
    mask = batch[self.name + '?']
    mask_sum = float(mask.sum())

    l = y * self._logSigmoid(predictions[self.name]) + (1 - y) * self._logSigmoid(-predictions[self.name])

    if mask_sum == 0.0:
      return 0.0
    return (l * mask).sum() / mask_sum

  def metrics(self, predictions : dict, batch : dict, it : int, run : str):
    yhat = predictions[self.name]
    y = batch[self.name]
    mask = batch[self.name + '?']
    mask_sum = int(mask.sum())
    if mask_sum == 0:
      return {}

    l = y * self._logSigmoid(predictions[self.name]) + (1 - y) * self._logSigmoid(-predictions[self.name])
    incorrect = ((yhat > 0) != (y > 0)).to(torch.float32)

    return {
      f"{self.name}:loss": (float((l * mask).sum()) / mask_sum, mask_sum),
      f"{self.name}:error": (float((incorrect * mask).sum()) / mask_sum, mask_sum),
    }

  def encode(self):
    return {
      "$type": "ml_utils.task.DetectionTask",
      "kwargs": {
        "name": self.name,
        "classes": self.classes,
      },
    }
  
  @classmethod
  def decode(klass, state):
    return DetectionTask(**state["kwargs"])

Encodable.add_type("ml_utils.task.DetectionTask", DetectionTask)


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
  
  def __getitem__(self, idx : int):
    i = 0
    while idx >= len(self.datasets[i]):
      idx -= len(self.datasets[i])
      i += 1
    batch = self.datasets[i][idx]
    for task in self.tasks:
      if task.name not in batch:
        batch[task.name] = task.default
        batch[task.name + '?'] = task.missing
    return batch
