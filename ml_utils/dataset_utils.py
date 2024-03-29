from torch.utils import data as tdata
import numpy as np

from .task import ClassificationTask
from .encodable import Encodable

class MNIST(tdata.Dataset, Encodable):
  def __init__(self, train : bool):
    self.train = train
    from torchvision import datasets, transforms
    self.dataset = datasets.MNIST(root='/tmp', train=train, download=True, transform=transforms.ToTensor())
    self.tasks = [ClassificationTask('mnist_labels', '0123456789')]

  def encode(self):
    return {
      "$type": "ml_utils.dataset_utils.mnist",
      "train": self.train
    }
  
  @classmethod
  def decode(klass, state):
    return MNIST(state["train"])
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, i):
    x, y = self.dataset[i]
    return {
      "images": x,
      "mnist_labels": y,
    }

Encodable.add_type("ml_utils.dataset_utils.mnist", MNIST)

class CIFAR10(tdata.Dataset, Encodable):
  def __init__(self, train : bool):
    self.train = train
    from torchvision import datasets, transforms
    self.dataset = datasets.CIFAR10(root='/tmp', train=train, download=True, transform=transforms.ToTensor())
    self.tasks = [ClassificationTask('cifar10_labels', '0123456789')]

  def encode(self):
    return {
      "$type": "ml_utils.dataset_utils.cifar10",
      "train": self.train
    }
  
  @classmethod
  def decode(klass, state):
    return CIFAR10(state["train"])
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, i):
    x, y = self.dataset[i]
    return {
      "images": x,
      "cifar10_labels": y,
    }

Encodable.add_type("ml_utils.dataset_utils.cifar10", CIFAR10)

class OverlappingSampler(tdata.Sampler):
  """
  This sampler reuses a certain amount of each batch. You can use this with
  CachingDataset to significantly speed up the dataloading for each batch, at
  the cost of repeating datapoints.
  """
  def __init__(self, dataset, batch_size, step_size = None) -> None:
    self.indices = np.arange(len(dataset))
    np.random.shuffle(self.indices)
    self.batch_size = batch_size
    self.step = step_size if step_size else batch_size
    self.i = 0
    self.j = 0
  
  def __len__(self):
    return self.indices.shape[0]  # todo: be more precise

  def __iter__(self):
    while self.i + self.j < len(self.indices):
      yield self.indices[self.i + self.j]
      self.j += 1
      if self.j >= self.batch_size:
        self.j = 0
        self.i += self.step

from functools import lru_cache
class CachingDataset(tdata.Dataset):
  """
  This sampler caches items from a dataset. Note that it does *not* reapply
  transforms, in fitting with our philosophy that transforms should live apart
  from datasets.
  """
  def __init__(self, dataset : tdata.Dataset, cache_size : int = 512):
    super().__init__()
    self._dataset = dataset
    self._getitem = lru_cache(maxsize = cache_size)(lambda idx: self._dataset[idx])

  def __getattr__(self, attr):
    return getattr(self._dataset, attr)

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    return self._getitem(idx)

class Subdataset(tdata.Dataset):
  def __init__(self, dataset, indices):
    super().__init__()
    self._dataset = dataset
    self._indices = indices
  def __getattr__(self, attr):
    return getattr(self._dataset, attr)
  def __len__(self):
    return self._indices.shape[0]
  def __getitem__(self, idx):
    return self._dataset[self._indices[idx]]

def interweaver(*A):
  """
  Merges multiple data loaders and loops over them endlessly.

  Example:

  # Run 1 test batch every 5 training batches.
  for split, (x, y) in interweaver(
    ("train", 5, trainloader),
    ("test", 1, testloader),
    ):
  """
  n = len(A)
  names = [a[0] for a in A]
  steps = [a[1] for a in A]
  loaders = [a[2] for a in A]
  iters = [iter(a) for a in loaders]
  i = 0
  while True:
    name = names[i % n]
    for _ in range(steps[i % n]):
      try:
        yield name, loaders[i % n].dataset, next(iters[i % n])
      except StopIteration:
        iters[i % n] = iter(loaders[i % n])
        yield name, loaders[i % n].dataset, next(iters[i % n])
    i += 1
