from torch.utils import data as tdata
import numpy as np

class DatasetWrapper(tdata.Dataset):
  def __init__(self, dataset, args):
    super().__init__()
    self.dataset = dataset
    self.args = args
  def __getattr__(self, attr):
    return getattr(self.dataset, attr)
  def __getitem__(self, idx):
    A = self.dataset[self.indices[idx]]
    assert len(A) == len(self.args)
    R = {}
    for k, v in zip(self.args, A):
      R[k] = v
    return R

class OverlappingSampler(tdata.Sampler):
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
  def __init__(self, dataset):
    super().__init__()
    self.dataset = dataset

  def __getattr__(self, attr):
    return getattr(self.dataset, attr)

  def __len__(self):
    return len(self.dataset)

  @lru_cache(maxsize = 512)
  def __getitem__(self, idx):
    return self.dataset[idx]

class Subdataset(tdata.Dataset):
  def __init__(self, dataset, indices):
    super().__init__()
    self.dataset = dataset
    self.indices = indices
  def __getattr__(self, attr):
    return getattr(self.dataset, attr)
  def __len__(self):
    return self.indices.shape[0]
  def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]

def interweaver(*A):
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
        yield name, next(iters[i % n])
      except StopIteration:
        iters[i % n] = iter(loaders[i % n])
        yield name, next(iters[i % n])
    i += 1