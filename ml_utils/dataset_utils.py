from torch.utils import data as tdata
import numpy as np

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

