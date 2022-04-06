import torch
import numpy as np
import time

def lpad(t, n, c=' '):
  t = str(t)
  return max(0, n - len(t)) * c + t

def count(A):
  C = {}
  for a in A:
    C[a] = C.get(a, 0) + 1
  return C

def dist2(A, B):
  A2 = (A**2).sum(1).reshape((A.shape[0], 1))
  B2 = (B**2).sum(1).reshape((1, B.shape[0]))
  r = A2 + B2 - A @ B.T * 2.0
  if isinstance(r, np.ndarray):
    return np.maximum(0.0, r)
  return torch.relu(r)

def print_table(A, precision=5):
  B = []
  column_widths = [0] * len(A[0])
  for row in A:
    r = []
    for x, cell in enumerate(row):
      if type(cell) is float:
        r.append(f'%.{precision}f' % cell)
      else:
        r.append(str(cell))
      column_widths[x] = max(column_widths[x], len(r[-1]))
    B.append(r)
  
  for row in B:
    t = ''
    for x, cell in enumerate(row):
      t += lpad(cell, column_widths[x]) + '   '
    print(t)

class MetricDict(dict):
  def __init__(self):
    super(MetricDict, self).__init__()
  def __getitem__(self, key):
    if key not in self:
      self[key] = []
    return super().__getitem__(key)

class Timer:
  def __init__(self):
    self._start = {}
    self.totals = MetricDict()
  def start(self, k):
    assert k not in self._start, f'key "{k}" already exists'
    self._start[k] = time.time()
  def end(self, k):
    assert k in self._start, f'key "{k}" does not exist'
    if k not in self.totals:
      self.totals[k] = 0.0
    self.totals[k] += time.time() - self._start[k]
    del self._start[k]

class PiecewiseFunction:
  def __init__(self, x, y):
    self.x = np.array(x, dtype=np.float64)
    self.y = np.array(y, dtype=np.float64)
  
  def __call__(self, x):
    if x <= self.x[0]:
      return self.y[0]
    if x >= self.x[-1]:
      return self.y[-1]
    high = np.searchsorted(self.x, x)
    low = high - 1
    t = (x - self.x[low]) / (self.x[high] - self.x[low])
    return self.y[low] * (1 - t) + self.y[high] * t

def num_params(model):
  numParams = 0
  for p in model.parameters():
    r = 1
    for s in p.shape:
      r *= s
    numParams += int(r)
  return numParams
