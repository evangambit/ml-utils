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
	return A2 + B2 - A @ B.T * 2.0

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

