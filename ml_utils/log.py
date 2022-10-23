import numpy as np
import json
import os
from collections import defaultdict

class RamLogger:
  def __init__(self, metricMinSize = 0.0):
    self.metrics = {}
    self.metricMinSize = metricMinSize

  def metric(self, name):
    A = [a for a in self.metrics[name] if a[2] >= self.metricMinSize]
    x = np.array([a[0] for a in A], dtype=np.float64)
    y = np.array([a[1] for a in A], dtype=np.float64)
    return x, y

  def log(self, metricName : str, x : float, y : float, n : float = 1.0):
    if metricName not in self.metrics:
      self.metrics[metricName] = [[0.0, 0.0, 0.0]]

    A = self.metrics[metricName]
    a = A[-1]
    if a[2] >= self.metricMinSize:
      A[-1] = (a[0] / a[2], a[1] / a[2], a[2])
      A.append([0, 0, 0])
      a = A[-1]
    a[0] += x * n
    a[1] += y * n
    a[2] += n


