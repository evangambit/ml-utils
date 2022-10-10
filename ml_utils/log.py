import numpy as np
import json
import os
from collections import defaultdict

class Logger:
  def __init__(self, path, sampleSizeToAggregate : float = 0.0 , logsPerWrite : int = 1):
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(path, exist_ok=True)
    self._logsPerWrite = logsPerWrite
    self._sampleSizeToAggregate = sampleSizeToAggregate
    self.logsSinceLastWrite = 0
    self.carry = defaultdict(lambda: np.zeros(3))
    self.logs = defaultdict(list)

  def log(self, metricName : str, x : float, y : float, n : float = 1.0):
    self.carry[metricName] += np.array([x * n, y * n, n])
    if self.carry[metricName][2] >= self._sampleSizeToAggregate:
      self.logs[metricName].append(self.carry[metricName])
      self.carry[metricName] = np.zeros(3)
      if len(self.logs[metricName]) >= self._logsPerWrite:
        self._flush_metric(metricName)

  def flush(self):
    for metricName in self.logs:
      self._flush_metric(metricName)

  def _flush_metric(self, metricName):
    with open(os.path.join(self.path, metricName), 'a+') as f:
      for a in self.logs[metricName]:
        t = json.dumps([
          float(a[0] / a[2]),
          float(a[1] / a[2]),
          float(a[2]),
        ])
        f.write(t + "\n")
    self.logs[metricName] = []


class LoggerFamily:
  """
  sampleSizeToAggregate: how large a sample should get before we log it (default: 0)
  logsPerWrite: how many logs should occur before we write to file (default: 1)
  """
  def __init__(self, path, sampleSizeToAggregate : float = 0.0, logsPerWrite : int = 1):
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(path, exist_ok=True)
    self._logsPerWrite = logsPerWrite
    self.logsSinceLastWrite = 0
    self.loggers = {}
    self._kwargs = {
      "logsPerWrite": logsPerWrite,
      "sampleSizeToAggregate": sampleSizeToAggregate,
    }

  def log(self, run, metricName, x, y, n : float = 1.0):
    if run not in self.loggers:
      self.loggers[run] = Logger(os.path.join(self.path, run), **self._kwargs)
    self.loggers[run].log(metricName, x, y, n)
