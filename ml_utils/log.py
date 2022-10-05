import json
import os
from collections import defaultdict

class Logger:
  def __init__(self, path, logsPerWrite = 20):
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(path, exist_ok=True)
    self._logsPerWrite = logsPerWrite
    self.logsSinceLastWrite = 0
    self.A = {}

  def log(self, name, x, y):
    if name not in self.A:
      self.A[name] = []
    self.A[name].append((x, y))
    self.logsSinceLastWrite += 1
    if self.logsSinceLastWrite >= self._logsPerWrite:
      self._write()

  def _write(self):
    for k in self.A:
      with open(os.path.join(self.path, k), 'a+') as f:
        for a in self.A[k]:
          f.write(json.dumps(a) + "\n")
      self.A[k] = []
    self.logsSinceLastWrite = 0

class LoggerFamily:
  def __init__(self, path, logsPerWrite = 20):
    self.path = path
    if not os.path.exists(self.path):
      os.makedirs(path, exist_ok=True)
    self._logsPerWrite = logsPerWrite
    self.logsSinceLastWrite = 0
    self.loggers = {}

  def log(self, run, name, x, y):
    if run not in self.loggers:
      self.loggers[run] = Logger(os.path.join(self.path, run), logsPerWrite=self._logsPerWrite)
    self.loggers[run].log(name, x, y)
