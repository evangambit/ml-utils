import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.utils.data as tdata
from torchvision import models

import ml_utils

logger = ml_utils.log.LoggerFamily('logs', sampleSizeToAggregate=3, logsPerWrite=2)

class Dataset(tdata.Dataset):
  def __init__(self):
    n = 200
    self.X = np.random.normal(0, 1, (n, 3, 224, 224)).astype(np.float32)
    self.colors = np.random.randint(0, 3, (n,))
    self.heights = np.random.uniform(0, 1, (n,)).astype(np.float32)
    self.tags = np.random.randint(0, 2, (n, 3)).astype(np.float32)
    self.tasks = [
      ml_utils.task.ClassificationTask('color', ['red', 'green', 'blue']),
      ml_utils.task.RegressionTask('height', 2.0, 0.5),
      ml_utils.task.DetectionTask('tags', ['good', 'cool', 'new'])
    ]

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return {
      "image": torch.tensor(self.X[idx]),

      "color": torch.tensor(self.colors[idx]),
      "color?": torch.tensor(1),

      "height": torch.tensor(self.heights[idx]),
      "height?": torch.tensor(1 if idx % 3 else 0, dtype=torch.int64),

      "tags": torch.tensor(self.tags[idx]) if idx % 4 else self.tasks[2].missing.clone(),
      "tags?": torch.tensor([1.0 if idx % 4 else 0.0] * 3),
    }

# Create fake dataset
dataset = Dataset()

# Split into train/test split
I = np.arange(len(dataset))
np.random.shuffle(I)
trainset = ml_utils.Subdataset(dataset, I[10:])
testset = ml_utils.Subdataset(dataset, I[:10])

# Load ResNext50
model = ml_utils.Resnet(block = models.resnet.Bottleneck, layers = [3, 4, 6, 3], groups=32, width_per_group=4)
model.load_state_dict(models.resnet.resnext50_32x4d(weights=models.resnet.ResNeXt50_32X4D_Weights.DEFAULT).state_dict())

# Load ResNet18
# model = ml_utils.Resnet(block = models.resnet.BasicBlock, layers = [2, 2, 2, 2])
# model.load_state_dict(models.resnet.resnet18().state_dict())

model = ml_utils.Resnet(block = models.resnet.BasicBlock, layers = [1, 1, 1, 1])

model.init(dataset.tasks)

kUseCuda = torch.cuda.device_count() > 0
if kUseCuda:
  model = model.cuda()

opt = optim.AdamW(model.parameters(), lr=3e-4)

it = 0
for split, batch in ml_utils.interweaver(
  ("train", 2, tdata.DataLoader(trainset, batch_size=2)),
  ("test", 1, tdata.DataLoader(testset, batch_size=2)),
  ):
  it += 1

  if split == 'train':
    model.train()
  else:
    model.eval()

  if kUseCuda:
    for k in batch:
      if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].cuda()

  yhat = model(batch['image'])

  if split == 'train':
    loss = 0.0
    for task in dataset.tasks:
      loss = loss + task.loss(yhat, batch)
      task.log_metrics(yhat, batch, it, split, logger)
    loss.backward()
    opt.step()

  if it >= 50:
    break

logger.flush()
