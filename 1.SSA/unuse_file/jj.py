import torch
from torch import nn

out = torch.rand(100, 2)
Y = torch.rand(100, 2)
m = Y.argmax(dim=1)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, Y.argmax(dim=1))
print(loss.shape)