import torch
from onnxruntime.training.ortmodule import ORTModule


class NNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)

    def forward(self, x, y):
        linear_output = self.fc1(x)
        out = linear_output.sum(axis=1)
        print(out.shape, y.shape)
        return torch.abs(out - y).sum()


model = NNLoss()
x = torch.rand(3, 3)
y = torch.rand(3)
print(model(x, y))

o = ORTModule(model)
print(o(x, y))
