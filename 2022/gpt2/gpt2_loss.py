import torch
from transformers import GPT2Model, GPT2Tokenizer


class GPT2Loss(torch.nn.Module):
    def __init__(self, model, input_dim: int, num_classes: int):
        super().__init__()
        # input_dim: 768 or 1280
        self.gpt2model = model
        self.fc1 = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x, y):
        gpt_out = self.gpt2model(x)
        linear_output = self.fc1(gpt_out.last_hidden_state)
        out = linear_output.sum(axis=2)
        return torch.abs(out - y).sum()
