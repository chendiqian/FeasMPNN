import torch


class CosineEncoder(torch.nn.Module):
    def __init__(self, bins, bias):
        super().__init__()
        self.bins = bins
        self.bias = bias

    def forward(self, x):
        if x.dim() == 1:
            x = x[:, None]
        steps = torch.arange(1, self.bins + 1, device=x.device).float()
        return torch.cos(steps / (x + self.bias))


class LogEncoder(torch.nn.Module):
    def __init__(self, bias=1.e-8):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return torch.log(x + self.bias)


class PowerEncoder:
    def __init__(self, pow: float):
        self.pow = pow

    def forward(self, x):
        sign = torch.sign(x)
        x = sign * torch.abs(x) ** self.pow
        return x

    def backward(self, x):
        sign = torch.sign(x)
        x = sign * torch.abs(x) ** (1. / self.pow)
        return x
