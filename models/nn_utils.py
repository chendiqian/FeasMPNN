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
        return torch.log(1. / (x + self.bias))


class SQRTEncoder:
    def __init__(self):
        pass

    def sqrt(self, x):
        sign = torch.sign(x)
        x = sign * torch.sqrt(torch.abs(x))
        return x

    def sqr(self, x):
        sign = torch.sign(x)
        x = sign * (x ** 2.)
        return x
