from collections import OrderedDict

from torch import Tensor, nn

from ..modules.utils import flatten


class VariationalAutoEncoder(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()

    def forward(self, x: Tensor):
        pass


def loss_function():
    pass
