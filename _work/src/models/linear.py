from collections import OrderedDict

from torch import Tensor, nn

from ..modules.utils import flatten


class Linear(nn.Module):
    def __init__(self, sizes: list[int], F=nn.ReLU, batch_norm=True, dropout: float = 0):
        super().__init__()

        assert len(sizes) >= 3

        # fmt: off
        self.layers = nn.Sequential(OrderedDict([
            *flatten([
                [
                    (f"linear{i + 1}", nn.Linear(sizes[i], sizes[i + 1])),
                    *(
                        [(f"batchnorm{i + 1}", nn.BatchNorm1d(sizes[i + 1]))]
                        if batch_norm == True else []
                    ),
                    (f"{F.__name__.lower()}{i + 1}", F()),
                    *(
                        [(f"dropout{i + 1}", nn.Dropout(p=dropout))]
                        if dropout > 0
                        else []
                    ),
                ]
                for i in range(len(sizes) - 2)
            ]),
            (f"linear{len(sizes) - 1}", nn.Linear(sizes[-2], sizes[-1])),
            ("sigmoid", nn.Sigmoid()),
        ]))
        # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def replay(self, x: Tensor) -> Tensor:
        pass
