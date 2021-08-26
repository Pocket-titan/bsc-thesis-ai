from collections import OrderedDict

from torch import Tensor, nn

from ..modules.utils import flatten


class AutoEncoder(nn.Module):
    def __init__(self, sizes: list[int], F=nn.ReLU, batch_norm=True, dropout: float = 0):
        super().__init__()

        assert len(sizes) > 0
        self.sizes = sizes

        # fmt: off
        self.encoder = nn.Sequential(OrderedDict([
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
                for i in range(0, len(sizes) - 1)
            ]),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            *flatten([
                [
                    (f"linear{len(sizes) - (i + 1)}", nn.Linear(sizes[i + 1], sizes[i])),
                    *(
                        [(f"batchnorm{len(sizes) - (i + 1)}", nn.BatchNorm1d(sizes[i]))]
                        if batch_norm == True else []
                    ),
                    (f"{F.__name__.lower()}{len(sizes) - (i + 1)}", F()),
                    *(
                        [(f"dropout{len(sizes) - (i + 1)}", nn.Dropout(p=dropout))]
                        if dropout > 0
                        else []
                    ),
                ]
                for i in reversed(range(1, len(sizes) - 1))
            ]),
            (f"linear{len(sizes) - 1}", nn.Linear(sizes[1], sizes[0])),
            *(
                [(f"batchnorm{len(sizes) - 1}", nn.BatchNorm1d(sizes[0]))]
                if batch_norm == True else []
            ),
            ("sigmoid", nn.Sigmoid()),
        ]))
        # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.sizes[0])
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded

    def replay(self, x_encoded: Tensor) -> Tensor:
        return self.decoder(x_encoded)

    def loss_function(self):
        pass
