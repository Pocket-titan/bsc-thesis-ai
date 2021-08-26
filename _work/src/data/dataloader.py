from typing import Literal, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, WeightedRandomSampler

from ..modules.replay import generate_samples
from ..modules.utils import int_to_onehot


class HierarchicalDataset(Dataset):
    """
    Dataset class for hierarchical multi-class data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()

        indices = list([i for i, _ in enumerate(df.index)])

        self.targets = int_to_onehot(indices, dtype=torch.float32)
        self.features = torch.tensor(df.values, dtype=torch.float32)
        self.classes = list(df.index)

        self.df = df
        self.NUM_ITEMS = self.targets.shape[1]
        self.NUM_ATTRIBUTES = self.features.shape[1]

        self.class_map = [
            {
                "index": x[0],
                "target": x[1],
                "feature": x[2],
                "class": x[3],
            }
            for x in zip(
                indices,
                self.targets,
                self.features,
                self.classes,
            )
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        x = self.targets[index]
        y = self.features[index]

        return {
            "x": x,
            "y": y,
        }


def create_dataloader(
    data: Union[Tensor, pd.DataFrame, Dataset, HierarchicalDataset],
    weights: list[float] = None,
    shuffle=True,
    batch_size=4,
) -> DataLoader:
    """
    Create a pytorch dataloader for hierarchical data.
    """
    if isinstance(data, (HierarchicalDataset, ReplayDataset, Dataset)):
        dataset = data
    elif isinstance(data, pd.DataFrame):
        dataset = HierarchicalDataset(
            pd.DataFrame(
                data,
                index=data.index,
                columns=data.columns,
            )
        )
    else:
        dataset = HierarchicalDataset(
            pd.DataFrame(
                data,
                index=[f"item_{i}" for i in range(data.shape[0])],
                columns=[f"attribute_{i}" for i in range(data.shape[1])],
            )
        )

    if weights != None:
        assert len(weights) == len(dataset.df.index)

        sampler = BatchSampler(
            WeightedRandomSampler(
                weights,
                num_samples=batch_size,
                replacement=True,
            ),
            batch_size=batch_size,
            drop_last=False,
        )
        batch_size = None
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        sampler=sampler,
    )


class ReplayDataset(Dataset):
    def __init__(self, data: pd.Series, features, targets) -> None:
        super().__init__()

        df = pd.DataFrame(np.stack(data.values).reshape(len(data), -1), index=data.index)

        self.df = df
        self.activations = torch.tensor(df.values, dtype=torch.float32)
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "x": self.targets[index],
            "y": self.features[index],
            "activations": self.activations[index],
        }


def create_replay_dataloader(
    n_samples: int,
    dataloader: DataLoader,
    outputs: pd.DataFrame,
    layer: str,
    strategy: Literal[
        "uniform",
        "conditional",
        "multimodal",
        "multivariate",
    ],
):
    samples = generate_samples(n_samples, outputs, layer, strategy)

    legend = [next(x for x in dataloader.dataset.class_map if x["class"] == item) for item in samples.index]
    features = list(map(lambda x: x["feature"], legend))
    # features = samples.values
    targets = list(map(lambda x: x["target"], legend))

    return create_dataloader(
        ReplayDataset(samples, features, targets),
        batch_size=dataloader.batch_size,
    )
