from copy import deepcopy
from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# from .utils import get_input_size, get_output_size
from _work.src.modules.utils import get_input_size, get_output_size
from torch import Tensor, nn
from torch.utils.data import DataLoader


class Metrics:
    def __init__(
        self,
        losses: dict[str, list[float]],
        predictions: pd.DataFrame,
        parameters: OrderedDict,
        inputs: pd.DataFrame,
        outputs: pd.DataFrame,
    ) -> None:
        self.losses = losses
        self.predictions = predictions
        self.parameters = parameters
        self.inputs = inputs
        self.outputs = outputs

    def join(self, other_metrics):
        other_metrics = deepcopy(other_metrics)
        _self = deepcopy(self)

        predictions = other_metrics.predictions
        predictions.loc[predictions["epoch"] == 0, "loss"] = [np.nan] * len(predictions.index.unique())
        predictions.loc[predictions["epoch"] == 0, "accuracy"] = [np.nan] * len(predictions.index.unique())
        other_metrics.losses["train"][0] = np.nan
        other_metrics.losses["test"][0] = np.nan

        for k, v in other_metrics.losses.items():
            if k in _self.losses:
                _self.losses[k].extend(v)
            else:
                _self.losses[k] = v

        last_epoch = _self.predictions["epoch"].max()
        predictions["epoch"] += last_epoch + 1
        _self.predictions = pd.concat([_self.predictions, predictions])

        for k, v in other_metrics.parameters.items():
            if k in _self.parameters:
                _self.parameters[k].extend(v)
            else:
                _self.parameters[k] = v

        last_epoch = _self.inputs["epoch"].max()
        other_metrics.inputs["epoch"] += last_epoch + 1
        other_metrics.outputs["epoch"] += last_epoch + 1
        _self.inputs = pd.concat([_self.inputs, other_metrics.inputs])
        _self.outputs = pd.concat([_self.outputs, other_metrics.outputs])

        return _self


@torch.no_grad()
def test_model(
    model: nn.Module,
    loss_fn: nn.MSELoss,
    dataloader: DataLoader,
) -> pd.DataFrame:
    """
    Test the performance of a model.
    """
    model.eval()
    input_size, output_size = get_input_size(model), get_output_size(model)
    accuracies, losses, predictions = [], [], []

    for data in dataloader.dataset:
        # The dataset yields 1D tensors, unsqueeze to make a batch with batch_size=1
        x = data["y" if input_size == data["y"].shape[0] else "x"].reshape(1, -1)
        y = data["y" if output_size == data["y"].shape[0] else "x"].reshape(1, -1)
        y_pred = model(x)

        if type(y_pred) == tuple and len(y_pred) > 1:
            y_pred, *args = y_pred
        else:
            args = []

        loss = loss_fn(y_pred, y, *args)
        # accuracy = nn.CosineSimilarity(dim=1, eps=1e-6)(y_pred, y)
        # accuracy = nn.PairwiseDistance(p=2, eps=1e-6)(y_pred, y)
        accuracy = (y_pred.round() == y).to(dtype=torch.float32).mean()

        # Squeeze here b/c we unsqueezed at the top
        predictions.append(y_pred.reshape(-1).numpy())
        accuracies.append(accuracy.item())
        losses.append(loss.item())

    df = dataloader.dataset.df
    predictions = np.array(predictions)
    index = df.index if predictions.shape[0] == df.index.shape[0] else df.columns
    columns = df.columns if predictions.shape[1] == df.columns.shape[0] else df.index

    predictions = pd.DataFrame(data=predictions, index=index, columns=columns)
    predictions["accuracy"] = accuracies
    predictions["loss"] = losses

    return predictions


@torch.no_grad()
def eval_model(model, dataloader):
    from .plot import plot_tensor

    df = dataloader.dataset.df
    model.eval()

    outs = []
    for b in dataloader.dataset:
        out = model(b["y"])
        outs.append(out.squeeze().tolist())

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_tensor(pd.DataFrame(outs, index=df.index, columns=df.columns), ax=axes[0])
    axes[0].set_title("Predictions after training")
    plot_tensor(df, ax=axes[1])
    axes[1].set_title("Dataset")

