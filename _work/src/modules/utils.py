import os
import pickle
from operator import attrgetter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist, squareform
from torch import Tensor, nn

#
# Python utils
#


def flatten(_list: list) -> list:
    """
    Flatten a list of lists (of lists (of lists...)).
    Does *not* flatten tuples!
    """
    for el in _list:
        if isinstance(el, list):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def chunk(_list: list, n: int) -> list:
    """
    Split a list into chunks of size `n`, or smaller if the size doesn't match up at the tail.
    """
    return [_list[i : i + n] for i in range(0, len(_list), n)]


#
# PyTorch utils
#


def int_to_onehot(ints: list[int], dtype=torch.int32) -> Tensor:
    """
    Convert a list (or tensor) of integers to one-hot vectors.
    """
    if not isinstance(ints, Tensor):
        ints = torch.tensor(ints)

    return F.one_hot(ints).type(dtype)


def onehot_to_int(one_hot: Tensor) -> Tensor:
    """
    Convert a Tensor of one-hot encoded vectors to their integer representations.
    """
    if len(one_hot.shape) != 2:
        raise Exception()

    return torch.argmax(one_hot, dim=1)


def reset_model(model: nn.Module) -> None:
    """
    Reset all of a model's modules that have parameters.
    """
    for module in list(model.modules()):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _get_layer_size(model: nn.Module, which: str("input") or str("output")) -> int:
    """
    Get the size of either the input or output layer of a model.
    """
    params = list(model.parameters())
    first, last = params[0], params[-1]
    input_dim = first.shape[0] if len(first.shape) == 1 else first.shape[1]
    output_dim = last.shape[0] if len(last.shape) == 1 else first.shape[0]
    return input_dim if which == "input" else output_dim


def get_input_size(model: nn.Module) -> int:
    return _get_layer_size(model, which="input")


def get_output_size(model: nn.Module) -> int:
    return _get_layer_size(model, which="output")


def add_noise(
    tensor: Tensor,
    mean: float = 0.0,
    stdev: float = 1.0,
    multiplier: float = 1,
    mode="gaussian",
) -> Tensor:
    """
    Add some noise to a tensor. `mode` = "gaussian" | "speckle".
    """

    noise = np.random.normal(mean, stdev, tensor.shape)

    if mode == "gaussian":
        return (tensor + multiplier * noise).to(dtype=torch.float32)

    if mode == "speckle":
        return (tensor + multiplier * tensor * noise).to(dtype=torch.float32)


def register_save_hooks(model: nn.Module, inputs: dict, outputs: dict):
    """
    Register hooks that save the models' input and output each epoch.
    """

    layers = dict(
        filter(
            lambda x: isinstance(x[1], (nn.Linear, nn.ReLU, nn.Sigmoid, nn.Dropout)),
            list(model.named_modules()),
        )
    ).keys()

    def save_hook(name: str):
        def hook(model, input, output):
            if not name in inputs:
                inputs[name] = []
            if not name in outputs:
                outputs[name] = []

            inputs[name].append(input[0].detach().numpy())
            outputs[name].append(output.detach().numpy())

        return hook

    handles = [attrgetter(layer)(model).register_forward_hook(save_hook(layer)) for layer in layers]

    def unregister_hooks():
        for handle in handles:
            handle.remove()

    return unregister_hooks


def toggle_learning(module: nn.Module, on_or_off: bool):
    """
    Toggle the learning in a particular module.
    """
    for param in module.parameters():
        param.requires_grad = on_or_off


def get_distances(tensor: Tensor, df: pd.DataFrame, metric: str = "cosine"):
    """
    Get all distances of a tensor with respect to the dataframe, in ascending order.
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)

    dists = cdist(df.values, tensor.view((1, -1)), metric)
    items = [*map(lambda x: (df.index[x[0]], x[1].item()), sorted(enumerate(dists), key=lambda x: x[1]))]
    return items


def get_closest_item(df: pd.DataFrame, tensor: Tensor, metric: str = "cosine", fix=True) -> tuple[str, float]:
    """
    Returns (closest_item_label, closest_item_distance)
    """
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)

    dists = cdist(df.values, tensor.view((1, -1)), metric)
    mins = np.where(dists == dists.min())[0]

    if len(mins) == 0:
        mins = np.random.choice(range(len(df.index)), 1)

    idx = mins[0] if not fix else np.random.choice(mins, 1).item()
    return df.index[idx], dists.min()


def round_tensor(tensor, decimals: int = None):
    """
    Round a tensor to `decimals` decimals.
    """

    if decimals is None:
        decimals = 0

    return torch.round(tensor * 10 ** decimals) / (10 ** decimals)


def save_metrics(metrics, filename: str):
    try:
        os.remove(f"../runs/{filename}.pickle")
    except OSError:
        pass

    if "/" in filename:
        Path(f"../runs/{'/'.join(filename.split('/')[:-1])}").mkdir(parents=True, exist_ok=True)

    with open(f"../runs/{filename}.pickle", "wb") as f:
        pickle.dump(metrics, f)


def load_metrics(dirname: str):
    filenames = os.listdir(f"../runs/{dirname}")
    filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))

    metrics = []
    for filename in filenames:
        with open(f"../runs/{dirname}/{filename}", "rb") as f:
            content = pickle.load(f)
            metrics.append(content[0])

    return metrics
