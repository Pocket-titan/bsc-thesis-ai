import sys
from collections import OrderedDict
from copy import deepcopy
from operator import xor
from typing import Literal

import numpy as np
import pandas as pd
import torch
from _work.src.models.autoencoder import AutoEncoder
from _work.src.modules.eval import Metrics, test_model
from _work.src.modules.utils import (
    add_noise,
    get_input_size,
    get_output_size,
    register_save_hooks,
    reset_model,
    toggle_learning,
)

# from _work.src.models.autoencoder import AutoEncoder
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# from ..models.autoencoder import AutoEncoder
# from ..modules.eval import Metrics, test_model
# from ..modules.utils import (
#     add_noise,
#     get_input_size,
#     get_output_size,
#     register_save_hooks,
#     reset_model,
#     toggle_learning,
# )


def train_epoch(
    model: nn.Module,
    optimizer: AdamW,
    loss_fn: nn.MSELoss,
    dataloader: DataLoader,
    l1_lambda: float = 0,
    noise_mean=None,
    noise_stdev=None,
    noise_multiplier=None,
    mode: Literal["train", "replay"] = "train",
) -> float:
    """
    Train a single epoch.
    """
    model.train()
    input_size, output_size = get_input_size(model), get_output_size(model)
    train_loss = 0

    # Technically these are minibatches
    for batch in dataloader:
        x = batch["activations" if mode == "replay" else "y" if input_size == batch["y"].shape[1] else "x"]
        y = batch["y" if output_size == batch["y"].shape[1] else "x"]

        if noise_mean != None and noise_stdev != None:
            x = add_noise(
                x, mean=noise_mean, stdev=noise_stdev, multiplier=noise_multiplier
            )  # TODO: multiplier 1 or 0.1??

        y_pred = model.replay(x) if mode == "replay" else model(x)

        if type(y_pred) == tuple and len(y_pred) > 1:
            y_pred, *args = y_pred
        else:
            args = []

        loss = loss_fn(y_pred, y, *args)

        if l1_lambda > 0:
            l1_regularization = 0
            for param in model.parameters():
                l1_regularization += param.abs().sum()

            loss += l1_lambda * l1_regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss


def train_epochs(
    model: nn.Module,
    optimizer: AdamW,
    loss_fn: nn.MSELoss,
    train_loader: DataLoader,
    test_loader: DataLoader = None,
    epochs: int = 2500,
    l1_lambda: float = 0,
    noise_mean=None,
    noise_stdev=None,
    noise_multiplier=None,
    reset=True,
    mode="train",
    callback=None,
) -> Metrics:
    """
    Train lots of epochs.
    """
    if test_loader == None:
        assert isinstance(train_loader, DataLoader)
        test_loader = train_loader

    if reset == True:
        reset_model(model.decoder)

    if xor(bool(noise_mean), bool(noise_stdev)):
        if bool(noise_mean):
            noise_stdev = 1.0
        elif bool(noise_stdev):
            noise_mean = 0.0

    train_losses = []
    predictions = pd.DataFrame()
    parameters, inputs, outputs = OrderedDict(), OrderedDict(), OrderedDict()

    for epoch in (
        progress_bar := tqdm(
            range(epochs),
            file=sys.stdout,
            colour="#59B1A6",
            unit="epochs",
            desc="training model",
            leave=False,
        )
    ):
        # Save the parameters
        for key, value in deepcopy(model.state_dict()).items():
            if not key in parameters:
                parameters[key] = []
            parameters[key].append(value)

        # Register the hooks which save the activations
        unregister_hooks = register_save_hooks(model, inputs, outputs)

        new_predictions = test_model(model, loss_fn, test_loader)
        new_predictions["epoch"] = [epoch] * len(new_predictions)
        predictions = pd.concat([predictions, new_predictions])

        # Remove the hooks that save the activations
        if "unregister_hooks" in locals():
            unregister_hooks()

        train_losses.append(
            train_loss := train_epoch(
                model,
                optimizer,
                loss_fn,
                train_loader[epoch] if isinstance(train_loader, list) else train_loader,
                l1_lambda=l1_lambda,
                noise_mean=noise_mean,
                noise_stdev=noise_stdev,
                noise_multiplier=noise_multiplier,
                mode=mode,
            )
        )

        if epoch % 60 == 0:
            accuracy = predictions["accuracy"].mean()
            progress_bar.set_postfix(loss=train_loss, accuracy=accuracy)

        if callback is not None:
            callback(epoch, train_loss, predictions)

    df = test_loader.dataset.df
    inputs = pd.DataFrame(data=inputs, index=list(df.index) * epochs)
    outputs = pd.DataFrame(data=outputs, index=list(df.index) * epochs)
    inputs["epoch"] = sorted(list(range(epochs)) * len(df.index))
    outputs["epoch"] = sorted(list(range(epochs)) * len(df.index))

    test_losses = predictions[["epoch", "loss"]].groupby("epoch").sum()["loss"].to_list()
    losses = {"train": train_losses, "test": test_losses}

    metrics = Metrics(losses, predictions, parameters, inputs, outputs)

    model.eval()

    return metrics


def train_model(
    model: nn.Module,
    optimizer: AdamW,
    loss_fn: nn.MSELoss,
    dataloader: DataLoader,
    test_loader: DataLoader = None,
    epochs: int = 2500,
    noise_mean: float = None,
    noise_stdev: float = None,
    noise_multiplier: float = 0.1,
    l1_lambda: float = 0,
    replay_interval: int = None,
    replay_duration: int = None,
    replay_strategy: Literal[
        "baseline",
        "uniform",
        "conditional",
        "multimodal",
        "multivariate",
    ] = "baseline",
    reset=True,
    callback=None,
) -> list[Metrics]:
    """
    Train a model, with replay or not.
    """
    if replay_interval != None and replay_duration != None:
        assert replay_interval < epochs
        blocks = [("train", replay_interval)]
        while sum([x[1] for x in blocks]) < epochs:
            if blocks[-1][0] == "train":
                blocks.append(("replay", replay_duration))
            else:
                blocks.append(("train", replay_interval))
    else:
        blocks = [("train", epochs)]

    replay_layer = model.encoder if isinstance(model, AutoEncoder) else None

    metrics_list = []

    for mode, duration in blocks:
        if mode == "replay":
            # from _work.src.data.dataloader import create_replay_dataloader

            # outputs = metrics_list[-1].outputs
            # last_epoch = outputs["epoch"].max()

            # replay_dataloader = create_replay_dataloader(
            #     1 * len(outputs.index.unique()),
            #     dataloader,
            #     outputs[outputs["epoch"] > (last_epoch - min(last_epoch, 100))],
            #     "encoder.relu1",
            #     strategy=replay_strategy,
            # )

            # kwargs = {
            #     "test_loader": dataloader,
            #     "train_loader": replay_dataloader,
            # }

            # toggle_learning(replay_layer, False)
            pass
        else:
            kwargs = {
                "train_loader": dataloader,
            }

            if test_loader is not None:
                kwargs["test_loader"] = test_loader

            toggle_learning(replay_layer, True)

        metrics_list.append(
            metrics := train_epochs(
                model,
                optimizer,
                loss_fn,
                reset=reset,
                epochs=duration,
                noise_mean=noise_mean,
                noise_stdev=noise_stdev,
                noise_multiplier=noise_multiplier,
                l1_lambda=l1_lambda,
                mode=mode,
                callback=callback,
                **kwargs,
            )
        )

    return metrics_list
