from collections import defaultdict
from email.policy import default
from typing import Literal, Union

import numpy as np
import pandas as pd
import torch
from _work.src.modules.utils import add_noise, get_input_size, round_tensor
from scipy.stats import multivariate_normal, norm, rv_continuous
from torch import Tensor, nn
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader


class MixtureModel(rv_continuous):
    def __init__(self, submodels, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        if weights is None:
            weights = [1] * len(submodels)
        else:
            assert len(weights) == len(submodels)

        weights = np.array(weights)
        weights = np.divide(weights, weights.sum())
        assert sum(weights) == 1

        self.submodels = submodels
        self.weights = weights

    def _pdf(self, x):
        pdf = 0

        for weight, model in zip(self.weights, self.submodels):
            pdf += weight * model.pdf(x)

        return pdf

    def rvs(self, size):
        choices = np.random.choice(np.arange(len(self.submodels)), size=size, p=self.weights)
        samples = [model.rvs(size=size) for model in self.submodels]
        rvs = np.choose(choices, samples)
        return rvs


class MixtureOfGaussians(MixtureModel):
    def __init__(self, means, stdevs, *args, weights=None, **kwargs):
        assert len(means) == len(stdevs)
        submodels = [norm(loc=mean, scale=stdev) for mean, stdev in zip(means, stdevs)]
        super().__init__(submodels, *args, weights=weights, **kwargs)


def generate_samples(
    n_samples: int,
    outputs: pd.DataFrame,
    layer: str,
    strategy: Literal[
        "baseline",
        "uniform",
        "conditional",
        "multimodal",
        "multivariate",
        # "recirculation",
    ],
) -> Union[pd.Series, np.ndarray]:
    assert layer in outputs.columns
    outputs = outputs[[layer, "epoch"]]

    if strategy == "baseline":
        samples = outputs[-n_samples:][layer]

    if strategy == "uniform":
        samples = outputs.sample(n=n_samples)[layer]

    if strategy == "conditional":
        n_samples_per_class = n_samples / len(outputs.index.unique())
        assert n_samples_per_class.is_integer()
        samples = outputs.groupby(outputs.index).sample(int(n_samples_per_class))[layer]

    if strategy == "multimodal" or strategy == "multivariate":
        data = np.concatenate(outputs[layer].values)
        means = np.mean(data, axis=0)

        # TODO: fix: one mode _per class_ instead of per weight
        if strategy == "multimodal":
            stdevs = np.std(data, axis=0)
            distr = MixtureOfGaussians(means, stdevs, weights=None)
            samples = distr.rvs(size=(n_samples, data.shape[1]))

        if strategy == "multivariate":
            covariance_matrix = np.cov(data.transpose())
            distr = multivariate_normal(means, covariance_matrix)
            samples = distr.rvs(size=n_samples)

        samples = pd.Series(
            list(samples),
            index=[f"replay_item_{i}" for i in range(len(samples))],
        )

    return samples


@torch.no_grad()
def recirculate(
    model: nn.Module,
    dataloader: DataLoader,
    mean=None,
    sd=None,
    steps: int = 10,
    runs: int = 100,
):
    model.eval()

    transitions = defaultdict(lambda: defaultdict(lambda: 0))
    chain = defaultdict(list)

    for _ in range(runs):
        for i, batch in enumerate(dataloader.dataset):
            item = dataloader.dataset.df.index[i]
            input = batch["y"]
            input = add_noise(input, mean, sd, multiplier=0.1)

            for step in range(steps):
                output = model(input)

                chain[item].append(output.squeeze().tolist())

                input = output
                # if mean != None and sd != None:
                #     # factor = 1 - step / steps
                #     factor = 1
                #     input = add_noise(input, mean, sd * factor, multiplier=0.1 * factor)
                input[input > 1] = 1
                input[input < 0] = 0
                # input = input.round()

    return dict(chain)


@torch.no_grad()
def iterate(model, initial_state=None, mean=None, stdev=None, multiplier=1, runs=1, steps=10):
    model.eval()

    layer = dict(model.named_modules())["decoder.linear1"]

    states = []
    activations = []

    for run in range(runs):
        if initial_state is not None:
            input = initial_state
        else:
            a, b = -4, 4
            input = (a - b) * torch.rand(get_input_size(model)) + b
            # input = torch.rand(get_input_size(model))

        if mean is not None and stdev is not None:
            input = add_noise(input, mean, stdev, multiplier=multiplier)

        states.append([input])
        activ = []

        def save_hook(model, input, output):
            activ.append(input[0].clone().detach())

        hook = layer.register_forward_hook(save_hook)

        for _ in range(steps):
            output = model(input)

            states[run].append(output.squeeze())
            # states[run].append(input.squeeze())

            input = output
            if mean is not None and stdev is not None:
                input = add_noise(input, mean, stdev, multiplier=multiplier)
            input[input > 1] = 1
            input[input < 0] = 0

        states[run] = torch.vstack(states[run])
        activations.append(torch.vstack(tuple(activ)))
        hook.remove()

    states = torch.stack(states)
    activations = torch.stack(activations)
    return states, activations


def remove_duplicates(a, min_occurrence=2, decimals=4, round=True):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)

    assert len(a.shape) == 2

    if round is True:
        a = round_tensor(a, decimals=decimals)
    rounded, counts = np.unique(a, axis=0, return_counts=True)
    rounded = np.array(
        [
            *map(
                lambda x: x[1],
                filter(lambda x: counts[x[0]] >= min_occurrence, enumerate(rounded)),
            ),
        ]
    )
    counts = counts[counts >= min_occurrence]

    indices = set()
    if round is True:
        for i, row in enumerate(rounded):
            if i in indices:
                continue

            matches = []

            for j, other_row in enumerate(rounded):
                if i == j or j in indices:
                    continue

                if np.all(np.isclose(row, other_row, atol=0.02)):
                    matches.append(j)

            if len(matches) == 1 and matches[0] < i:
                indices.add(matches[0])
            elif len(matches) > 1:
                indices.update(matches)

        mask = np.ones(len(rounded), dtype=bool)
        mask[[*indices]] = False
        masked_tensor = torch.tensor(rounded[mask, ...])
        masked_counts = counts[mask, ...]

        order = np.array(
            [
                *map(
                    lambda x: x[0],
                    sorted(
                        enumerate(masked_counts),
                        key=lambda x: x[1],
                        reverse=True,
                    ),
                ),
            ]
        )
        return masked_tensor[order], masked_counts[order]

    return torch.tensor(rounded), counts


def get_attractors(
    model,
    runs=1,
    steps=10,
    burn_in_time=None,
    states_or_activations="states",
    min_occurrence=5,
    decimals=4,
    round=True,
):
    if burn_in_time is None:
        burn_in_time = steps // 10

    assert steps > burn_in_time

    states, activations = iterate(model, runs=runs, steps=steps)

    states = torch.vstack(tuple(states[:, burn_in_time:, :]))
    uniq_states, state_counts = remove_duplicates(states, min_occurrence=min_occurrence, round=round)

    activations = torch.vstack(tuple(activations[:, burn_in_time:, :]))
    uniq_activations, activ_counts = remove_duplicates(
        activations, min_occurrence=min_occurrence, round=round, decimals=decimals,
    )

    return (uniq_states, state_counts), (uniq_activations, activ_counts)

    if states_or_activations == "states":
        states = torch.vstack(tuple(states[:, burn_in_time:, :]))
        uniq_states, counts = remove_duplicates(states, min_occurrence=min_occurrence, round=round)
        return uniq_states, counts
    else:
        activations = torch.vstack(tuple(activations[:, burn_in_time:, :]))
        uniq_activations, counts = remove_duplicates(activations, min_occurrence=min_occurrence, round=round)
        return uniq_activations, counts


@torch.no_grad()
def get_fixed_points(model, initial_state_fn=None, n_runs=500, n_steps=100, min_occurrence=1, return_counts=False):
    model.eval()

    fixed_points = []
    for _ in range(n_runs):
        x = torch.randn(9) if initial_state_fn is None else initial_state_fn()
        for _ in range(n_steps):
            y = model(x)
            if torch.allclose(x, y, atol=1e-10, rtol=1e-7):
                fixed_points.append(x)
                break
            x = y

    fixed_points = torch.vstack(fixed_points)
    uniqs, indices, counts = torch.unique(
        round_tensor(fixed_points, decimals=3),
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    t = [[] for _ in indices.unique()]
    for i, j in enumerate(indices):
        t[j].append(fixed_points[i])
    t = [torch.vstack(x).mean(dim=0) for x in t]

    mask = counts > min_occurrence
    counts = counts[mask, ...]
    points = torch.vstack(t)[mask, ...]
    sorted_points = torch.vstack([x for _, x in sorted(zip(counts, points), key=lambda x: x[0], reverse=True)])

    if return_counts is True:
        return sorted_points, [x.item() for x in sorted(counts, reverse=True)]

    return sorted_points


def get_jacobian(model, fixed_point, it=1):
    model.eval()

    is_fixed_point = torch.allclose(model(fixed_point), fixed_point, atol=0.05)
    assert is_fixed_point

    def f(*args):
        x = torch.vstack(args)
        for _ in range(it):
            y = model(x)
            x = y
        return y

    return jacobian(f, fixed_point)


def is_attractor(model, fixed_point):
    j = get_jacobian(model, fixed_point)
    m = j.abs().max()
    return (m < 1).item()
