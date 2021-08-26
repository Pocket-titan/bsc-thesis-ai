import os
import sys

sys.path.append("../../../")

import matplotlib.pyplot as plt
import ray
import seaborn as sns
import torch
from ray import tune

# ray.init(local_mode=True)

# os.environ["TUNE_DISABLE_DATED_SUBDIR"] = 1


def trf(config):
    import sys

    sys.path.append("/home/jelmar/Github/deep/")

    from _work.src.data import small_dataloader as dataloader
    from _work.src.modules import AutoEncoder, train_model

    model = AutoEncoder(sizes=config["sizes"], batch_norm=config["b_norm"], dropout=config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss(reduction="sum")

    def callback(epoch, loss, predictions):
        accuracy = predictions["accuracy"].mean()
        tune.report(mean_loss=loss)

    metrics = train_model(
        model,
        optimizer,
        loss_fn,
        dataloader,
        epochs=config["epochs"],
        noise_mean=config["mean"],
        noise_stdev=config["stdev"],
        noise_multiplier=config["mult"],
        l1_lambda=config["l1"],
        callback=callback,
    )


analysis = tune.run(
    trf,
    config={
        "sizes": [9, 2],
        "b_norm": tune.grid_search([True, False]),
        "dropout": 0,
        "lr": 1e-3,
        "epochs": 1500,
        "mean": tune.grid_search([0, 0.5, 1]),
        "stdev": tune.grid_search([0.5, 1, 1.5]),
        "mult": tune.grid_search([0, 0.1, 0.5, 1]),
        "l1": 0,
    },
    # resources_per_trial={"cpu": 1},
    num_samples=1,
)
