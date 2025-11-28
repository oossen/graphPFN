import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from evaluation.evaluate import evaluate, evaluate_on_markov_blanket
from priors.observational_dataloader import ObservationalDataLoader
from configs.favorable_prior import prior_config
from tfmplayground.utils import get_default_device
from datetime import datetime


def remove_outliers(x):
    q1 = np.percentile(x, 10)
    q3 = np.percentile(x, 90)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (x >= lower) & (x <= upper)


def compare(model_1, model_2, num_samples, filename, eval_1_on_blanket=False, eval_2_on_blanket=False, include_scatter=False):
    prior = ObservationalDataLoader(num_steps=num_samples, batch_size=1, prior_config=prior_config, seed=42)
    if eval_1_on_blanket:
        df1 = evaluate_on_markov_blanket(model_1, prior)
    else:
        df1 = evaluate(model_1, prior)
    prior = ObservationalDataLoader(num_steps=num_samples, batch_size=1, prior_config=prior_config, seed=42)
    if eval_2_on_blanket:
        df2 = evaluate_on_markov_blanket(model_2, prior)
    else:
        df2 = evaluate(model_2, prior)
    
    col_labels = ['num_nodes', 'edge_prob', 'root_std', 'non_root_std', 'number_train_samples_per_dataset']
    log_scale = [False, True, True, True, False]
    n_cols = int(len(col_labels) ** 0.5)
    n_rows = (len(col_labels) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()
    
    print(f"Mean difference is {df2['R2'].mean() - df1['R2'].mean()}.")
    print(f"Median difference is {(df2['R2'] - df1['R2']).median()}.")
    
        
    for i, label in enumerate(col_labels):
        ax = axes[i]

        diff = df2['R2'] - df1['R2']
        outlier_mask = remove_outliers(diff)

        x = df1[label][outlier_mask]
        y = diff[outlier_mask]

        # original scatter
        if include_scatter:
            ax.scatter(x, y, alpha=0.2, color='gray')

        # bucketing
        n_buckets = 20
        bins = np.linspace(x.min(), x.max(), n_buckets + 1)
        bucket_ids = np.digitize(x, bins) - 1

        bucket_centers = []
        bucket_means = []

        for b in range(n_buckets):
            mask = bucket_ids == b
            if mask.any():
                bucket_centers.append((bins[b] + bins[b+1]) / 2)
                bucket_means.append(y[mask].mean())

        # overlay bucket means
        ax.plot(bucket_centers, bucket_means, marker='o', color='red')
        if log_scale[i]:
            ax.set_xscale('log')

        ax.set_xlabel(label)
        ax.set_ylabel("R2 improvement")
        ax.grid(True)

    # hide unused axes
    for j in range(len(col_labels), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    

import argparse
from pfns.bar_distribution import FullSupportBarDistribution
from graphpfn.interface import Regressor, init_model_from_state_dict_file


parser = argparse.ArgumentParser()
parser.add_argument("--dir_1", type=str, required=True)
parser.add_argument("--model_1", type=str, required=True)
parser.add_argument("--dir_2", type=str, required=True)
parser.add_argument("--model_2", type=str, required=True)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--blanket_1", action="store_true")
parser.add_argument("--blanket_2", action="store_true")
parser.add_argument("--scatter", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    model_1 = init_model_from_state_dict_file(args.model_1, f"{args.dir_1}/latest_checkpoint.pth")
    model_2 = init_model_from_state_dict_file(args.model_2, f"{args.dir_2}/latest_checkpoint.pth")
    buckets_1 = torch.load(f"{args.dir_1}/dist.pth")
    buckets_2 = torch.load(f"{args.dir_2}/dist.pth")
    dist_1 = FullSupportBarDistribution(buckets_1)
    dist_2 = FullSupportBarDistribution(buckets_2)
    reg_1 = Regressor(model_1, dist_1, get_default_device())
    reg_2 = Regressor(model_2, dist_2, get_default_device())

    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"evaluation/output/{datetime_str}", exist_ok=True)
    compare(reg_1,
            reg_2,
            args.steps,
            f"evaluation/output/{datetime_str}/comparisons.png",
            eval_1_on_blanket=args.blanket_1,
            eval_2_on_blanket=args.blanket_2,
            include_scatter=args.scatter)