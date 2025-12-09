import os
import argparse
from typing import Dict
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pfns.bar_distribution import FullSupportBarDistribution
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.observational_dataloader_graph_prior import ObservationalDataLoader
from configs.default_configs import prior_config
from tfmplayground.utils import get_default_device
from graphpfn.interface import cross_validate


def compare_all(models: Dict, num_steps: int, filename: str):
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=42)
    rows = []
    for data in prior:
        # add sampled parameters to data frame
        sampled_params = data["graph_information"]["sampled_params"]
        flat = {}
        for _, inner_dict in sampled_params.items():
            for k, v in inner_dict.items():
                flat[k] = v
        rows.append(flat)
        # evaluate on model
        X = data['x'][0].cpu().numpy()
        y = data['y'][0].cpu().numpy()
        single_eval_pos = data['single_eval_pos']
        for name, model in models.items():
            prob_adj = data['graph_information']['prob_adj']
            score = cross_validate(model, X, y, single_eval_pos, 5, prob_adj=prob_adj)
            flat[name] = score
            flat[f"{name}_truncated"] = max([0.0, score])
    df = pd.DataFrame(rows)
    df.to_csv(f"{filename}.csv")
    
    col_labels = ['num_nodes', 'edge_prob', 'root_std', 'non_root_std', 'number_train_samples_per_dataset']
    log_scale = [False, True, True, True, False]
    n_cols = int(len(col_labels) ** 0.5)
    n_rows = (len(col_labels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()
    for i, label in enumerate(col_labels):
        ax = axes[i]
        x = df[label]
        for name, model in models.items():
            y = df[f"{name}_truncated"]
            # bucketing
            n_buckets = 10
            bins = np.linspace(x.min(), x.max(), n_buckets + 1)
            bucket_ids = np.digitize(x, bins) - 1
            bucket_centers = []
            bucket_means = []
            bucket_stds = []

            for b in range(n_buckets):
                mask = bucket_ids == b
                if mask.any():
                    bucket_center = (bins[b] + bins[b+1]) / 2
                    bucket_centers.append(bucket_center)
                    bucket_means.append(y[mask].mean())
                    bucket_stds.append(y[mask].std())
            centers_arr = np.array(bucket_centers)
            means_arr = np.array(bucket_means)
            stds_arr = np.array(bucket_stds)
            upper_bound = means_arr + stds_arr
            lower_bound = means_arr - stds_arr
            line, = ax.plot(centers_arr, means_arr, marker='o', label=name)
            color = line.get_color()
            # ax.plot(centers_arr, upper_bound, alpha=0.3, linestyle='--', color=color)
            # ax.plot(centers_arr, lower_bound, alpha=0.3, linestyle='--', color=color)
            if log_scale[i]:
                ax.set_xscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel("R²")
        ax.grid(True)

    # hide unused axes
    for j in range(len(col_labels), len(axes)):
        axes[j].set_visible(False)
    # legend
    axes[0].legend()

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=500)
    plt.close(fig)
        
    
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    model_paths = {'pfn': 'workdir/nano_tab_pfn',
              'graph_prior': 'workdir/graph_prior',
              'binary': 'workdir/binary',
              'fallback': 'workdir/nano_tab_pfn_fallback',
              'graph_prior_fallback': 'workdir/graph_prior_fallback',
              'binary_fallback': 'workdir/binary_fallback'}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(name, f"{path}/latest_checkpoint.pth")
        buckets = torch.load(f"{path}/dist.pth")
        dist = FullSupportBarDistribution(buckets)
        reg = Regressor(model, dist, get_default_device())
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"evaluation/output/{datetime_str}", exist_ok=True)
    compare_all(models, args.steps, f"evaluation/output/{datetime_str}/full_comparison_prob_graph")