import os
import argparse
from typing import Dict
import pandas as pd
from datetime import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from priors.observational_dataloader import ObservationalDataLoader
from configs.default_configs import prior_config, training_config
from graphpfn.interface import cross_validate


def compare_all(models: Dict, num_steps: int, filename: str):
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=42)
    rows = []
    for data in prior:
        # add sampled parameters to data frame
        sampled_params = data["graph_information"]["sampled_params"]
        row = {}
        for k, v in sampled_params.items():
            row[k] = v
        rows.append(row)
        # evaluate on model, select first and only batch
        X = data['x'][0].cpu().numpy()
        y = data['y'][0].cpu().numpy()
        single_eval_pos = data['single_eval_pos']
        for name, model in models.items():
            score = cross_validate(model, X, y, single_eval_pos, 5, **data['graph_information'])
            row[name] = score
    df = pd.DataFrame(rows)
    df.to_csv(f"{filename}.csv")
    
    col_labels = ['num_nodes', 'edge_prob', 'root_std', 'non_root_std', 'number_train_samples_per_dataset']
    use_log_scale = [False, True, True, True, False]
    df = df.melt(
        id_vars=col_labels, 
        value_vars=[name for name in models.keys()], 
        var_name='model', 
        value_name='score')
    bins = 10
    for label, log_scale in zip(col_labels, use_log_scale):
        if log_scale:
             df[label] = df[label].apply(lambda x: np.log(x) if pd.notnull(x) else x)
        df[f"{label}_binned"] = pd.cut(df[label], bins=bins, labels=False)
        bin_centers = pd.interval_range(start=df[label].min(), end=df[label].max(), periods=bins).mid
        df[f"{label}_center"] = df[f"{label}_binned"].map(lambda x: bin_centers[int(x)] if pd.notnull(x) else x)

    for label, log_scale in zip(col_labels, use_log_scale):
        if log_scale:
             df[label] = df[label].apply(lambda x: np.exp(x) if pd.notnull(x) else x)
        sns.lineplot(data=df, x=f"{label}_center", y='score', hue='model', errorbar='ci')
        plt.savefig(f"{filename}_{label}.png")
        plt.clf()
    
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    model_paths = {'pfn': 'workdir/baseline_02_10_14_16'}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"visualization/output/{datetime_str}", exist_ok=True)
    compare_all(models, args.steps, f"visualization/output/{datetime_str}/full_comparison")