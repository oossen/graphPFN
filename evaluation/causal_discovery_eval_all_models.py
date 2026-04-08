import os
import torch
from datetime import datetime
from matplotlib import pyplot as plt
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from graphpfn.interface import cross_validate
from configs.default_configs import buckets
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor

from visualization.plotting import plot_adj


tasks = ["fish_toxicity",
            "concrete_compressive_strength",
            "healthcare_insurance_expenses",
            "airfoil_self_noise",
            "used_fiat_500",
            "wine_quality",
            "miami_housing",
            "houses",
            "food_delivery_time",
            "physiochemical_protein",
            "diamonds",]

model_paths = {'attention_beta': 'workdir/attention_beta_04_07_18_14/latest_checkpoint.pth',
               'attention_binary': 'workdir/attention_binary_04_07_19_02/latest_checkpoint.pth',
               'attention_uniform': 'workdir/attention_uniform_04_07_18_45/latest_checkpoint.pth',
               'baseline_beta': 'workdir/baseline_beta_04_07_18_10/latest_checkpoint.pth',
               'baseline_binary': 'workdir/baseline_binary_04_07_19_00/latest_checkpoint.pth',
               'baseline_uniform': 'workdir/baseline_uniform_04_07_18_17/latest_checkpoint.pth',}
models = {}
for name, path in model_paths.items():
    model = init_model_from_state_dict_file(path)
    reg = Regressor(model, buckets)
    models[name] = reg
    
tabpfn_model = TabPFNRegressor()
models['tabpfn'] = tabpfn_model
    
single_eval_positions = [2, 4, 8, 16, 32, 64, 128, 256]

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"evaluation/output/{datetime_str}"
os.makedirs(output_dir, exist_ok=True)

seed = 42
generator = torch.Generator().manual_seed(1234)

for task in tasks:
    df = pd.read_csv(f'evaluation/input/{task}/data.csv')
    df = df.sample(n=min(len(df), 500), random_state=seed)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    n_nodes = X.shape[1] + 1
    
    def evaluate(model, single_eval_pos, n_evals=5, **kwargs):
        if isinstance(model, TabPFNRegressor):
            return cross_validate(model, X, y, single_eval_pos, n_evals)
        elif isinstance(model, Regressor):
            return cross_validate(model, X, y, single_eval_pos, n, **kwargs)

    causal_discovery_adj = torch.tensor(np.load(f"evaluation/input/{task}/probabilistic_adjacency.npy"), dtype=torch.float32)

    # baseline scores
    results = {}
    for name, model in models.items():
        if name.startswith('baseline'):
            y_values = []
            for n in single_eval_positions:
                score = evaluate(model, n)
                y_values.append(score)
            results[name] = y_values
    
    # tabpfn scores
    y_values = []
    for n in single_eval_positions:
        score = evaluate(models['tabpfn'], n)
        y_values.append(score)
    results['tabpfn'] = y_values
    
    # causal discovery scores
    for name, model in models.items():
        if name.startswith('attention'):
            y_values = []
            for n in single_eval_positions:
                score = evaluate(model, n, prob_adj=causal_discovery_adj)
                y_values.append(score)
            results[name] = y_values

    plt.figure(figsize=(10, 6))  # Set the figure size
    for model_name, y_vals in results.items():
        plt.plot(single_eval_positions, y_vals, label=model_name, marker='.', linestyle='-')
    plt.xlabel("number of train samples")
    plt.ylabel("R²")
    # plt.ylim(bottom=0, top=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{output_dir}/{task}.png", dpi=500)
    plt.close()
