import os
import torch
from datetime import datetime
from matplotlib import pyplot as plt
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from graphpfn.interface import cross_validate
from configs.default_configs import buckets
import pandas as pd
import numpy as np

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

model_paths = {'baseline': 'workdir/baseline_beta_03_30_19_12/latest_checkpoint.pth',
                'attention': 'workdir/attention_beta_03_30_19_13/latest_checkpoint.pth',}
models = {}
for name, path in model_paths.items():
    model = init_model_from_state_dict_file(path)
    reg = Regressor(model, buckets)
    models[name] = reg
    
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
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    n_nodes = X.shape[1] + 1
    
    def evaluate(model, single_eval_pos, prob_adj):
        return cross_validate(model, X, y, single_eval_pos, 50, prob_adj=prob_adj)

    causal_discovery_adj = torch.tensor(np.load(f"evaluation/input/{task}/probabilistic_adjacency.npy"), dtype=torch.float32)
    arbitrary_adj = (torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)) / 3

    incumbent_adj = torch.rand((n_nodes, n_nodes), generator=generator)
    incumbent_scores = [0.0 for _ in single_eval_positions]

    # baseline scores
    results = {}
    y_values = []
    for n in single_eval_positions:
        score = evaluate(models['baseline'], n, incumbent_adj)
        y_values.append(score)
    results['baseline'] = y_values
    
    # causal discovery scores
    y_values = []
    for n in single_eval_positions:
        score = evaluate(models['attention'], n, causal_discovery_adj)
        y_values.append(score)
    results['causal_discovery'] = y_values
    
    # arbitrary graph
    y_values = []
    for n in single_eval_positions:
        score = evaluate(models['attention'], n, arbitrary_adj)
        y_values.append(score)
    results['arbitrary'] = y_values

    for i in range(10):
        adj_perturbation = 0.2 * (torch.rand((n_nodes, n_nodes), generator=generator) - 0.5)
        prob_adj = incumbent_adj + adj_perturbation
        prob_adj = prob_adj.clamp(0, 1)  # Ensure probabilities are valid

        for name, model in [p for p in models.items() if p[0] != 'baseline']:
            y_values = []
            for n in single_eval_positions:
                score = evaluate(model, n, prob_adj)
                y_values.append(score)
            results[f"{name}_evolutionary"] = y_values
            
            new_incumbent = False
            rel_improvements = [(score - incumbent_score) / (abs(incumbent_score) + 1e-8) for score, incumbent_score in zip(results[f"{name}_evolutionary"], incumbent_scores)]
            avg_rel_improvement = sum(rel_improvements) / len(rel_improvements)
            if avg_rel_improvement > 0:
                new_incumbent = True
                incumbent_scores = results[f"{name}_evolutionary"]
                incumbent_adj = prob_adj

    plt.figure(figsize=(10, 6))  # Set the figure size
    for model_name, y_vals in results.items():
        plt.plot(single_eval_positions, y_vals, label=model_name, marker='.', linestyle='-')
    plt.xlabel("number of train samples")
    plt.ylabel("R²")
    # plt.ylim(bottom=0, top=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{output_dir}/{i}_{task}{'_new_incumbent' if new_incumbent else ''}.png", dpi=500)
    plt.close()
    plot_adj(prob_adj, None, f"{output_dir}/{task}_adj{'_new_incumbent' if new_incumbent else ''}.png")
