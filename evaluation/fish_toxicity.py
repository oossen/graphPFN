import os
import torch
from datetime import datetime
from matplotlib import pyplot as plt
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from graphpfn.interface import cross_validate
from configs.default_configs import buckets
import openml
import numpy as np

from visualization.plotting import plot_adj


task = openml.tasks.get_task(363625, download_splits=False)
dataset = task.get_dataset(download_data=False)
X, y, _, _ = dataset.get_data(
    target=task.target_name, 
    dataset_format="array"
)
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()
single_eval_positions = [2, 4, 8, 16, 32, 64, 128, 256]

adj = torch.tensor(np.load("evaluation/input/probabilistic_adjacency.npy"), dtype=torch.float32)

model_paths = {'baseline': 'workdir/baseline_beta/latest_checkpoint.pth',
              'attention': 'workdir/attention_beta/latest_checkpoint.pth',}
models = {}
for name, path in model_paths.items():
    model = init_model_from_state_dict_file(path)
    reg = Regressor(model, buckets)
    models[name] = reg

def evaluate(model, single_eval_pos, prob_adj):
    return cross_validate(model, X, y, single_eval_pos, 50, prob_adj=prob_adj)

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"evaluation/output/{datetime_str}"
os.makedirs(output_dir, exist_ok=True)

generator = torch.Generator().manual_seed(1234)
incumbent_adj = torch.rand((9, 9), generator=generator)
incumbent_scores = [0.0 for _ in single_eval_positions]

# baseline scores
results = {}
y_values = []
for n in single_eval_positions:
    score = evaluate(models['baseline'], n, incumbent_adj)
    y_values.append(score)
results['baseline'] = y_values

for i in range(1000):
    adj_perturbation = 0.2 * (torch.rand((9, 9), generator=generator) - 0.5)
    prob_adj = incumbent_adj + adj_perturbation
    prob_adj = prob_adj.clamp(0, 1)  # Ensure probabilities are valid

    for name, model in [p for p in models.items() if p[0] != 'baseline']:
        y_values = []
        for n in single_eval_positions:
            score = evaluate(model, n, adj)
            y_values.append(score)
        results[name] = y_values
        
        new_incumbent = False
        rel_improvements = [(score - incumbent_score) / (abs(incumbent_score) + 1e-8) for score, incumbent_score in zip(results[name], incumbent_scores)]
        avg_rel_improvement = sum(rel_improvements) / len(rel_improvements)
        if avg_rel_improvement > 0:
            new_incumbent = True
            incumbent_scores = results[name]
            incumbent_adj = prob_adj

    plt.figure(figsize=(10, 6))  # Set the figure size
    for model_name, y_vals in results.items():
        plt.plot(single_eval_positions, y_vals, label=model_name, marker='.', linestyle='-')
    plt.xlabel("number of train samples")
    plt.ylabel("R²")
    plt.ylim(bottom=0, top=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{output_dir}/{i}_concrete{'_new_incumbent' if new_incumbent else ''}.png", dpi=500)
    plt.close()
    plot_adj(prob_adj, None, f"{output_dir}/{i}_adj{'_new_incumbent' if new_incumbent else ''}.png")
