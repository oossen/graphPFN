import os
import torch
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from graphpfn.interface import cross_validate
from configs.default_configs import buckets


df = pd.read_csv('evaluation/amazon_sales.csv', index_col='Date')
df = df.astype(float)
df = (df - df.mean()) / df.std()
y = df['Profit'].to_numpy()
X = df.drop('Profit', axis=1).to_numpy()
single_eval_positions = range(1, 20)

# features
# 0               1        2          3          4          5       6                7
# Shopping Event? Ad Spend Page Views Unit Price Sold Units Revenue Operational Cost Profit
adj = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0]])

model_paths = {'pfn': 'workdir/baseline_binary/latest_checkpoint.pth',
              'attention': 'workdir/attention_binary/latest_checkpoint.pth',}
models = {}
for name, path in model_paths.items():
    model = init_model_from_state_dict_file(path)
    reg = Regressor(model, buckets)
    models[name] = reg

def evaluate(model, single_eval_pos, prob_adj):
    return cross_validate(model, X, y, single_eval_pos, 20, prob_adj=prob_adj)

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"evaluation/output/{datetime_str}"
os.makedirs(output_dir, exist_ok=True)

for i in range(100):
    prob_adj = torch.rand((8, 8))
    results = {}
    for name, model in models.items():
        y_values = []
        for n in single_eval_positions:
            score = evaluate(model, n, prob_adj)
            y_values.append(score)
        results[name] = y_values
        print(results[name])

    plt.figure(figsize=(10, 6))  # Set the figure size
    for model_name, y_vals in results.items():
        plt.plot(single_eval_positions, y_vals, label=model_name, marker='.', linestyle='-')
    plt.xlabel("number of train samples")
    plt.ylabel("R²")
    plt.ylim(bottom=0, top=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{output_dir}/amazon_sales_{i}.png", dpi=500)
    plt.close()