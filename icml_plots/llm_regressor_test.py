from icml_plots.models import LLMRegressor
from icml_plots.eval_real import evaluate_regression_models
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime
import torch
import numpy as np


task = "fish_toxicity"

dataset_path = f"icml_plots/input/{task}/data.csv"
reg = LLMRegressor()

prob_adj = torch.tensor(np.load(f"icml_plots/input/{task}/oracle_causal_discovery/probabilistic_adjacency.npy"), dtype=torch.float32)
reg_with_causal = LLMRegressor(prob_adj=prob_adj)

metrics = {'r2': r2_score, 'mse': mean_squared_error}
context_sizes = [2, 4]
n_query_samples = 5
n_evals = 5
now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"icml_plots/output/{datetime_str}/{task}"
os.makedirs(output_dir, exist_ok=True)
evaluate_regression_models({'llm': reg, 'llm_with_causal': reg_with_causal}, dataset_path, metrics, context_sizes, n_query_samples, n_evals, f"{output_dir}/results.csv")