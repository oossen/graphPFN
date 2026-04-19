from icml_plots.models import LLMRegressor
from icml_plots.eval_real import evaluate_regression_models
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime


task = "fish_toxicity"

dataset_path = f"icml_plots/input/{task}/data.csv"
reg = LLMRegressor()

metrics = {'r2': r2_score, 'mse': mean_squared_error}
context_sizes = [2, 4]
n_evals = 10
now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"icml_plots/output/{datetime_str}/{task}"
os.makedirs(output_dir, exist_ok=True)
evaluate_regression_models({'llm': reg}, dataset_path, metrics, context_sizes, n_evals, f"{output_dir}/results.csv")