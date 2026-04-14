import pandas as pd
import numpy as np
import os
from datetime import datetime
import torch
from typing import Any

from tabpfn import TabPFNRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error

from graphpfn.interface import Regressor, init_model_from_state_dict_file
from configs.default_configs import training_config
from icml_plots.models import PFNWrapper


def evaluate_regression_models(
    regs: dict,
    dataset: str,
    metrics: dict,
    context_sizes: list[int],
    n_evals: int,
    path: str
):
    """
    Evaluates regression models on a dataset across different context sizes.
    
    Args:
        regs: Dictionary of model names to model objects (sklearn-like).
        dataset: Path to the CSV dataset.
        metrics: Dictionary of metric names to callables (y_pred, y_true) -> float.
        context_sizes: List of integers representing training set sizes.
        n_evals: Number of random samples to run per context size.
        path: Path to save/append the results CSV.
    """
    # Load dataset
    df = pd.read_csv(dataset)
    X_full = df.iloc[:, :-1]
    y_full = df.iloc[:, -1]
    
    # Determine the starting ID for this run
    current_id = 0
    file_exists = os.path.isfile(path)
    if file_exists:
        existing_df = pd.read_csv(path)
        if not existing_df.empty:
            current_id = existing_df['eval_id'].max() + 1

    results = []

    for n in context_sizes:
        for _ in range(n_evals):
            # Sample context + 100 test samples
            total_needed = n + 100
            if total_needed > len(df):
                raise ValueError(f"Dataset too small for context size {n} + 100 test samples.")
            
            # Get random indices for this specific evaluation instance
            indices = np.random.choice(df.index, size=total_needed, replace=False)
            context_idx = indices[:n]
            test_idx = indices[n:]
            
            X_train, y_train = X_full.iloc[context_idx], y_full.iloc[context_idx]
            X_test, y_test = X_full.iloc[test_idx], y_full.iloc[test_idx]

            for model_name, model in regs.items():
                # Fit and predict
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Apply all metrics
                for metric_name, metric_fn in metrics.items():
                    score = metric_fn(y_test.values, preds)
                    
                    results.append({
                        "model": model_name,
                        "metric": metric_name,
                        "value": float(score),
                        "context_size": n,
                        "eval_id": current_id
                    })
            
            # Increment ID for the next random subsample
            current_id += 1

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Append to CSV: use header=True only if file doesn't exist
    results_df.to_csv(path, mode='a', index=False, header=not file_exists)
    

if __name__ == "__main__":
    
    task = "fish_toxicity"
    
    dataset_path = f"icml_plots/input/{task}/data.csv"
    regs: dict[str, Any] = {'dummy': DummyRegressor(strategy='mean'),
                            'tabpfn': TabPFNRegressor()}
    att_reg = Regressor(init_model_from_state_dict_file("workdir/attention_beta/latest_checkpoint.pth"), training_config['buckets'])
    prob_adjs = {'causal_discovery': torch.tensor(np.load(f"icml_plots/input/{task}/oracle_causal_discovery/probabilistic_adjacency.npy"), dtype=torch.float32),
                 'evolution': torch.tensor(np.load(f"icml_plots/input/{task}/oracle_evolutionary/best_matrix.npy"), dtype=torch.float32)}
    for name, prob_adj in prob_adjs.items():
        regs[f'att_{name}'] = PFNWrapper(att_reg, prob_adj)
    baseline_reg = Regressor(init_model_from_state_dict_file("workdir/baseline_beta/latest_checkpoint.pth"), training_config['buckets'])
    regs['baseline'] = PFNWrapper(baseline_reg)
    metrics = {'r2': r2_score, 'mse': mean_squared_error}
    context_sizes = [2, 4, 8, 16, 32, 64]
    n_evals = 100
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"icml_plots/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    evaluate_regression_models(regs, dataset_path, metrics, context_sizes, n_evals, f"{output_dir}/results.csv")
    