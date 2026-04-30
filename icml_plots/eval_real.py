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
from icml_plots.models import PFNWrapper, LLMRegressor, CausalExplorerRegressor


def evaluate_regression_models(
    regs: dict,
    dataset: str,
    metrics: dict,
    context_sizes: list[int],
    n_query_samples: int,
    n_evals: int,
    path: str,
    seed: int = 42
):
    """
    Evaluates regression models on a dataset across different context sizes.
    
    Args:
        regs: Dictionary of model names to model objects (sklearn-like).
        dataset: Path to the CSV dataset.
        metrics: Dictionary of metric names to callables (y_pred, y_true) -> float.
        context_sizes: List of integers representing training set sizes.
        n_query_samples: Number of query samples to use for each evaluation.
        n_evals: Number of random samples to run per context size.
        path: Path to save/append the results CSV.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    
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
            # Sample context + test samples
            total_needed = n + n_query_samples
            if total_needed > len(df):
                raise ValueError(f"Dataset too small for context size {n} + {n_query_samples} test samples.")
            
            # Get random indices for this specific evaluation instance
            indices = rng.choice(df.index, size=total_needed, replace=False)
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
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    for task in tasks:
        output_dir = f"icml_plots/output/real/{task}"
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = f"icml_plots/input/{task}/data.csv"
        # create models
        dataset_mean = pd.read_csv(dataset_path).iloc[:, -1].mean()
        regs: dict[str, Any] = {'train_mean': DummyRegressor(strategy='mean'),
                                'global_mean': DummyRegressor(strategy='constant', constant=dataset_mean),
                                'tabpfn': TabPFNRegressor()}
        att_beta_reg = Regressor(init_model_from_state_dict_file("workdir/attention_beta/latest_checkpoint.pth"), training_config['buckets'])
        att_binary_reg = Regressor(init_model_from_state_dict_file("workdir/attention_binary/latest_checkpoint.pth"), training_config['buckets'])
        att_uniform_reg = Regressor(init_model_from_state_dict_file("workdir/attention_uniform/latest_checkpoint.pth"), training_config['buckets'])
        att_regs = {'beta': att_beta_reg, 'binary': att_binary_reg, 'uniform': att_uniform_reg}
        prob_adjs = {'causal_discovery': torch.tensor(np.load(f"icml_plots/input/{task}/oracle_causal_discovery/probabilistic_adjacency.npy"), dtype=torch.float32),
                    'evolution': torch.tensor(np.load(f"icml_plots/input/{task}/oracle_evolutionary/best_matrix.npy"), dtype=torch.float32),}
        prob_adjs['arbitrary'] = (torch.ones_like(prob_adjs['causal_discovery']) - torch.eye(prob_adjs['causal_discovery'].shape[0])) / 3
        for name, prob_adj in prob_adjs.items():
            for uncertainty_level in ['beta', 'binary', 'uniform']:
                regs[f'att_{uncertainty_level}_{name}'] = PFNWrapper(att_regs[uncertainty_level], prob_adj)
        baseline_reg = Regressor(init_model_from_state_dict_file("workdir/baseline_beta/latest_checkpoint.pth"), training_config['buckets'])
        regs['baseline'] = PFNWrapper(baseline_reg)
        # regs['llm'] = LLMRegressor()
        # regs['llm_causal'] = LLMRegressor(prob_adj=prob_adjs['causal_discovery'])
        regs['causal'] = CausalExplorerRegressor(att_beta_reg, f"{output_dir}/causal_explorer_workdir")
        
        metrics = {'r2': r2_score, 'mse': mean_squared_error}
        context_sizes = [4, 8, 16, 32, 64]
        n_query_samples = 100
        n_evals = 100
        evaluate_regression_models(regs, dataset_path, metrics, context_sizes, n_query_samples, n_evals, f"{output_dir}/results.csv")
    