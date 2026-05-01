import shutil

import pandas as pd
import numpy as np
import os
import torch
from typing import Any

from tabpfn import TabPFNRegressor
from sap_rpt_oss import SAP_RPT_OSS_Regressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error

from graphpfn.interface import Regressor, init_model_from_state_dict_file
from configs.default_configs import training_config
from icml_plots.models import PFNWrapper, CausalExplorerRegressor


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
    
    df = pd.read_csv(dataset)
    X_full = df.iloc[:, :-1]
    y_full = df.iloc[:, -1]
    
    current_id = 0
    results = []

    for n in context_sizes:
        for _ in range(n_evals):
            # Sample context + test samples
            total_needed = n + n_query_samples
            if total_needed > len(df):
                raise ValueError(f"Dataset too small for context size {n} + {n_query_samples} test samples.")
            
            indices = rng.choice(df.index, size=total_needed, replace=False)
            context_idx = indices[:n]
            test_idx = indices[n:]
            
            X_train, y_train = X_full.iloc[context_idx], y_full.iloc[context_idx]
            X_test, y_test = X_full.iloc[test_idx], y_full.iloc[test_idx]
            
            task_path = os.path.dirname(dataset)
            workdir_path = f"{task_path}/workdir"
            os.makedirs(workdir_path, exist_ok=True)
            overview_path = f"{workdir_path}/overview.txt"
            with open(overview_path, 'w') as f:
                f.write(f"Index: {current_id}\n\n")
                f.write(df.to_string())

            for model_name, model in regs.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                for metric_name, metric_fn in metrics.items():
                    try:
                        score = metric_fn(y_test.values, preds)
                    except Exception as e:
                        print(f"Error computing {metric_name} for model {model_name} at context size {n}: {e}")
                        score = np.nan
                    
                    results.append({
                        "model": model_name,
                        "metric": metric_name,
                        "value": float(score),
                        "context_size": n,
                        "eval_id": current_id
                    })
            
            shutil.rmtree(workdir_path)
            current_id += 1

    results_df = pd.DataFrame(results)
    # Append to CSV: use header=True only if file doesn't exist
    file_exists = os.path.isfile(path)
    results_df.to_csv(path, mode='a', index=False, header=not file_exists)
    

if __name__ == "__main__":
    
    stage = 3
    
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
    
    for task in tasks:
        output_dir = f"icml_plots/output/real/{task}"
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = f"icml_plots/input/{task}/data.csv"
        
        regs = {}
        # 1 - Mean baselines and big TFMs
        if stage == 1:
            dataset_mean = pd.read_csv(dataset_path).iloc[:, -1].mean()
            regs = {'train_mean': DummyRegressor(strategy='mean'),
                                    'global_mean': DummyRegressor(strategy='constant', constant=dataset_mean),
                                    'tabpfn': TabPFNRegressor(),
                                    'contexttab': SAP_RPT_OSS_Regressor(),}
        
        # 2 - Oracle causal discovery and non-graph-conditioning baseline
        architectures = ['attention', 'gcn', 'attention_gcn']
        uncertainty_levels = ['binary', 'beta', 'uniform']
        if stage == 2:
            for arch in architectures:
                for uncertainty in uncertainty_levels:
                    model_name = f"{arch}_{uncertainty}"
                    reg = Regressor(init_model_from_state_dict_file(f"workdir/{model_name}/latest_checkpoint.pth"), training_config['buckets'])
                    causal_discovery_adj = torch.tensor(np.load(f"icml_plots/input/{task}/oracle_causal_discovery/probabilistic_adjacency.npy"), dtype=torch.float32)
                    regs[f"causal_discovery_oracle_{model_name}"] = PFNWrapper(reg, causal_discovery_adj)
            baseline_reg = Regressor(init_model_from_state_dict_file("workdir/baseline_beta/latest_checkpoint.pth"), training_config['buckets'])
            regs['baseline'] = PFNWrapper(baseline_reg)
            
        # 3 - Non-cheating causal discovery
        if stage == 3:
            for arch in architectures:
                for uncertainty in uncertainty_levels:
                    model_name = f"{arch}_{uncertainty}"
                    reg = Regressor(init_model_from_state_dict_file(f"workdir/{model_name}/latest_checkpoint.pth"), training_config['buckets'])
                    regs[f"causal_discovery_{model_name}"] = CausalExplorerRegressor(reg, f"{output_dir}/workdir")

        
        metrics = {'r2': r2_score, 'mse': mean_squared_error}
        context_sizes = [4, 8, 16, 32, 64]
        n_query_samples = 100
        n_evals = 100
        evaluate_regression_models(regs, dataset_path, metrics, context_sizes, n_query_samples, n_evals, f"{output_dir}/results_{stage}.csv")
    