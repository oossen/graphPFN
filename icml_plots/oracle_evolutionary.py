from graphpfn.interface import init_model_from_state_dict_file, Regressor
from configs.default_configs import training_config
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import os

from causal_discovery.utils.io_utils import save_array
from causal_discovery.utils.plot_utils import plot_adjacency_heatmap, plot_graph_from_adjacency_matrix
from icml_plots.evolution import run_evolution


n_train_samples = 100
n_test_samples = 100
n_tables = 10
num_generations = 100
num_parents_mating = 10
sol_per_pop = 20
seed = 42
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
    print(f"Processing task: {task}")
    df = pd.read_csv(f"icml_plots/input/{task}/data.csv")
    numeric_cols = df.select_dtypes(include=['number']).columns
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(n_tables):
        total_needed = n_train_samples + n_test_samples
        indices = rng.choice(len(df), size=total_needed, replace=False)
        train_df = df.loc[indices[:n_train_samples]].copy()
        test_df = df.loc[indices[n_train_samples:]].copy()
            
        # z-normalization
        train_mean = train_df[numeric_cols].mean()
        train_std = train_df[numeric_cols].std()
        # clip std to avoid division by zero
        train_std = np.maximum(train_std, 0.5)
        train_df[numeric_cols] = (train_df[numeric_cols] - train_mean) / train_std
        test_df[numeric_cols] = (test_df[numeric_cols] - train_mean) / train_std
        
        tables.append({
            'X_train': train_df.iloc[:, :-1].to_numpy(),
            'y_train': train_df.iloc[:, -1].to_numpy(),
            'X_test': test_df.iloc[:, :-1].to_numpy(),
            'y_test': test_df.iloc[:, -1].to_numpy(),
        })
    
    model = init_model_from_state_dict_file("workdir/attention_beta/latest_checkpoint.pth")
    buckets = training_config['buckets']
    reg = Regressor(model, buckets)
    metric = r2_score
    best_matrix = run_evolution(tables=tables,
                                model=reg,
                                metric=metric,
                                num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                sol_per_pop=sol_per_pop,
                                seed=seed)

    output_dir = f"icml_plots/input/{task}/oracle_evolutionary"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/best_matrix.npy"
    save_array(path=save_path, array=best_matrix)
    viz_path = f"{output_dir}/probabilistic_adjacency.png"
    plot_adjacency_heatmap(
        adjacency_matrix=best_matrix,
        title="Probabilistic Adjacency Matrix",
        path=viz_path,
    )
    graph_path = f"{output_dir}/graph.png"
    plot_graph_from_adjacency_matrix(
        adjacency_matrix=best_matrix,
        title="All likely edges",
        path=graph_path,
    )
    