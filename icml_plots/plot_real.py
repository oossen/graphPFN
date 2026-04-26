import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict
from datetime import datetime
import os

def plot_regressor_performance(metric: str, models: Dict, csv_path: str, output_path: str):
    """Plots model performance with bootstrapped confidence intervals."""
    df = pd.read_csv(csv_path)
    df_filtered = df[df['metric'] == metric].copy()
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    for model_key, style in models.items():
        model_data = df_filtered[df_filtered['model'] == model_key]     
        sns.lineplot(
            data=model_data,
            x='context_size',
            y='value',
            label=style.get('label', model_key),
            color=style.get('color', None),
            linestyle=style.get('ls', '-'),
            marker=style.get('marker', 'o'),
            markersize=8,
            errorbar=('ci', 95), # Bootstrapped 95% confidence interval
            n_boot=100
        )
    
    plt.title(f'Performance Comparison: {metric.upper()}', fontsize=14, pad=15)
    plt.xlabel('Context Size', fontsize=12)
    plt.ylabel(f'Value ({metric})', fontsize=12)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    

def plot_regressor_rank(metric: str, models: Dict, csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['metric'] == metric].copy()
    df_filtered = df_filtered[df_filtered['model'].isin(models.keys())]
    higher_is_better = metric.lower() in ['r2', 'accuracy', 'f1', 'precision', 'recall']
    ascending_logic = not higher_is_better
    df_filtered['rank'] = df_filtered.groupby(['context_size', 'eval_id'])['value'].rank(
        method='min', 
        ascending=ascending_logic
    )
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for model_key, style in models.items():
        model_data = df_filtered[df_filtered['model'] == model_key]
        
        sns.lineplot(
            data=model_data,
            x='context_size',
            y='rank',
            label=style.get('label', model_key),
            color=style.get('color', None),
            linestyle=style.get('ls', '-'),
            marker=style.get('marker', 'o'),
            markersize=8,
            errorbar=('ci', 95),
            n_boot=1000  # Increased for rank stability
        )
    
    plt.gca().invert_yaxis()
    max_rank = len(models)
    plt.yticks(range(1, max_rank + 1))
    plt.title(f'Average Model Rank: {metric.upper()} (Lower is Better)', fontsize=14, pad=15)
    plt.xlabel('Context Size', fontsize=12)
    plt.ylabel('Average Rank', fontsize=12)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)


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
    
    my_models = {
        'train_mean': {
            'label': 'Baseline (Mean)', 
            'color': "#777777", 
            'ls': '--', 
            'marker': 'o'
        },
        'tabpfn': {
            'label': 'TabPFN', 
            'color': '#777777', 
            'ls': '-', 
            'marker': 'o'
        },
        'att_causal_discovery': {
            'label': 'Oracle Causal', 
            'color': '#e74c3c', 
            'ls': '-', 
            'marker': 'o'
        },
        'att_evolution': {
            'label': 'Oracle Evolution', 
            'color': '#3498db', 
            'ls': '-', 
            'marker': 'o'
        },
        'att_arbitrary': {
            'label': 'Arbitrary', 
            'color': "#35cab6", 
            'ls': '-', 
            'marker': 'o'
        },
        'baseline': {
            'label': 'Baseline',
            'color': '#2ecc71',
            'ls': '-',
            'marker': 'o'
        },
        'global_mean': {
            'label': 'Global Mean',
            'color': '#777777',
            'ls': ':',
            'marker': 'o'
        },
        'causal': {
            'label': 'Causal',
            'color': '#9b59b6',
            'ls': '-',
            'marker': 'o'
        }
    }

    input_dir = "icml_plots/output/04_25_16_30"
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    for task in tasks:
        result_file = f"{input_dir}/{task}/results.csv"
        output_dir = f"icml_plots/output/{datetime_str}/{task}"
        os.makedirs(output_dir, exist_ok=True)
        plot_regressor_performance('r2', my_models, result_file, f"{output_dir}/plot_real_r2.png")
        plot_regressor_performance('mse', my_models, result_file, f"{output_dir}/plot_real_mse.png")
        plot_regressor_rank('r2', my_models, result_file, f"{output_dir}/plot_real_r2_rank.png")
        plot_regressor_rank('mse', my_models, result_file, f"{output_dir}/plot_real_mse_rank.png")