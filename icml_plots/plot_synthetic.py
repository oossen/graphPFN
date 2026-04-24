import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from datetime import datetime
import os

def plot_regressor_performance(metric: str, models: Dict, csv_path: str, output_path: str, num_buckets: int = 10, baseline_model=None):
    """Plots model performance with bootstrapped confidence intervals."""
    df = pd.read_csv(csv_path)
    metric_df = df[df['metric'] == metric].copy()
    x_axes = [
        'num_nodes', 
        'edge_prob', 
        'number_train_samples_per_dataset', 
        'root_std', 
        'non_root_std'
    ]
    
    if baseline_model in models:
        base_df = metric_df[metric_df['model'] == baseline_model][['ds_id', 'metric', 'value']].copy()
        base_df = base_df.rename(columns={'value': 'base_value'})
        metric_df = metric_df.merge(base_df, on=['ds_id', 'metric'])
        
        def calc_abs_improvement(row):
            metric_name = row['metric']
            model_val = row['value']
            base_val = row['base_value']
            if any(m in metric_name for m in ['nrmse', 'nll', 'mse', 'mae']):
                return base_val - model_val
            else:
                return model_val - base_val
            
        metric_df['value'] = metric_df.apply(calc_abs_improvement, axis=1)
    
    sns.set_style("whitegrid")
    for x_var in x_axes:
        plt.figure(figsize=(8, 5))
        # bucketize
        metric_df['bucket'] = pd.qcut(metric_df[x_var], q=num_buckets, duplicates='drop')
        bucket_means = metric_df.groupby('bucket', observed=True)[x_var].mean().to_dict()
        metric_df['x_bucket_plot_coord'] = metric_df['bucket'].map(bucket_means)
        # Iterate through models in the config to ensure style and order
        for model_name, style in models.items():
            if model_name == baseline_model:
                continue
            model_data = metric_df[metric_df['model'] == model_name]
            sns.lineplot(
                    data=model_data,
                    x='x_bucket_plot_coord',
                    y='value',
                    label=style['label'],
                    color=style['color'],
                    linestyle=style['ls'],
                    marker=style['marker'],
                    errorbar=('ci', 95),
                    markersize=7,
                    linewidth=2
                )
        
        plt.title(f'Model Performance: {metric.upper()} vs {x_var.replace("_", " ").title()}')
        plt.xlabel(x_var.replace("_", " ").title())
        plt.ylabel(metric.upper())
        plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        filename = f"{metric}_vs_{x_var}_{'rel' if baseline_model else 'abs'}.png"
        plt.savefig(f"{output_path}/{filename}")
        

if __name__ == "__main__":
    
    my_models = {
        'attention_beta': {
            'label': 'Attention', 
            'color': "#e74c3c", 
            'ls': '-', 
            'marker': 'o'
        },
        'gcn_beta': {
            'label': 'GCN', 
            'color': '#3498db', 
            'ls': '-', 
            'marker': 'o'
        },
        'baseline_beta': {
            'label': 'Baseline',
            'color': "#808080",
            'ls': '-',
            'marker': 'o'
        }
    }

    input_dir = "icml_plots/output/04_22_00_39"
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    result_file = f"{input_dir}/results.csv"
    output_dir = f"icml_plots/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    for metric in ['r2', 'nrmse', 'nll']:
        plot_regressor_performance(metric, my_models, result_file, output_dir)
        plot_regressor_performance(metric, my_models, result_file, output_dir, baseline_model='baseline_beta')