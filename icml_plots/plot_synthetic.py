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

    fancy_names = {'edge_prob': 'Edge probability',
                    'non_root_std': 'Average noise standard deviation at non-root nodes',
                    'root_std': 'Average noise standard deviation at root nodes',
                    'num_nodes': 'Number of features',
                    'number_train_samples_per_dataset': 'Context size'}
    metric_df.rename(columns=fancy_names, inplace=True)
    metric_df['Number of features'] = metric_df['Number of features'] - 1 # target is included in count
    
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
    for x_var in fancy_names.values():
        plt.figure(figsize=(8, 6))
        sns.set_context("paper", font_scale=2.0)
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
                    err_style='bars',
                    err_kws={'capsize': 5},
                    markersize=7,
                    linewidth=2
                )
        
        plt.xlabel(x_var)
        addition = " (improvement over baseline)" if baseline_model else ""
        fancy_metric_names = {'r2': f'$R^2${addition}', 'nrmse': f'NRMSE{addition}', 'nll': f'NLL{addition}'}
        plt.ylabel(fancy_metric_names[metric])
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
            title=None,
            frameon=True
        )
        plt.tight_layout()

        filename = f"{metric}_vs_{x_var}_{'rel' if baseline_model else 'abs'}.png"
        plt.savefig(f"{output_path}/{filename}")
        

if __name__ == "__main__":
    
    uncertainty_level = 'beta'
    
    my_models = {
        f'attention_{uncertainty_level}': {
            'label': 'Attention', 
            'color': "#ffc400", 
            'ls': '-', 
            'marker': 'o'
        },
        f'gcn_{uncertainty_level}': {
            'label': 'GCN', 
            'color': "#960000", 
            'ls': '-', 
            'marker': 'o'
        },
        f'attention_gcn_{uncertainty_level}': {
            'label': 'Attention + GCN', 
            'color': "#F471F8", 
            'ls': '-', 
            'marker': 'o'
        },
        f'baseline_beta': {
            'label': 'Baseline',
            'color': "#808080",
            'ls': '-',
            'marker': 'o'
        }
    }

    input_dir = f"icml_plots/output/synthetic/{uncertainty_level}"
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    result_file = f"{input_dir}/results.csv"
    output_dir = f"icml_plots/output/synthetic/{uncertainty_level}"
    os.makedirs(output_dir, exist_ok=True)
    for metric in ['r2']:
        plot_regressor_performance(metric, my_models, result_file, output_dir)
        plot_regressor_performance(metric, my_models, result_file, output_dir, baseline_model=f'baseline_beta')