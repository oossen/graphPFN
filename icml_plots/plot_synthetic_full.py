import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

def plot_multi_regressor_performance(
    metric: str, 
    configs: List[Dict], # List of {'csv_path': str, 'models': dict, 'source_label': str}
    output_path: str, 
    num_buckets: int = 10, 
    baseline_model=None
):
    """
    Combines data from multiple CSVs into single plots per x_var.
    """
    all_data_frames = []
    
    fancy_names = {
        'edge_prob': 'Edge probability',
        'non_root_std': 'Average noise standard deviation at non-root nodes',
        'root_std': 'Average noise standard deviation at root nodes',
        'num_nodes': 'Number of features',
        'number_train_samples_per_dataset': 'Context size'
    }

    for config in configs:
        df = pd.read_csv(config['csv_path'])
        models = config['models']
        source_label = config.get('source_label', '') # e.g., "Small", "Large"
        
        metric_df = df[df['metric'] == metric].copy()
        metric_df.rename(columns=fancy_names, inplace=True)
        if 'Number of features' in metric_df.columns:
            metric_df['Number of features'] = metric_df['Number of features'] - 1
        
        # Handle Baseline Improvement logic per CSV
        if baseline_model in models:
            base_df = metric_df[metric_df['model'] == baseline_model][['ds_id', 'metric', 'value']].copy()
            base_df = base_df.rename(columns={'value': 'base_value'})
            metric_df = metric_df.merge(base_df, on=['ds_id', 'metric'])
            
            def calc_abs_improvement(row):
                m_name, m_val, b_val = row['metric'], row['value'], row['base_value']
                if any(m in m_name for m in ['nrmse', 'nll', 'mse', 'mae']):
                    return b_val - m_val
                return m_val - b_val
                
            metric_df['value'] = metric_df.apply(calc_abs_improvement, axis=1)

        # Tag the data so we can distinguish it in the plot
        # We also store the specific style info for this config
        metric_df['config_source'] = source_label
        all_data_frames.append((metric_df, models))

    sns.set_style("whitegrid")
    for x_var in fancy_names.values():
        plt.figure(figsize=(10, 7))
        sns.set_context("paper", font_scale=1.8)
        
        # Iterate through each config to plot its specific models
        for metric_df, models in all_data_frames:
            if x_var not in metric_df.columns:
                continue
                
            # Calculate buckets for this specific dataset
            temp_df = metric_df.copy()
            temp_df['bucket'] = pd.qcut(temp_df[x_var], q=num_buckets, duplicates='drop')
            bucket_means = temp_df.groupby('bucket', observed=True)[x_var].mean().to_dict()
            temp_df['x_bucket_plot_coord'] = temp_df['bucket'].map(bucket_means)

            for model_name, style in models.items():
                if model_name == baseline_model:
                    continue
                
                model_data = temp_df[temp_df['model'] == model_name]
                
                # Append source label to the model label to distinguish them in the legend
                full_label = f"{style['label']} ({temp_df['config_source'].iloc[0]})" if temp_df['config_source'].iloc[0] else style['label']
                
                sns.lineplot(
                    data=model_data,
                    x='x_bucket_plot_coord',
                    y='value',
                    label=full_label,
                    color=style['color'],
                    linestyle=style['ls'],
                    marker=style['marker'],
                    errorbar=('ci', 95),
                    err_style='bars',
                    err_kws={'capsize': 5},
                    markersize=8,
                    linewidth=2.5,
                    legend=False
                )

        plt.xlabel(x_var)
        addition = " (improvement over baseline)" if baseline_model else ""
        fancy_metric_names = {'r2': f'$R^2${addition}', 'nrmse': f'NRMSE{addition}', 'nll': f'NLL{addition}'}
        plt.ylabel(fancy_metric_names.get(metric, metric))
        
        # Legend outside of plot
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True)
        plt.tight_layout()

        filename = f"combined_{metric}_vs_{x_var}.png"
        plt.savefig(f"{output_path}/{filename}", bbox_inches='tight')
        plt.close()
        

if __name__ == "__main__":
    
    models_beta = {
        'attention_beta': {
            'label': 'Attention', 
            'color': "#ffc400", 
            'ls': '-', 
            'marker': 'o'
        },
        'gcn_beta': {
            'label': 'GCN', 
            'color': "#960000", 
            'ls': '-', 
            'marker': 'o'
        },
        'attention_gcn_beta': {
            'label': 'Attention + GCN', 
            'color': "#F471F8", 
            'ls': '-', 
            'marker': 'o'
        },
        'baseline_beta': {
            'label': 'Baseline',
            'color': "#808080",
            'ls': '-',
            'marker': 'o'
        }}
    models_binary = {'attention_binary': {
            'label': 'Attention', 
            'color': "#1100ff", 
            'ls': '-', 
            'marker': 'o'
        },
        'gcn_binary': {
            'label': 'GCN', 
            'color': "#2D5D8A", 
            'ls': '-', 
            'marker': 'o'
        },
        'attention_gcn_binary': {
            'label': 'Attention + GCN', 
            'color': "#01AFFF", 
            'ls': '-', 
            'marker': 'o'
        },
        'baseline_beta': {
            'label': 'Baseline',
            'color': "#808080",
            'ls': '-',
            'marker': 'o'
        }}
    models_uniform = {'attention_uniform': {
            'label': 'Attention', 
            'color': "#01632a", 
            'ls': '-', 
            'marker': 'o'
        },
        'gcn_uniform': {
            'label': 'GCN', 
            'color': "#10C000", 
            'ls': '-', 
            'marker': 'o'
        },
        'attention_gcn_uniform': {
            'label': 'Attention + GCN', 
            'color': "#A6FF00", 
            'ls': '-', 
            'marker': 'o'
        },
        'baseline_beta': {
            'label': 'Baseline',
            'color': "#808080",
            'ls': '-',
            'marker': 'o'
        }}
    
    input_dir = f"icml_plots/output/synthetic/"
    configs = [{'csv_path': f"{input_dir}/beta/results.csv", 'models': models_beta, 'source_label': 'Beta'},
               {'csv_path': f"{input_dir}/binary/results.csv", 'models': models_binary, 'source_label': 'Binary'},
               {'csv_path': f"{input_dir}/uniform/results.csv", 'models': models_uniform, 'source_label': 'Uniform'}]

    output_dir = f"icml_plots/output/synthetic/all/"
    os.makedirs(output_dir, exist_ok=True)
    for metric in ['r2']:
        plot_multi_regressor_performance(metric, configs, output_dir, baseline_model=f'baseline_beta')