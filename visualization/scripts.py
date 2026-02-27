import argparse
from datetime import datetime
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from visualization.mcmc import mcmc_suite
from priors.observational_dataloader import ObservationalDataLoader

from visualization.full_comparison import compare_all
from graphpfn.interface import init_model_from_state_dict_file
from tfmplayground.utils import get_default_device
    

def ppd_plots():
    from configs.simple_configs import prior_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 45
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior_config['graph_config']['num_nodes'] = {'value': 3}
    prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": ["binary"], "probabilities": [1.0]}}
    prior = ObservationalDataLoader(1000, 1, prior_config, seed=seed)
    prior._make_statistics(steps=10000)
    
    mcmc_parameters = (10000, 1000, 2) # steps, burn-in, thinning
    train_sample_sizes = [2, 10, 20, 50]
    
    models = [{"name": "simple_binary", "color": "red", "label": "p(y|x, D) (PFN)"},
              {"name": "simple_attention_binary", "color": "orange", "label": "p(y|x, D, γ) (PFN))"}]
    
    for i in range(0, 10):
        for num_train_samples in train_sample_sizes:
            mcmc_suite(prior, 
                       num_train_samples, 
                       generator, 
                       f"{output_dir}/run_{num_train_samples}_{i}", 
                       include_mcmc=True, 
                       include_pfns=models, 
                       mcmc_parameters=mcmc_parameters)
            
    # KL divergence plots
    def make_kl_plot(model_configs, filepath: str, errorbar='ci'):
        kl_input_df = pd.read_csv(f"{output_dir}/kl.csv")
        plt.figure(figsize=(10, 6))
        
        for cfg in model_configs:
            p_name, q_name = cfg['pair']
            results = []
            for (run_id, n), group in kl_input_df.groupby(['run_id', 'num_train_samples']):
                group = group.sort_values('y')
                y = group['y'].to_numpy(dtype=float)
                p = group[p_name].to_numpy(dtype=float)
                q = group[q_name].to_numpy(dtype=float)
                
                dy = np.diff(y)
                dy = np.append(dy, dy[-1])
                
                p = p / (np.sum(p * dy) + 1e-12)
                q = q / (np.sum(q * dy) + 1e-12)

                kl_val = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)) * dy)
                results.append({
                    'num_train_samples': n,
                    'kl_divergence': kl_val
                })

            temp_df = pd.DataFrame(results)
            sns.lineplot(
                data=temp_df, 
                x='num_train_samples', 
                y='kl_divergence', 
                label=cfg.get('label', f"{p_name} vs {q_name}"),
                color=cfg.get('color'),
                linestyle=cfg.get('linestyle', '-'),
                marker='o',
                errorbar=errorbar,
            )

        plt.xlabel("number of training samples")
        plt.ylabel("KL divergence")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig(filepath)

    plot_configs = [
        {
            'pair': ('mcmc', 'mcmc_graph'), 
            'label': 'KL(  ,  )', 
            'color': 'black', 
            'linestyle': '-'
        },
        {
            'pair': (models[0]['name'], 'mcmc'), 
            'label': "KL(  ,  )", 
            'color': 'red', 
            'linestyle': '-'
        },
        {
            'pair': (models[1]['name'], 'mcmc_graph'), 
            'label': 'KL(  ,  )', 
            'color': 'orange', 
            'linestyle': '-'
        },
        {
            'pair': (models[0]['name'], models[1]['name']), 
            'label': 'KL(  ,  )', 
            'color': 'black', 
            'linestyle': '--'
        }
    ]

    # all four KL divergences
    make_kl_plot(plot_configs, f"{output_dir}/combined_kl_divergence.png", errorbar='ci')
    make_kl_plot(plot_configs, f"{output_dir}/combined_kl_divergence_no_error_bars.png", errorbar=None) # type: ignore
    # only the ones that go to 0
    make_kl_plot([plot_configs[0], plot_configs[3]], f"{output_dir}/kl_divergence_partial.png", errorbar='ci')
    make_kl_plot([plot_configs[0], plot_configs[3]], f"{output_dir}/kl_divergence_partial_no_error_bars.png", errorbar=None) # type: ignore


def improvements():
    from configs.default_configs import prior_config, training_config
    seed = 42
    
    num_steps = 20000
    
    buckets = training_config['buckets']
    model_paths_binary = {'baseline': 'workdir/baseline_binary', "attention": "workdir/attention_binary"}
    model_paths_beta = {'baseline': 'workdir/baseline_beta', 'attention': 'workdir/attention_beta'}
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    col_labels = ['num_nodes', 'edge_prob', 'root_std', 'non_root_std', 'number_train_samples_per_dataset']
    col_label_names = ['number of nodes in DAG', 'edge probability in DAG', 'mean noise σ at root nodes', 'mean noise σ at non-root nodes', 'number of training samples']
    
    prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": ["beta"], "probabilities": [1.0]}}
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    models = {name: init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth").to(get_default_device()) for name, path in model_paths_beta.items()}
    beta_df = compare_all(prior, models, buckets, output_dir, metric='r2')
    for name in models.keys():
        beta_df[f'{name}_improvement'] = beta_df[name] - beta_df['baseline']
    
    absolute_beta_df = beta_df.melt(
        id_vars=col_labels, 
        value_vars=[name for name in models.keys()], 
        var_name='model', 
        value_name='score')
    improvement_beta_df = beta_df.melt(
        id_vars=col_labels, 
        value_vars=[f'{name}_improvement' for name in models.keys() if name != 'baseline'], 
        var_name='model', 
        value_name='score')
    
    prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": ["binary"], "probabilities": [1.0]}}
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    models = {name: init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth").to(get_default_device()) for name, path in model_paths_binary.items()}
    binary_df = compare_all(prior, models, buckets, output_dir, metric='r2')
    for name in models.keys():
        binary_df[f'{name}_improvement'] = binary_df[name] - binary_df['baseline']
        
    absolute_binary_df = binary_df.melt(
        id_vars=col_labels, 
        value_vars=[name for name in models.keys()], 
        var_name='model', 
        value_name='score')
    improvement_binary_df = binary_df.melt(
        id_vars=col_labels, 
        value_vars=[f'{name}_improvement' for name in models.keys() if name != 'baseline'], 
        var_name='model', 
        value_name='score')

    # plotting the absolute scores for the beta models
    for label, label_name in zip(col_labels, col_label_names):
        absolute_beta_df[f'{label}_quantile'] = pd.qcut(absolute_beta_df[label], q=10, precision=1)
        absolute_beta_df[f'{label}_center'] = absolute_beta_df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=absolute_beta_df, x=f"{label}_center", y='score', hue='model', palette=['red', 'orange'],errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R²")   
        plt.savefig(f"{output_dir}/{label}_absolute_beta.png")
        plt.clf()
    # plotting the improvements for the beta models
    for label, label_name in zip(col_labels, col_label_names):
        improvement_beta_df[f'{label}_quantile'] = pd.qcut(improvement_beta_df[label], q=10, precision=1)
        improvement_beta_df[f'{label}_center'] = improvement_beta_df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=improvement_beta_df, x=f"{label}_center", y='score', hue='model', palette=['orange'], errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R² improvement over baseline")   
        plt.savefig(f"{output_dir}/{label}_improvement_beta.png")
        plt.clf()
    # plotting the absolute scores for the binary models
    for label, label_name in zip(col_labels, col_label_names):
        absolute_binary_df[f'{label}_quantile'] = pd.qcut(absolute_binary_df[label], q=10, precision=1)
        absolute_binary_df[f'{label}_center'] = absolute_binary_df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=absolute_binary_df, x=f"{label}_center", y='score', hue='model', palette=['red', 'violet'], errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R²")   
        plt.savefig(f"{output_dir}/{label}_absolute_binary.png")
        plt.clf()
    # plotting the improvements for the binary models
    for label, label_name in zip(col_labels, col_label_names):
        improvement_binary_df[f'{label}_quantile'] = pd.qcut(improvement_binary_df[label], q=10, precision=1)
        improvement_binary_df[f'{label}_center'] = improvement_binary_df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=improvement_binary_df, x=f"{label}_center", y='score', hue='model', palette=['violet'], errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R² improvement over baseline")   
        plt.savefig(f"{output_dir}/{label}_improvement_binary.png")
        plt.clf()
    # plotting the improvements for beta and binary models together
    improvement_binary_df['model'] = improvement_binary_df['model'].apply(lambda x: x.replace('_improvement', '') + '_binary')
    improvement_beta_df['model'] = improvement_beta_df['model'].apply(lambda x: x.replace('_improvement', '') + '_beta')
    combined_improvement_df = pd.concat([improvement_binary_df, improvement_beta_df])
    for label, label_name in zip(col_labels, col_label_names):
        combined_improvement_df[f'{label}_quantile'] = pd.qcut(combined_improvement_df[label], q=10, precision=1)
        combined_improvement_df[f'{label}_center'] = combined_improvement_df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=combined_improvement_df, x=f"{label}_center", y='score', hue='model', palette=['violet', 'orange'], errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R² improvement over baseline")   
        plt.savefig(f"{output_dir}/{label}_improvement_combined.png")
        plt.clf()
    
    
def swapped_input_mode():
    from configs.default_configs import prior_config, training_config
    seed = 42
    
    num_steps = 3000
    buckets = training_config['buckets']
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    modes = ['binary', 'beta', 'uncertain']
    model_paths = {'baseline_binary': 'workdir/baseline_binary', 
                       'attention_binary': 'workdir/attention_binary', 
                       'baseline_beta': 'workdir/baseline_beta', 
                       'attention_beta': 'workdir/attention_beta',
                       'baseline_uncertain': 'workdir/baseline_uncertain',
                       'attention_uncertain': 'workdir/attention_uncertain',}
    models = {name: init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth").to(get_default_device()) for name, path in model_paths.items()}
    results = {}
    for mode in modes:
        prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": [mode], "probabilities": [1.0]}}
        prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
        df = compare_all(prior, models, buckets, output_dir, metric='r2')
        for name in models.keys():
            baseline_name = 'baseline_' + name.split('_')[1]
            df[f'{name}_improvement'] = df[name] - df[baseline_name]
        df['input_mode'] = mode
        results[mode] = df
        
    df = pd.concat([results[mode] for mode in modes])
    df = df.melt(id_vars='input_mode', 
                 value_vars=[f'{name}_improvement' for name in models.keys() if not name.startswith('baseline')], 
                 var_name='model', 
                 value_name='score')
    
    sns.barplot(data=df, x='model', y='score', hue='input_mode', palette=['violet', 'orange', 'yellow'], errorbar='ci')
    plt.ylabel("R² improvement over baseline")   
    plt.savefig(f"{output_dir}/input_mode_comparison.png")
    plt.clf()
        

def add_edges():
    from configs.default_configs import prior_config, training_config
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    buckets = training_config['buckets']
    
    low, high = 0.0, 1.0
    num_steps = 3000
    prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                "distribution_parameters": {"choices": ["binary"], "probabilities": [1.0]}}
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    
    model_paths = {'baseline_binary': 'workdir/baseline_binary', 
                   "attention_binary": "workdir/attention_binary",
                   "baseline_beta": "workdir/baseline_beta",
                   "attention_beta": "workdir/attention_beta",
                   "baseline_uncertain": "workdir/baseline_uncertain",
                   "attention_uncertain": "workdir/attention_uncertain",
                   }
    models = {name: init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth").to(get_default_device()) for name, path in model_paths.items()}
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    def perturbate(graph_info):
        prob_adj = graph_info['prob_adj']
        p = low + (high - low) * torch.rand(1, generator=generator).item()
        probs = torch.full(prob_adj.shape, float(p), device=prob_adj.device)
        swap_mask = torch.bernoulli(probs, generator=generator).to(prob_adj.dtype)
        ones = 1 - torch.eye(prob_adj.shape[0], device=prob_adj.device)
        perturbed_prob_adj = torch.where(swap_mask.bool(), 0.25 * ones, prob_adj)
        dist = torch.sum(swap_mask).item() / swap_mask.numel()
        return {'prob_adj': perturbed_prob_adj}, dist
    
    df = compare_all(prior, models, buckets, output_dir, graph_info_trafo=perturbate)
    for name in models.keys():
        baseline_name = 'baseline_' + name.split('_')[1]
        df[f'{name}_improvement'] = df[name] - df[baseline_name]
    df = df.melt(id_vars='graph_info_score', 
                 value_vars=[f'{name}_improvement' for name in models.keys() if not name.startswith('baseline')], 
                 var_name='model', 
                 value_name='score')
    
    df['quantile'] = pd.qcut(df['graph_info_score'], q=10, precision=1)
    df['center'] = df['quantile'].apply(lambda x: x.mid).astype(float)
    sns.lineplot(data=df, x="center", y='score', hue='model', palette=['violet', 'orange', 'yellow'], errorbar='ci')
    plt.ylabel("R²")   
    plt.savefig(f"{output_dir}/perturbed_input.png")
    plt.clf()
    

def nll_visualization():
    from configs.default_configs import prior_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior = ObservationalDataLoader(10, 1, prior_config, seed=seed)
    
    models = [{"name": "baseline_beta", "color": "red", "label": "p(y|x, D) (CEL)"},
              {"name": "baseline_nll_beta", "color": "blue", "label": "p(y|x, D) (NLL)"}]
    
    for i in range(50):
        num_train_samples = 20
        mcmc_suite(prior, 
                   num_train_samples, 
                   generator, 
                   f"{output_dir}/run_{i}", 
                   include_mcmc=False, 
                   include_entropy=False, 
                   include_pfns=models)
        

def nll_comparison():
    from configs.default_configs import prior_config, training_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    num_steps = 10000
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior = ObservationalDataLoader(num_steps, 1, prior_config, seed=seed)
    buckets = training_config['buckets']
    
    models = [{"name": "baseline_beta", "color": "red", "label": "p(y|x, D) (CEL)"},
              {"name": "baseline_nll_beta", "color": "blue", "label": "p(y|x, D) (NLL)"}]
    
    model_paths_beta = {'baseline': 'workdir/baseline_beta', 'nll': 'workdir/baseline_nll_beta'}
    models = {name: init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth").to(get_default_device()) for name, path in model_paths_beta.items()}
    
    output_dir = f"visualization/output/{datetime_str}"
    for metric in ["r2", "mse", "nll", "cel", "smoothness"]:
        df = compare_all(prior, models, buckets, output_dir, metric=metric)
        wins = (df.iloc[:, -1] > df.iloc[:, -2]).astype(int)
        winrate = wins.mean()

        # bootstrapped CI
        n_iterations = 1000
        bootstrapped_stats = []
        for _ in range(n_iterations):
            sample = wins.sample(frac=1, replace=True)
            bootstrapped_stats.append(sample.mean())
        ci_lower = np.percentile(bootstrapped_stats, 2.5)
        ci_upper = np.percentile(bootstrapped_stats, 97.5)
        
        print(f"Win Rate under {metric}: {winrate:.2%}")
        print(f"95% CI:   [{ci_lower:.4f}, {ci_upper:.4f}]")
                    

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ppd_plots", action="store_true")
    group.add_argument("--improvements", action="store_true")
    group.add_argument("--swapped_input_mode", action="store_true")
    group.add_argument("--add_edges", action="store_true")
    group.add_argument("--nll", action="store_true")
    group.add_argument("--nll_comparison", action="store_true")

    args = parser.parse_args()

    if args.ppd_plots:
        ppd_plots()
    elif args.improvements:
        improvements()
    elif args.swapped_input_mode:
        swapped_input_mode()
    elif args.add_edges:
        add_edges()
    elif args.nll:
        nll_visualization()
    elif args.nll_comparison:
        nll_comparison()
        
        
if __name__ == "__main__":
    main()