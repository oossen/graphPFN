import argparse
from datetime import datetime
from functools import partial
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Literal

from visualization.mcmc import mcmc_suite
from priors.observational_dataloader import ObservationalDataLoader

from visualization.full_comparison import compare_all
from graphpfn.interface import Regressor, init_model_from_state_dict_file


def compute_entropies():
    from configs.simple_configs import prior_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior_config['graph_config']['num_nodes'] = {'value': 3}
    prior = ObservationalDataLoader(100, 1, prior_config, seed=seed)
    prior._make_statistics(steps=10000)
    
    mcmc_parameters = (1, 0, 1) # steps, burn-in, thinning, unused for this experiment
    n_iterations = 100
    num_train_samples_list = list(range(0, 10)) + list(range(10, 20, 2)) + list(range(20, 50, 5)) + list(range(50, 110, 10))
    for i in range(n_iterations):
        for num_train_samples in num_train_samples_list:
            mcmc_suite(prior, 
                       num_train_samples, 
                       generator, 
                       f"{output_dir}/run_{i}_{num_train_samples}", 
                       include_mcmc=False, 
                       include_pfn=False, 
                       mcmc_parameters=mcmc_parameters)
    # load df from csv file
    df = pd.read_csv(f"{output_dir}/entropies.csv")
    sns.lineplot(data=df, x='samples', y='entropy', errorbar='ci')
    plt.savefig(f"{output_dir}/entropies.png")
    plt.clf()
    
    
def r2_comparison():
    from configs.default_configs import prior_config, training_config
    seed = 42
    
    num_steps = 100
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    
    model_paths = {'baseline': 'workdir/baseline_100', "binary_attention": "workdir/binary_attention_02_15_01_28", "soft_attention": "workdir/soft_attention_02_15_01_30"}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    df = compare_all(prior, models, output_dir)
    
    col_labels = ['num_nodes', 'edge_prob', 'root_std', 'non_root_std', 'number_train_samples_per_dataset']
    col_label_names = ['number of nodes in DAG', 'edge probability in DAG', 'mean noise σ at root nodes', 'mean noise σ at non-root nodes', 'number of training samples']
    df = df.melt(
        id_vars=col_labels, 
        value_vars=[name for name in models.keys()], 
        var_name='model', 
        value_name='score')

    for label, label_name in zip(col_labels, col_label_names):
        df[f'{label}_quantile'] = pd.qcut(df[label], q=10, precision=1)
        df[f'{label}_center'] = df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=df, x=f"{label}_center", y='score', hue='model', errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R²")   
        plt.savefig(f"{output_dir}/{label}.png")
        plt.clf()
    

def ppd_plots():
    from configs.simple_configs import prior_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior_config['graph_config']['num_nodes'] = {'value': 3}
    prior = ObservationalDataLoader(1000, 1, prior_config, seed=seed)
    prior._make_statistics(steps=10000)
    
    mcmc_parameters = (10000, 1000, 2) # steps, burn-in, thinning
    train_sample_sizes = [2, 10, 20, 50]
    
    for num_train_samples in train_sample_sizes:
        for i in range(10):
            mcmc_suite(prior, num_train_samples, generator, f"{output_dir}/run_{num_train_samples}_{i}", include_mcmc=True, include_pfn=True, mcmc_parameters=mcmc_parameters)


def improvements():
    from configs.default_configs import prior_config, training_config
    seed = 42
    
    num_steps = 1000
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    
    model_paths = {'baseline': 'workdir/baseline_100', "binary_attention": "workdir/binary_attention_02_15_01_28", "soft_attention": "workdir/soft_attention_02_15_01_30"}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    df = compare_all(prior, models, output_dir, baseline='baseline')
    
    col_labels = ['num_nodes', 'edge_prob', 'root_std', 'non_root_std', 'number_train_samples_per_dataset']
    col_label_names = ['number of nodes in DAG', 'edge probability in DAG', 'mean noise σ at root nodes', 'mean noise σ at non-root nodes', 'number of training samples']
    df = df.melt(
        id_vars=col_labels, 
        value_vars=[name for name in models.keys() if name != 'baseline'], 
        var_name='model', 
        value_name='score')

    for label, label_name in zip(col_labels, col_label_names):
        df[f'{label}_quantile'] = pd.qcut(df[label], q=10, precision=1)
        df[f'{label}_center'] = df[f'{label}_quantile'].apply(lambda x: x.mid).astype(float)
        sns.lineplot(data=df, x=f"{label}_center", y='score', hue='model', errorbar='ci')
        plt.xlabel(label_name)
        plt.ylabel("R²")   
        plt.savefig(f"{output_dir}/{label}.png")
        plt.clf()
    
    
def swapped_input_mode():
    from configs.default_configs import prior_config, training_config
    seed = 42
    
    num_steps = 100
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    
    model_paths = {'baseline': 'workdir/baseline_100', "binary_attention": "workdir/binary_attention_02_15_01_28", "soft_attention": "workdir/soft_attention_02_15_01_30"}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    def all_binary(graph_info):
        return {'adj': graph_info['adj'], 'prob_adj': graph_info['adj']}, 0.0
    def all_probabilistic(graph_info):
        return {'adj': graph_info['prob_adj'], 'prob_adj': graph_info['prob_adj']}, 0.0
    
    df_binary = compare_all(prior, models, output_dir, baseline='baseline', graph_info_trafo=all_binary)
    df_probabilistic = compare_all(prior, models, output_dir, baseline='baseline', graph_info_trafo=all_probabilistic)
    df_binary['input_mode'] = 'binary'
    df_probabilistic['input_mode'] = 'probabilistic'
    df = pd.concat([df_binary, df_probabilistic])
    df = df.melt(id_vars='input_mode', value_vars=[name for name in models.keys() if name != 'baseline'], var_name='model', value_name='score')
    
    sns.barplot(data=df, x='model', y='score', hue='input_mode', errorbar='ci')
    plt.ylabel("R²")   
    plt.savefig(f"{output_dir}/input_mode_comparison.png")
    plt.clf()
        

def add_edges():
    from configs.default_configs import prior_config, training_config
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    low, high = 0.0, 1.0
    num_steps = 2000
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    
    binary_model_paths = {'baseline': 'workdir/baseline_100', "binary_attention": "workdir/binary_attention_02_15_01_28", "binary_attention_fallback": "workdir/binary_attention_fallback_02_15_01_28"}
    models = {}
    for name, path in binary_model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    def perturbate(graph_info, name: Literal['adj', 'prob_adj']):
        adj = graph_info[name]
        p = low + (high - low) * torch.rand(1, generator=generator).item()
        probs = torch.full(adj.shape, float(p), device=adj.device)
        swap_mask = torch.bernoulli(probs, generator=generator).to(adj.dtype)
        ones = 1 - torch.eye(adj.shape[0], device=adj.device)
        perturbed_adj = torch.where(swap_mask.bool(), ones, adj)
        dist = torch.mean(torch.abs(perturbed_adj - adj)).item()
        return {name: perturbed_adj}, dist
    
    df = compare_all(prior, models, output_dir, baseline='baseline', graph_info_trafo=partial(perturbate, name='adj'))
    df = df.melt(id_vars='graph_info_score', value_vars=[name for name in models.keys() if name != 'baseline'], var_name='model', value_name='score')
    df['quantile'] = pd.qcut(df['graph_info_score'], q=10, precision=1)
    df['center'] = df['quantile'].apply(lambda x: x.mid).astype(float)
    sns.lineplot(data=df, x="center", y='score', hue='model', errorbar='ci')
    plt.ylabel("R²")   
    plt.savefig(f"{output_dir}/perturbed_input_binary.png")
    plt.clf()
    
    probabilistic_model_paths = {'baseline': 'workdir/baseline_100', "probabilistic_attention": "workdir/soft_attention_02_15_01_30", "probabilistic_attention_fallback": "workdir/soft_attention_fallback_02_15_01_32"}
    models = {}
    for name, path in probabilistic_model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
            
    df = compare_all(prior, models, output_dir, baseline='baseline', graph_info_trafo=partial(perturbate, name='prob_adj'))
    df = df.melt(id_vars='graph_info_score', value_vars=[name for name in models.keys() if name != 'baseline'], var_name='model', value_name='score')
    df['quantile'] = pd.qcut(df['graph_info_score'], q=10, precision=1)
    df['center'] = df['quantile'].apply(lambda x: x.mid).astype(float)
    sns.lineplot(data=df, x="center", y='score', hue='model', errorbar='ci')
    plt.ylabel("R²")   
    plt.savefig(f"{output_dir}/perturbed_input_probabilistic.png")
    plt.clf()
    

def remove_edges():
    from configs.default_configs import prior_config, training_config
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    low, high = 0.0, 0.3
    num_steps = 20
    prior = ObservationalDataLoader(num_steps=num_steps, batch_size=1, prior_config=prior_config, seed=seed)
    
    binary_model_paths = {'baseline': 'workdir/baseline_100', "binary_attention": "workdir/binary_attention_02_15_01_28", "binary_attention_fallback": "workdir/binary_attention_fallback_02_15_01_28"}
    models = {}
    for name, path in binary_model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    def perturbate(graph_info, name: Literal['adj', 'prob_adj']):
        adj = graph_info[name]
        p = low + (high - low) * torch.rand(1, generator=generator).item()
        probs = torch.full(adj.shape, float(p), device=adj.device)
        swap_mask = torch.bernoulli(probs, generator=generator).to(adj.dtype)
        zeros = torch.zeros_like(adj)
        perturbed_adj = torch.where(swap_mask.bool(), zeros, adj)
        dist = torch.mean(torch.abs(perturbed_adj - adj)).item()
        return {name: perturbed_adj}, dist
    
    df = compare_all(prior, models, output_dir, baseline='baseline', graph_info_trafo=partial(perturbate, name='adj'))
    df = df.melt(id_vars='graph_info_score', value_vars=[name for name in models.keys() if name != 'baseline'], var_name='model', value_name='score')
    df['quantile'] = pd.qcut(df['graph_info_score'], q=10, precision=1, duplicates='drop')
    df['center'] = df['quantile'].apply(lambda x: x.mid).astype(float)
    sns.lineplot(data=df, x="center", y='score', hue='model', errorbar='ci')
    plt.ylabel("R²")   
    plt.savefig(f"{output_dir}/perturbed_input_binary.png")
    plt.clf()
    
    probabilistic_model_paths = {'baseline': 'workdir/baseline_100', "probabilistic_attention": "workdir/soft_attention_02_15_01_30", "probabilistic_attention_fallback": "workdir/soft_attention_fallback_02_15_01_32"}
    models = {}
    for name, path in probabilistic_model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
            
    df = compare_all(prior, models, output_dir, baseline='baseline', graph_info_trafo=partial(perturbate, name='prob_adj'))
    df = df.melt(id_vars='graph_info_score', value_vars=[name for name in models.keys() if name != 'baseline'], var_name='model', value_name='score')
    df['quantile'] = pd.qcut(df['graph_info_score'], q=10, precision=1, duplicates='drop')
    df['center'] = df['quantile'].apply(lambda x: x.mid).astype(float)
    sns.lineplot(data=df, x="center", y='score', hue='model', errorbar='ci')
    plt.ylabel("R²")   
    plt.savefig(f"{output_dir}/perturbed_input_probabilistic.png")
    plt.clf()
        

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--entropies", action="store_true")
    group.add_argument("--r2_comparison", action="store_true")
    group.add_argument("--ppd_plots", action="store_true")
    group.add_argument("--improvements", action="store_true")
    group.add_argument("--swapped_input_mode", action="store_true")
    group.add_argument("--add_edges", action="store_true")
    group.add_argument("--remove_edges", action="store_true")

    args = parser.parse_args()

    if args.entropies:
        compute_entropies()
    elif args.r2_comparison:
        r2_comparison()
    elif args.ppd_plots:
        ppd_plots()
    elif args.improvements:
        improvements()
    elif args.swapped_input_mode:
        swapped_input_mode()
    elif args.add_edges:
        add_edges()
    elif args.remove_edges:
        remove_edges()
        
        
if __name__ == "__main__":
    main()