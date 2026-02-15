import argparse
from datetime import datetime
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from visualization.mcmc import mcmc_suite
from priors.observational_dataloader import ObservationalDataLoader

from visualization.full_comparison import compare_all
from graphpfn.interface import Regressor, init_model_from_state_dict_file


def compute_entropies():
    from configs.simple_configs import prior_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 100
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior_config['graph_config']['num_nodes'] = {'value': 3}
    prior = ObservationalDataLoader(100, 1, prior_config, seed=seed)
    prior._make_statistics(steps=10000)
    
    mcmc_parameters = (1, 0, 1) # steps, burn-in, thinning
    n_iterations = 2
    min_samples, max_samples = 2, 10
    for i in range(n_iterations):
        for num_train_samples in range(min_samples, max_samples + 1):
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
    from configs.default_configs import training_config
    
    steps = 100
    
    model_paths = {'pfn': 'workdir/baseline_02_15_01_26', "binary_attention": "workdir/binary_attention_02_15_01_28"}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"visualization/output/{datetime_str}", exist_ok=True)
    compare_all(models, steps, f"visualization/output/{datetime_str}/full_comparison")
    

def ppd_plots():
    from configs.simple_configs import prior_config
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"visualization/output/{datetime_str}"
    
    seed = 100
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    prior_config['graph_config']['num_nodes'] = {'value': 3}
    prior = ObservationalDataLoader(100, 1, prior_config, seed=seed)
    prior._make_statistics(steps=10000)
    
    mcmc_parameters = (10000, 1000, 2) # steps, burn-in, thinning
    train_sample_sizes = [2, 10, 20, 50]
    
    for num_train_samples in train_sample_sizes:
        for i in range(10):
            mcmc_suite(prior, num_train_samples, generator, f"{output_dir}/run_{num_train_samples}_{i}", include_mcmc=True, include_pfn=True, mcmc_parameters=mcmc_parameters)


def improvements():
    from configs.default_configs import training_config
    
    steps = 100
    
    model_paths = {'baseline': 'workdir/baseline_02_15_01_26', "binary_attention": "workdir/binary_attention_02_15_01_28"}
    models = {}
    for name, path in model_paths.items():
        model = init_model_from_state_dict_file(f"{path}/latest_checkpoint.pth")
        buckets = training_config['buckets']
        reg = Regressor(model, buckets)
        models[name] = reg
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"visualization/output/{datetime_str}", exist_ok=True)
    compare_all(models, steps, f"visualization/output/{datetime_str}/full_comparison", baseline='baseline')
    

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--entropies", action="store_true")
    group.add_argument("--r2_comparison", action="store_true")
    group.add_argument("--ppd_plots", action="store_true")
    group.add_argument("--improvements", action="store_true")

    args = parser.parse_args()

    if args.entropies:
        compute_entropies()
    elif args.r2_comparison:
        r2_comparison()
    elif args.ppd_plots:
        ppd_plots()
    elif args.improvements:
        improvements()
if __name__ == "__main__":
    main()