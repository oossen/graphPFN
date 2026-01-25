import os
from typing import Dict

from priors.basic_dataloader_graph_prior import ObservationalDataLoader
from visualization.check_seeding import compare_dataloaders
from visualization.plotting import plot_prob_adj, plot_r2, plot_correlation, plot_graph, plot_point_clouds


def plot_all(prior, output_dir: str, n: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    for i, data in enumerate(prior):
        X = data['x'][0]
        y = data['y'][0]
        single_eval_pos = data['single_eval_pos']
        g = data['graph_information']['graph']
        scm = data['graph_information']['scm']
        plot_graph(g, f"{output_dir}/graph_{n}_{i}.png")
        if 'new_graph' in data['graph_information']:
            new_g = data['graph_information']['new_graph']
            plot_graph(new_g, f"{output_dir}/new_graph_{n}_{i}.png")
        if 'confounding_graph' in data['graph_information']:
            plot_graph(confounding_g, f"{output_dir}/confounding_graph_{n}_{i}.png")
            confounding_g = data['graph_information']['confounding_graph']
        plot_correlation(X, f"{output_dir}/correlation_{n}_{i}.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{n}_{i}.png", single_eval_pos=single_eval_pos, graph=g)
        if 'prob_adj' in data['graph_information']:
            plot_prob_adj(data['graph_information']['prob_adj'], f"{output_dir}/prob_adj_{n}_{i}.png")
        if 'prob_confounding_adj' in data['graph_information']:
            plot_prob_adj(data['graph_information']['prob_confounding_adj'], f"{output_dir}/prob_confounding_adj_{n}_{i}.png")
            
        # write sampled parameters to file
        if 'sampled_params' in data['graph_information']:
            with open(f"{output_dir}/sampled_params_{n}_{i}.py", "w") as f:
                for name, param_dict in data['graph_information']["sampled_params"].items():
                    f.write(f"{name} = {repr(param_dict)}\n")
        # write SCM mechanisms to file
        with open(f"{output_dir}/scm_{n}_{i}.py", "w") as f:
            f.write(f"mechanisms = {repr(scm.mechanisms)}")
    

def make_all(prior_class,
             prior_config: Dict,
             output_dir: str,
             n_steps: int = 5,
             check_seeding=False,
             include_r2=False):
    """
    Run the complete visualization suite for the prior specified by `config`.
    This must be a string specifying a path to a configuration file.
    """
    if check_seeding:
        prior_1 = prior_class(n_steps, 5, prior_config, 42)
        prior_2 = prior_class(n_steps, 5, prior_config, 42)
        compare_dataloaders(prior_1, prior_2)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # save the configs we used
    with open(f"{output_dir}/configs.py", "w") as f:
        f.write(f"prior_config = {repr(prior_config)}\n")
    
    if include_r2:
        big_prior = prior_class(10 * n_steps, 1, prior_config, 42)
        plot_r2(big_prior, f"{output_dir}/r2.png")
    
    prior = prior_class(n_steps, 1, prior_config, 42)             
    # call the plotting two times to check if subsequent iterators are different
    plot_all(prior, output_dir, 0)
    plot_all(prior, output_dir, 1)
    

if __name__ == "__main__":
    from configs.test_configs import prior_config
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    prior = ObservationalDataLoader(30, 1, 5, 1, 42)
    plot_all(prior, f"visualization/output/{datetime_str}")