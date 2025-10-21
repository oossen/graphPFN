import os
from typing import Dict

from prior.dataloaders.constant_dataloader import ConstantDataLoader
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from visualization.check_seeding import compare_dataloaders
from visualization.plotting import plot_r2, plot_correlation, plot_graph, plot_point_clouds


def make_all(prior_class,
             prior_config: Dict,
             output_dir: str,
             n_steps: int = 5,
             check_seeding=True,
             include_r2=True):
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
    def plot_all_(prior, n: int):
        for i, data in enumerate(prior):
            X = data['x'][0]
            y = data['y'][0]
            g = data['graph']
            plot_graph(g, f"{output_dir}/graph_{n}_{i}.png")
            plot_correlation(X, f"{output_dir}/correlation_{n}_{i}.png")
            plot_point_clouds(X, y, f"{output_dir}/point_clouds_{n}_{i}.png")
            
            # write sampled parameters to file
            with open(f"{output_dir}/sampled_params_{n}_{i}.py", "w") as f:
                for name, param_dict in data["sampled_params"].items():
                    f.write(f"{name} = {repr(param_dict)}\n")
                    
    # call the plotting two times to check if subsequent iterators are different
    plot_all_(prior, 1)
    plot_all_(prior, 2)
    

if __name__ == "__main__":
    from configs.default_configs import prior_config
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    make_all(ConstantDataLoader, prior_config, f"visualization/output/{datetime_str}")