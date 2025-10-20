from datetime import datetime
import os
import shutil
from typing import Dict

from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from visualization.check_seeding import compare_dataloaders
from visualization.plotting import plot_r2, plot_correlation, plot_graph, plot_point_clouds


def make_all(prior_config: Dict,
             output_dir: str,
             n_steps: int = 10,
             check_seeding=True,
             include_r2=True):
    """
    Run the complete visualization suite for the prior specified by `config`.
    This must be a string specifying a path to a configuration file.
    """
    if check_seeding:
        prior_1 = ObservationalDataLoader(n_steps, 5, prior_config, 42)
        prior_2 = ObservationalDataLoader(n_steps, 5, prior_config, 42)
        compare_dataloaders(prior_1, prior_2)
    
    
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_path = f"{output_dir}/{datetime_str}"
    os.makedirs(output_path, exist_ok=True)
    
    # save the configs we used
    with open(f"{output_path}/configs.py", "w") as f:
        f.write(f"prior_config = {repr(prior_config)}\n")
    
    if include_r2:
        big_prior = ObservationalDataLoader(10 * n_steps, 1, prior_config, 42)
        plot_r2(big_prior, f"{output_path}/r2.png")
    
    prior = ObservationalDataLoader(n_steps, 1, prior_config, 42)
    for i, data in enumerate(prior):
        X = data['x'][0]
        y = data['y'][0]
        g = data['graph']
        plot_graph(g, f"{output_path}/graph_{i}.png")
        plot_correlation(X, f"{output_path}/correlation_{i}.png")
        plot_point_clouds(X, y, f"{output_path}/point_clouds_{i}.png")
        
        # write sampled parameters to file
        with open(f"{output_path}/sampled_params_{i}.py", "w") as f:
            for name, param_dict in data["sampled_params"].items():
                f.write(f"{name} = {repr(param_dict)}\n")
    

if __name__ == "__main__":
    from configs.default_configs import prior_config
    make_all(prior_config, "visualization/output")