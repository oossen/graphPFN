import os
from typing import Dict
from datetime import datetime

from configs.probabilistic_configs import prior_config
from priors.basic_dataloader_probabilistic import ObservationalDataLoader
from visualization.plotting import plot_graph, plot_point_clouds


def plot_all(prior, output_dir: str, n: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    for i, data in enumerate(prior):
        X = data['x'][0]
        y = data['y'][0]
        single_eval_pos = data['single_eval_pos']
        graph_test = data['graph_information']['graph_test']
        plot_graph(graph_test, f"{output_dir}/graph_{n}_{i}_test.png")
        graph_train = data['graph_information']['graph_train']
        plot_graph(graph_train, f"{output_dir}/graph_{n}_{i}_train.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{n}_{i}.png", single_eval_pos=single_eval_pos, graph=graph_test)
            
        # write sampled parameters to file
        if 'sampled_params' in data['graph_information']:
            with open(f"{output_dir}/sampled_params_{n}_{i}.py", "w") as f:
                for name, param_dict in data['graph_information']["sampled_params"].items():
                    f.write(f"{name} = {repr(param_dict)}\n")

def make_all(prior_class,
             prior_config: Dict,
             output_dir: str,
             n_steps: int = 5):
    os.makedirs(output_dir, exist_ok=True)
    
    # save the configs we used
    with open(f"{output_dir}/configs.py", "w") as f:
        f.write(f"prior_config = {repr(prior_config)}\n")
    prior = prior_class(n_steps, 1, prior_config, 42)             
    # call the plotting two times to check if subsequent iterators are different
    plot_all(prior, output_dir, 0)
    plot_all(prior, output_dir, 1)
    

if __name__ == "__main__":
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    make_all(ObservationalDataLoader, prior_config, f"visualization/output/{datetime_str}")