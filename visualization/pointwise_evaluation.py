import os
from typing import Dict
from sklearn.metrics import r2_score

from nanotabpfn.utils import get_default_device

from graphpfn.interface import Regressor
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from visualization.plotting import plot_correlation, plot_graph, plot_point_clouds, plot_scores


def evaluate(prior,
             model,
             dist,
             output_dir: str,
             n_steps: int = 50,
             ):
    """
    Evaluate `model` on data sampled from the specified prior.
    The result is reported in detail for each individual data point.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prior = ObservationalDataLoader(num_steps=n_steps,
                                batch_size=1,
                                prior_config=prior.prior_config,
                                seed=42)
    regressor = Regressor(model, dist, get_default_device())
    scores = []
        
    for i, data in enumerate(prior):
        X = data['x'][0]
        y = data['y'][0]
        g = data['graph']
        new_g = data['new_graph']
        scm = data['scm']
        plot_graph(g, f"{output_dir}/graph_{i}.png")
        plot_graph(new_g, f"{output_dir}/new_graph_{i}.png")
        plot_correlation(X, f"{output_dir}/correlation_{i}.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{i}.png")
            
        # write sampled parameters to file
        with open(f"{output_dir}/sampled_params_{i}.py", "w") as f:
            for name, param_dict in data["sampled_params"].items():
                f.write(f"{name} = {repr(param_dict)}\n")
        # write SCM mechanisms to file
        with open(f"{output_dir}/scm_{i}.py", "w") as f:
            f.write(f"mechanisms = {repr(scm.mechanisms)}")
            
        # evaluate on model
        X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
        y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
        X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
        y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
        adjacency_matrix = data['adjacency_matrix']
            
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_test, adjacency_matrix=adjacency_matrix)
        scores.append(r2_score(y_test, pred))
    
    plot_scores(scores, f"{output_dir}/scores.png")