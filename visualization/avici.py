import os
import avici
import torch

from priors.observational_dataloader_graph_prior import ObservationalDataLoader
from visualization.plotting import plot_prob_adj


def compare_adjacency_matrices(prior, output_dir: str):
    """
    Compare the ground truth probabilistic adjacency matrix
    with the one inferred by the AVICI causal discovery model.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, data in enumerate(prior):
        X = data['x'][0]
        y = data['y'][0]
        all_data = torch.concat([X, y], dim=-1)
        prob_adj = data['graph_information']['prob_adj']
        adj = data['graph_information']['adjacency_matrix']
        avici_model = avici.load_pretrained(download="scm-v0")
        prob_adj_avici = avici_model(x=all_data.detach().cpu().numpy())
        plot_prob_adj([prob_adj, adj, prob_adj_avici], f"{output_dir}/prob_adj_{i}.png")
    

if __name__ == "__main__":
    from datetime import datetime
    from configs.default_graph_prior_configs import prior_config
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    prior = ObservationalDataLoader(5, 1, prior_config, seed=42)
    compare_adjacency_matrices(prior, f"visualization/output/{datetime_str}")