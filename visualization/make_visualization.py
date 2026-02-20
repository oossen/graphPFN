import os
from datetime import datetime

import torch

from configs.default_configs import prior_config, training_config
from priors.observational_dataloader import ObservationalDataLoader
from visualization.plotting import plot_graph, plot_point_clouds, plot_correlation, plot_adj, plot_likelihoods


def plot_all(prior, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, data in enumerate(prior):
        X = data['x'][0]
        y = data['y'][0]
        single_eval_pos = data['single_eval_pos']
        graph = data['graph_information']['graph']
        prob_adj = data['graph_information']['prob_adj']
        adj = data['graph_information']['adj']
        plot_graph(graph, f"{output_dir}/graph_{i}_test.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{i}.png", single_eval_pos=single_eval_pos, graph=graph)
        plot_correlation(X, f"{output_dir}/correlation_{i}.png")
        plot_adj(prob_adj, adj, f"{output_dir}/prob_adj_{i}.png")
        # write SCM to file
        with open(f"{output_dir}/scm_{i}.txt", "w") as f:
            f.write(str(data['graph_information']['scm']))
        # plot likelihoods
        buckets = training_config['buckets']
        bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0
        bucket_mids = bucket_mids.unsqueeze(0) # add batch dimension
        test_data = {v: data['data'][v][:, single_eval_pos:] for v in data['data']}
        scm = data['graph_information']['scm']
        log_probs = scm.log_likelihood_batch(test_data, bucket_mids)
        probs = torch.exp(log_probs)
        plot_likelihoods(bucket_mids[0], probs[0][0], data['y'][0][single_eval_pos].item(), f"{output_dir}/likelihoods_{i}.png")
    
    
if __name__ == "__main__":
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    prior = ObservationalDataLoader(20, 1, prior_config, seed=43)
    plot_all(prior, f"visualization/output/{datetime_str}")