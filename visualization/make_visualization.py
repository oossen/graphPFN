import os
from datetime import datetime

from configs.default_configs import prior_config
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
        probs = data['probs'][0][0]
        bucket_mids = prior.bucket_mids
        test_y = y[single_eval_pos][0].item()
        plot_graph(graph, f"{output_dir}/graph_{i}_test.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{i}.png", single_eval_pos=single_eval_pos, graph=graph)
        plot_correlation(X, f"{output_dir}/correlation_{i}.png")
        plot_adj(prob_adj, adj, f"{output_dir}/prob_adj_{i}.png")
        plot_likelihoods(bucket_mids, probs, test_y, f"{output_dir}/likelihoods_{i}.png")
        # write SCM to file
        with open(f"{output_dir}/scm_{i}.txt", "w") as f:
            f.write(str(data['graph_information']['scm']))
    
    
if __name__ == "__main__":
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    prior = ObservationalDataLoader(20, 1, prior_config, seed=42)
    plot_all(prior, f"visualization/output/{datetime_str}")