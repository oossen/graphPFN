from matplotlib import pyplot as plt
from datetime import datetime
import os
from sklearn.metrics import r2_score
import argparse
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from configs.default_configs import prior_config
from priors.observational_dataloader import ObservationalDataLoader
from pfns.bar_distribution import FullSupportBarDistribution
import torch
from tfmplayground.utils import get_default_device
import numpy as np


def remove_outliers(x):
    q1 = np.percentile(x, 10)
    q3 = np.percentile(x, 90)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (x >= lower) & (x <= upper)


def hamming_distance(adj_1: torch.Tensor, adj_2: torch.Tensor):
    """
    Compute the Hamming distance between the graphs represented by adj_1 and adj_2.
    This is the proportion of pairs of nodes {v, w} such that one of the following holds:
    1) v and w are connected by an edge in graph 1, but not in graph 2.
    2) v and w are connected by an edge in graph 2, but not in graph 1.
    Note that we do not take the orientation of edges into account.
    """
    adj_1 = (adj_1.bool() | adj_1.T.bool())
    adj_2 = (adj_2.bool() | adj_2.T.bool())
    diff = adj_1 != adj_2
    lower_diff = torch.tril(diff, diagonal=-1)
    total_positions = adj_1.shape[0] * (adj_1.shape[0]) / 2
    dist = lower_diff.sum().item()  / total_positions
    return dist


def missing_edges_distance(adj_1: torch.Tensor, adj_2: torch.Tensor):
    """
    Return the proportion of pairs of nodes {v, w} such that an edge between v and w
    is present in graph 1, but not in graph 2.
    """
    adj_1 = (adj_1.bool() | adj_1.T.bool())
    adj_2 = (adj_2.bool() | adj_2.T.bool())
    diff = adj_1 & ~adj_2
    lower_diff = torch.tril(diff, diagonal=-1)
    total_positions = adj_1.shape[0] * (adj_1.shape[0]) / 2
    dist = lower_diff.sum().item()  / total_positions
    return dist
    
    
def superfluous_edges_distance(adj_1: torch.Tensor, adj_2: torch.Tensor):
    """
    Return the proportion of pairs of nodes {v, w} such that an edge between v and w
    is present in graph 2, but not in graph 1.
    """
    adj_1 = (adj_1.bool() | adj_1.T.bool())
    adj_2 = (adj_2.bool() | adj_2.T.bool())
    diff = ~adj_1 & adj_2
    lower_diff = torch.tril(diff, diagonal=-1)
    total_positions = adj_1.shape[0] * (adj_1.shape[0]) / 2
    dist = lower_diff.sum().item()  / total_positions
    return dist
    

def wrong_orientation_distance(adj_1: torch.Tensor, adj_2: torch.Tensor):
    """
    Return the proportion of pairs of nodes {v, w} such that an edge between v and w
    is present in both graph 1 and graph 2, but with opposite orientations.
    (Or else graph 2 is misspecified by having edges in both directions.)
    """
    adj_1 = adj_1.bool()
    adj_2 = adj_2.bool()
    diff = adj_1 == adj_2.T
    lower_diff = torch.tril(diff, diagonal=-1)
    total_positions = adj_1.shape[0] * (adj_1.shape[0]) / 2
    dist = lower_diff.sum().item()  / total_positions
    return dist
    

def evaluate_on_wrong_graphs(model_1, model_2, prior, filename: str): 
    diffs = []
    distances = []
    for data in prior:
        adj = data['graph_information']['adjacency_matrix']
        # make modified adjacency matrix
        p = torch.rand(1).item() * 0.5
        mask = torch.rand(adj.shape) < p
        modified_adj = (adj != mask).int()
        # evaluate on model
        X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
        y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
        X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
        y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
        model_1.fit(X_train, y_train)
        pred_1 = model_1.predict(X_test, adjacency_matrix=modified_adj)
        score_1 = r2_score(y_test, pred_1)
        model_2.fit(X_train, y_train)
        pred_2 = model_2.predict(X_test, adjacency_matrix=modified_adj)
        score_2 = r2_score(y_test, pred_2)
        diff = score_2 - score_1
        diffs.append(diff)
        distances.append(hamming_distance(adj, modified_adj))
        
    # remove outliers    
    diffs = np.array(diffs)
    distances = np.array(distances)
    outlier_mask = remove_outliers(diffs)
    diffs = diffs[outlier_mask]
    distances = distances[outlier_mask]
    
    # bucketing
    n_buckets = 20
    bins = np.linspace(distances.min(), distances.max(), n_buckets + 1)
    bucket_ids = np.digitize(distances, bins) - 1

    bucket_centers = []
    bucket_means = []
    bucket_stds = []

    for b in range(n_buckets):
        mask = bucket_ids == b
        if mask.any():
            bucket_centers.append((bins[b] + bins[b+1]) / 2)
            bucket_means.append(diffs[mask].mean())
            bucket_stds.append(diffs[mask].std())
            
    centers_arr = np.array(bucket_centers)
    means_arr = np.array(bucket_means)
    stds_arr = np.array(bucket_stds)
    upper_bound = means_arr + stds_arr
    lower_bound = means_arr - stds_arr

    plt.plot(centers_arr, means_arr, marker='o', color='red')
    plt.fill_between(centers_arr, lower_bound, upper_bound, alpha=0.3, color='red')
    plt.xlabel("Hamming distance")
    plt.ylabel("R²")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
        

parser = argparse.ArgumentParser()
parser.add_argument("--dir_1", type=str, required=True)
parser.add_argument("--model_1", type=str, required=True)
parser.add_argument("--dir_2", type=str, required=True)
parser.add_argument("--model_2", type=str, required=True)
parser.add_argument("--steps", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    model_1 = init_model_from_state_dict_file(args.model_1, f"{args.dir_1}/latest_checkpoint.pth")
    model_2 = init_model_from_state_dict_file(args.model_2, f"{args.dir_2}/latest_checkpoint.pth")
    buckets_1 = torch.load(f"{args.dir_1}/dist.pth")
    buckets_2 = torch.load(f"{args.dir_2}/dist.pth")
    dist_1 = FullSupportBarDistribution(buckets_1)
    dist_2 = FullSupportBarDistribution(buckets_2)
    reg_1 = Regressor(model_1, dist_1, get_default_device())
    reg_2 = Regressor(model_2, dist_2, get_default_device())
    prior = ObservationalDataLoader(num_steps=args.steps, batch_size=1, prior_config=prior_config, seed=42)

    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"evaluation/output/{datetime_str}", exist_ok=True)
    evaluate_on_wrong_graphs(reg_1, reg_2, prior, f"evaluation/output/{datetime_str}/wrong_graphs.png")