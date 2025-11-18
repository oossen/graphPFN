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
    

def evaluate_on_wrong_graphs(model, prior, filename: str): 
    scores = []
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
        model.fit(X_train, y_train)
        pred = model.predict(X_test, adjacency_matrix=modified_adj)
        scores.append(r2_score(y_test, pred))
        distances.append(hamming_distance(adj, modified_adj))
    
    plt.scatter(distances, scores, color='blue', marker='o')
    plt.xlabel("Hamming distance")
    plt.ylabel("R²")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
        

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--model", type=str, choices=["pfn", "attention", "additive"], required=True)
parser.add_argument("--steps", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    model = init_model_from_state_dict_file(args.model, f"{args.dir}/latest_checkpoint.pth")
    buckets = torch.load(f"{args.dir}/dist.pth")
    dist = FullSupportBarDistribution(buckets)
    reg = Regressor(model, dist, get_default_device())
    
    prior = ObservationalDataLoader(num_steps=args.steps, batch_size=1, prior_config=prior_config, seed=42)
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    os.makedirs(f"evaluation/output/{datetime_str}", exist_ok=True)
    evaluate_on_wrong_graphs(reg, prior, f"evaluation/output/{datetime_str}/wrong_graphs.png")