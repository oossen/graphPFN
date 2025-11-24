import pandas as pd
from sklearn.metrics import r2_score
import argparse
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from configs.default_configs import prior_config
from priors.observational_dataloader import ObservationalDataLoader
from pfns.bar_distribution import FullSupportBarDistribution
import torch
from tfmplayground.utils import get_default_device
import networkx as nx


def evaluate(model, prior): 
    rows = []
    for data in prior:
        # add sampled parameters to data frame
        sampled_params = data["graph_information"]["sampled_params"]
        flat = {}
        for _, inner_dict in sampled_params.items():
            for k, v in inner_dict.items():
                flat[k] = v
        rows.append(flat)
        # evaluate on model
        X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
        y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
        X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
        y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
        model.fit(X_train, y_train)
        pred = model.predict(X_test, **data['graph_information'])
        flat["R2"] = r2_score(y_test, pred)
    return pd.DataFrame(rows)


def evaluate_on_markov_blanket(model, prior):
    rows = []
    for data in prior:
        # add sampled parameters to data frame
        sampled_params = data["graph_information"]["sampled_params"]
        flat = {}
        for _, inner_dict in sampled_params.items():
            for k, v in inner_dict.items():
                flat[k] = v
        rows.append(flat)
        # find features in Markov blanket
        g: nx.DiGraph = data["graph_information"]["new_graph"]
        parents = list(g.predecessors('y'))
        children = list(g.successors('y'))
        coparents = [v for w in children for v in g.predecessors(w)]
        blanket = set(parents + children + coparents)
        blanket.discard('y')
        blanket_indices = [int(v[1]) for v in blanket] # blanket contains strings of the form 'xi', so v[1]=i
        flat["blanket_size"] = len(blanket_indices)
        # evaluate on model
        X_train = data['x'][0, :data['single_eval_pos'], blanket_indices].cpu().numpy()
        y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
        X_test = data['x'][0, data['single_eval_pos']:, blanket_indices].cpu().numpy()
        y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
        model.fit(X_train, y_train)
        pred = model.predict(X_test, **data['graph_information'])
        flat["R2"] = r2_score(y_test, pred)
    return pd.DataFrame(rows)
        
        

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
    df = evaluate(reg, prior)
    print(df)