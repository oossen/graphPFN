import pandas as pd
from sklearn.metrics import r2_score
import argparse
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from configs.default_configs import prior_config
from priors.observational_dataloader_graph_prior import ObservationalDataLoader
from pfns.bar_distribution import FullSupportBarDistribution
import torch
from tfmplayground.utils import get_default_device
import avici
import numpy as np


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


def evaluate_avici(model, prior):
    """Use AVICI to infer probabilistic adjacency matrix from data."""
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
        all_data = np.concatenate([X_train, y_train], axis=-1)
        avici_model = avici.load_pretrained(download="scm-v0")
        prob_adj_avici = avici_model(x=all_data)
        model.fit(X_train, y_train)
        pred = model.predict(X_test, prob_adj=prob_adj_avici)
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