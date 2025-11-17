import pandas as pd
from sklearn.metrics import r2_score


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
        adjacency_matrix = data['graph_information']['adjacency_matrix']
        model.fit(X_train, y_train)
        pred = model.predict(X_test, adjacency_matrix=adjacency_matrix)
        flat["R2"] = r2_score(y_test, pred)
    return pd.DataFrame(rows)
        

import argparse
import graphpfn.attention_model, graphpfn.additive_encoding_model, tfmplayground.model
from graphpfn.interface import Regressor, init_model_from_state_dict_file
from configs.default_configs import prior_config
from priors.observational_dataloader import ObservationalDataLoader
from pfns.bar_distribution import FullSupportBarDistribution
import torch
from datetime import datetime
from tfmplayground.utils import get_default_device

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--model", type=str, choices=["pfn", "attention", "additive"], required=True)
parser.add_argument("--steps", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    model_dir = args.dir
    if args.model == "pfn":
        model_class = tfmplayground.model.NanoTabPFNModel
    elif args.model == "attention":
        model_class = graphpfn.attention_model.GraphPFNModel
    elif args.model == "additive":
        model_class = graphpfn.additive_encoding_model.GraphPFNModel
    model = init_model_from_state_dict_file(model_class, f"{model_dir}/latest_checkpoint.pth")
    buckets = torch.load(f"{model_dir}/dist.pth")
    dist = FullSupportBarDistribution(buckets)
    reg = Regressor(model, dist, get_default_device())
    
    prior = ObservationalDataLoader(num_steps=args.steps, batch_size=1, prior_config=prior_config, seed=42)
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    df = evaluate(reg, prior)
    print(df)