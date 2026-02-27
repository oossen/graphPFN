from typing import Dict
import pandas as pd
import torch
from tfmplayground.utils import get_default_device
from sklearn.metrics import r2_score, mean_squared_error


def compare_all(prior, models: Dict, buckets: torch.Tensor, filename: str, metric: str = "r2", graph_info_trafo=None):
    """
    Make plots of the performance of the given models against various dataset characteristics.
    If `baseline` is given, instead save the improvment over the baseline model.
    If `graph_info_trafo` is given, it must be a function that transforms the graph information before adding it to the data frame.
    
    Supported metrics: "r2", "mse", "nll", "cel", "smoothness"
    """
    rows = []
    for data in prior:
        # add sampled parameters to data frame
        sampled_params = data["graph_information"]["sampled_params"]
        row = {}
        for k, v in sampled_params.items():
            row[k] = v
        rows.append(row)
        # evaluate on model, select first and only batch
        graph_info = data['graph_information']
        if graph_info_trafo is not None:
            graph_info, graph_info_score = graph_info_trafo(graph_info)
            row['graph_info_score'] = graph_info_score
        for name, model in models.items():
            score = compute_score(model, buckets, data, metric, **graph_info)
            row[name] = score
    df = pd.DataFrame(rows)
    df.to_csv(f"{filename}/full_comparison.csv")
    return df


@torch.no_grad()
def compute_score(model, buckets, data, metric, **graph_info):
    device = get_default_device()
    buckets = buckets.to(device)
    bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0
    X = data['x'].to(device)
    y = data['y'].to(device)
    single_eval_pos = data['single_eval_pos']
    y_train = y[:, :single_eval_pos]
    y_target = y[:, single_eval_pos:]
    y_target = y_target.reshape((-1,))
    y_target_buckets = (torch.bucketize(y_target, buckets) - 1).clamp(0, buckets.size(0) - 2)
            
    test_data = {v: data['data'][v][:, single_eval_pos:] for v in data['data']}
    scm = data['graph_information']['scm']
    log_probs = scm.log_likelihood_batch(test_data, bucket_mids.unsqueeze(0))
    probs = torch.exp(log_probs).to(device)
    y_target_dist = probs / probs.sum(dim=-1, keepdim=True)
    y_target_dist = y_target_dist.view(-1, y_target_dist.shape[-1])
            
    logits = model((X, y_train), single_eval_pos=single_eval_pos, **graph_info)
    logits = logits.view(-1, logits.shape[-1])
    probs = torch.softmax(logits, dim=-1)
    y_pred = probs @ bucket_mids
            
    ce_loss = torch.nn.CrossEntropyLoss()
    if metric == "r2":
        score = r2_score(y_target.cpu().numpy(), y_pred.cpu().numpy())
    elif metric == "mse":
        score = mean_squared_error(y_target.cpu().numpy(), y_pred.cpu().numpy())
    elif metric == "nll":
        score = ce_loss(logits, y_target_buckets).item()
    elif metric == "cel":
        score = ce_loss(logits, y_target_dist).item()
    elif metric == "smoothness":
        score = torch.mean(torch.sum(torch.diff(probs) ** 2, dim=-1)).item()
    else:
        raise ValueError(f"Unsupported metric {metric}")
    return score