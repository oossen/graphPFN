import os
import pandas as pd
from datetime import datetime
import torch

from tfmplayground.utils import get_default_device


@torch.no_grad()
def evaluate_on_prior(models, prior, metrics, path):
    device = get_default_device()
    for model in models.values():
        model.to(device)
        model.eval()

    # Determine the starting dataset ID if the file already exists
    start_id = 0
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        if not existing_df.empty:
            start_id = existing_df['ds_id'].max() + 1

    results = []

    for local_idx, data in enumerate(prior):
        # Calculate global unique ID
        ds_id = start_id + local_idx
        
        # Prepare data
        single_eval_pos = data['single_eval_pos']
        X = data['x'].to(device)
        y = data['y'].to(device)
        
        y_train = y[:, :single_eval_pos]
        y_target = y[:, single_eval_pos:]
        y_target = y_target.reshape((-1,))
        graph_info = data['graph_information']
        sampled_params = graph_info.get("sampled_params", {})

        for model_name, model in models.items():
            logits = model(
                (X, y_train), 
                single_eval_pos=single_eval_pos, 
                **graph_info
            )
            logits = logits.reshape(-1, logits.shape[-1])

            for metric_name, metric_fn in metrics.items():
                value = metric_fn(logits, y_target)
                
                row = {
                    "ds_id": ds_id,
                    "model": model_name,
                    "metric": metric_name,
                    "value": value,
                    **sampled_params
                }
                results.append(row)

        # Periodically flush to disk to manage memory and prevent data loss
        if (local_idx + 1) % 10 == 0:
            _append_to_csv(results, path)
            results = []
            
    if results:
        _append_to_csv(results, path)

def _append_to_csv(results, path):
    df = pd.DataFrame(results)
    file_exists = os.path.isfile(path)
    # Write header only if the file is being created for the first time
    df.to_csv(path, mode='a', index=False, header=not file_exists)
    

def metric_r2(logits, y_target):
    from sklearn.metrics import r2_score
    from configs.default_configs import training_config
    device = get_default_device()
    buckets = training_config['buckets'].to(device)
    bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0
    probs = torch.softmax(logits, dim=-1)
    y_pred = probs @ bucket_mids
    return r2_score(y_target.cpu().numpy(), y_pred.cpu().numpy())


if __name__ == "__main__":
    from configs.default_configs import prior_config
    from graphpfn.interface import init_model_from_state_dict_file
    from priors.observational_dataloader import ObservationalDataLoader
    model_names = ["attention_beta", "gcn_beta_04_09_20_22", "baseline_beta"]
    model_paths = {name: f"workdir/{name}/latest_checkpoint.pth" for name in model_names}
    models = {name: init_model_from_state_dict_file(path) for name, path in model_paths.items()}
    metrics = {"r2": metric_r2}
    prior = ObservationalDataLoader(100, 1, prior_config, seed=42)
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"icml_plots/output/{datetime_str}"
    os.makedirs(output_dir, exist_ok=True)
    evaluate_on_prior(models, prior, metrics, f"{output_dir}/results.csv")
    
    
    