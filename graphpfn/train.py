import torch
from torch import nn
import time
import schedulefree
import os
from pfns.bar_distribution import FullSupportBarDistribution
from tfmplayground.callbacks import Callback
from priors.observational_dataloader import ObservationalDataLoader
from graphpfn.base_model import GraphPFNModel
from tfmplayground.utils import get_default_device


def train(model: GraphPFNModel,
          prior: ObservationalDataLoader, 
          buckets: torch.Tensor,
          epochs: int,
          lr: float = 1e-4,
          nll: bool = False,
          callbacks: list[Callback] = [], 
          run_name: str = 'graphPFN'):
    """
    Trains our model on the given prior using the given criterion.

    Parameters
    ----------
    model : GraphPFNModel
        The model to train.
    prior: ObservationalDataLoader
        A dataloader providing training data in the necessary format.
    buckets: torch.Tensor
        The buckets to which the model's outputs will be fit.
    epochs : int
        The number of epochs to train for. One epoch consists of on iteration over `prior`.
    lr : float
        The learning rate.
    nll : bool
        Whether to train against just one class label per dataset and test sample (corresponding to the value of `y`).
        In other words, train using NLL instead of cross-entropy loss with class proabilities.
    callbacks : List[Callback]
        A list of callback instances to execute at the end of each epoch (e.g. logging, validation).
    run_name : str
        The name of this training run. Used to create a folder saving the trained model.
    """
    work_dir = 'workdir/'+run_name
    os.makedirs(work_dir, exist_ok=True)
    device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    bar_dist = FullSupportBarDistribution(buckets)

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data['single_eval_pos']
                data = (full_data['x'].to(device),
                        full_data['y'][:, :single_eval_pos].to(device))
                if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
                    continue
                
                output = model(data, single_eval_pos=single_eval_pos, **full_data['graph_information'])
                output = output.view(-1, output.shape[-1])
                
                if nll:
                    y_values = full_data['y'][:, single_eval_pos:].to(device)
                    y_values = y_values.reshape((-1,))
                    targets = torch.bucketize(y_values, buckets.to(device)) - 1
                else:
                    targets = full_data['probs'].to(device)
                    # renormalize targets from density values to discrete probabilities
                    targets = targets / targets.sum(dim=-1, keepdim=True)
                    targets = targets.view(-1, targets.shape[-1])

                losses = loss_fn(output, targets)
                loss = losses.mean()
                loss.backward()
                total_loss += loss.cpu().detach().item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            model.eval()
            optimizer.eval()

            training_state = {
                'epoch': epoch,
                'model_class': type(model),
                'architecture': model.architecture,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(training_state, work_dir+'/latest_checkpoint.pth')

            for callback in callbacks:
                callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model, dist=bar_dist)
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return model, total_loss