import torch
from torch import nn
import time
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pfns.bar_distribution import FullSupportBarDistribution
import schedulefree

from nanotabpfn.callbacks import Callback
from nanotabpfn.utils import get_default_device
from nanotabpfn.model import NanoTabPFNModel


def train(model: NanoTabPFNModel, prior: DataLoader, criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
          epochs: int, accumulate_gradients: int = 1, lr: float = 1e-4, device: Optional[str] = None,
          callbacks: list[Callback]=[], ckpt: Optional[Dict] = None):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for, the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)

    assert len(prior) % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'

    try:
        for epoch in range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.
            total_r2 = 0.
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data['single_eval_pos']
                data = (full_data['x'].to(device),
                        full_data['y'][:, :single_eval_pos].to(device))
                if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
                    continue
                targets = full_data['y'].to(device)

                output = model(data, single_eval_pos=single_eval_pos)
                targets = targets[:, single_eval_pos:]
                if classification_task:
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])
                losses = criterion(output, targets)
                loss = losses.mean() / accumulate_gradients
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients
                
                # Compute R² scores
                if not classification_task:
                    preds = criterion.mean(output)
                    ss_res = torch.sum((targets.squeeze(-1) - preds) ** 2, dim=1)
                    ss_tot = torch.sum((targets.squeeze(-1) - preds.mean(dim=1, keepdim=True)) ** 2, dim=1)
                    r2 = (1 - ss_res / ss_tot).mean()
                    total_r2 += r2.cpu().detach().item() * accumulate_gradients

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            if not classification_task:
                mean_r2 = total_r2 / len(prior)
            model.eval()
            optimizer.eval()

            training_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(training_state, 'latest_checkpoint.pth')

            for callback in callbacks:
                if not classification_task:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model, dist=criterion, r2=mean_r2)
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model)
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return model, total_loss
