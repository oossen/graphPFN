import argparse
from functools import partial
from typing import List

import torch
from sklearn.metrics import r2_score

from nanotabpfn.callbacks import ConsoleLoggerCallback
from nanotabpfn.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from graphpfn.interface import Regressor
from nanotabpfn.model import NanoTabPFNModel
from graphpfn.train import train
from nanotabpfn.utils import get_default_device
from nanotabpfn.callbacks import Callback

from graphpfn.utils import make_bar_distribution
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.debugging_configs import prior_config, preprocessing_config
from prior.preprocessing.preprocessing import Preprocessor
from prior.utils.hyperparameter_sampling import sample_parameters

parser = argparse.ArgumentParser()

parser.add_argument("-saveweights", type=str, default="nanotabpfn_weights.pth", help="path to save the trained model to")
parser.add_argument("-heads", type=int, default=6, help="number of attention heads")
parser.add_argument("-embeddingsize", type=int, default=192, help="the size of the embeddings used for the cells")
parser.add_argument("-hiddensize", type=int, default=768, help="size of the hidden layer of the mlps")
parser.add_argument("-layers", type=int, default=6, help="number of transformer layers")
parser.add_argument("-batchsize", type=int, default=1, help="batch size used during training (before gradient accumulation)")
parser.add_argument("-accumulate", type=int, default=1, help="number of gradients to accumulate before updating the weights")
parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-steps", type=int, default=100, help="number of steps that constitute one epoch (important for lr scheduler)")
parser.add_argument("-epochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("-loadcheckpoint", type=str, default=None, help="checkpoint from which to continue training")
parser.add_argument("-n_buckets", type=int, default=100, help="number of buckets for the data loader")
parser.add_argument("-n_bardist_samples", type=int, default=100, help="number of data batches used to infer buckets for bar distribution")

args = parser.parse_args()


device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = ObservationalDataLoader(num_steps=args.steps,
                                batch_size=args.batchsize,
                                prior_config=prior_config,
                                preprocessing_config=preprocessing_config,
                                seed=42)

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

prior_factory = partial(ObservationalDataLoader,
                        batch_size=10,
                        prior_config=prior_config,
                        preprocessing_config=preprocessing_config,
                        seed=42)
dist = make_bar_distribution(prior_factory, n_buckets=args.n_buckets, n_samples=args.n_bardist_samples)

if ckpt:
    model.load_state_dict(ckpt['model'])

preprocessor = Preprocessor(**sample_parameters(prior.preprocessing_samplers))

class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        regressor = Regressor(model, dist, preprocessor, device)
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(f'diabetes dataset | avg r2 score {avg_score:.3f}',
              flush=True)
        

class SanityCheckLoggerCallback(ConsoleLoggerCallback):
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        prior = ObservationalDataLoader(num_steps=10,
                                batch_size=1,
                                prior_config=prior_config,
                                preprocessing_config=preprocessing_config,
                                seed=42)
        regressor = Regressor(model, dist, preprocessor, device)
        scores = []
        for data in prior:
            X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
            y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
            X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
            y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
            
            regressor.fit(X_train, y_train)
            pred = regressor.predict(X_test)
            scores.append(r2_score(y_test, pred))
        avg_score = sum(scores) / len(scores)
        print(f'synthetic data | avg r2 score {avg_score:.3f}',
              flush=True)
            


evaluation_callback = EvaluationLoggerCallback(TOY_TASKS_REGRESSION)
sanity_callback = SanityCheckLoggerCallback()
logger_callback = ConsoleLoggerCallback()
callbacks: List[Callback] = [logger_callback, evaluation_callback, sanity_callback]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt
)

torch.save(trained_model.to('cpu').state_dict(), args.saveweights)
