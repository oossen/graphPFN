from collections import defaultdict
from tabpfn import TabPFNRegressor
from graphpfn.interface import Regressor
from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader

from tfmplayground.callbacks import TensorboardLoggerCallback
from tfmplayground.evaluation import get_openml_predictions
from tfmplayground.utils import get_default_device

from sklearn.metrics import r2_score
from torch.utils.tensorboard.writer import SummaryWriter


class EvaluationLoggerCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on `tasks`.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, log_dir: str, tasks, prior):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        dist = kwargs['dist']
        regressor = Regressor(model, dist, get_default_device())
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        self.writer.add_scalar('diabetes R²', avg_score, epoch)
        

class SanityCheckLoggerCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, log_dir: str, prior_factory, num_steps=500):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_factory = prior_factory
        self.num_steps = num_steps
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = self.prior_factory(self.num_steps)
        dist = kwargs['dist']
        regressor = Regressor(model, dist, get_default_device())
        scores = []
        for data in test_prior:
            X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
            y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
            X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
            y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
            
            regressor.fit(X_train, y_train)
            pred = regressor.predict(X_test, **data['graph_information'])
            scores.append(r2_score(y_test, pred))
        avg_score = sum(scores) / len(scores)
        self.writer.add_scalar('synthetic R²', avg_score, epoch)
        

class SanityCheckPerGraphLoggerCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    Report results separately for each possible graph index.
    """
    def __init__(self, log_dir: str, prior_factory, num_steps=500):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_factory = prior_factory
        self.num_steps = num_steps
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = self.prior_factory(self.num_steps)
        dist = kwargs['dist']
        regressor = Regressor(model, dist, get_default_device())
        scores = defaultdict(list)
        for data in test_prior:
            X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
            y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
            X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
            y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
            index = data['graph_information']['graph_index']
            
            regressor.fit(X_train, y_train)
            pred = regressor.predict(X_test, **data['graph_information'])
            print(index, pred)
            r2 = r2_score(y_test, pred)
            scores[index].append(r2)
            scores['all'].append(r2)
        for key, value in scores.items():
            avg_score = sum(value) / len(value)
            self.writer.add_scalar(f'synthetic R² for graph {key}', avg_score, epoch)