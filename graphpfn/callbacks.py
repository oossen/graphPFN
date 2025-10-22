from tabpfn import TabPFNRegressor
from graphpfn.interface import GraphPFNRegressor
from prior.dataloaders.observational_dataloader import ObservationalDataLoader

from nanotabpfn.callbacks import TensorboardLoggerCallback
from nanotabpfn.evaluation import get_openml_predictions
from nanotabpfn.utils import get_default_device
from nanotabpfn.interface import NanoTabPFNRegressor

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
        regressor = NanoTabPFNRegressor(model, dist, get_default_device())
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
    def __init__(self, log_dir: str, prior, num_steps=50):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_config = prior.prior_config
        self.num_steps = num_steps
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = ObservationalDataLoader(num_steps=self.num_steps,
                                batch_size=1,
                                prior_config=self.prior_config,
                                seed=42)
        dist = kwargs['dist']
        regressor = NanoTabPFNRegressor(model, dist, get_default_device())
        scores = []
        for data in test_prior:
            X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
            y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
            X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
            y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
            
            regressor.fit(X_train, y_train)
            pred = regressor.predict(X_test)
            scores.append(r2_score(y_test, pred))
        avg_score = sum(scores) / len(scores)
        self.writer.add_scalar('synthetic R²', avg_score, epoch)
        

class SanityCheckLoggerGraphCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, log_dir: str, prior, num_steps=50):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_config = prior.prior_config
        self.num_steps = num_steps
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = ObservationalDataLoader(num_steps=self.num_steps,
                                batch_size=1,
                                prior_config=self.prior_config,
                                seed=42)
        dist = kwargs['dist']
        regressor = GraphPFNRegressor(model, dist, get_default_device())
        scores = []
        for data in test_prior:
            X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
            y_train = data['y'][0, :data['single_eval_pos'], 0].cpu().numpy()
            X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
            y_test = data['y'][0, data['single_eval_pos']:, 0].cpu().numpy()
            adjacency_matrix = data['adjacency_matrix'].cpu().numpy()
            
            regressor.fit(X_train, y_train, adjacency_matrix)
            pred = regressor.predict(X_test)
            scores.append(r2_score(y_test, pred))
        avg_score = sum(scores) / len(scores)
        self.writer.add_scalar('synthetic R²', avg_score, epoch)
        

class TensorboardLoggerR2Callback(TensorboardLoggerCallback):
    """ Logger callback that prints epoch information to the console, including R²-score. """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        r2 = kwargs["r2"]
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('R²/train', r2, epoch)
        self.writer.add_scalar('Time/epoch', epoch_time, epoch)