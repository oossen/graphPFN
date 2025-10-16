from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.preprocessing.preprocessing import Preprocessor
from prior.utils.hyperparameter_sampling import sample_parameters

from nanotabpfn.callbacks import ConsoleLoggerCallback
from nanotabpfn.evaluation import get_openml_predictions
from nanotabpfn.utils import get_default_device
from graphpfn.interface import Regressor

from sklearn.metrics import r2_score


class EvaluationLoggerCallback(ConsoleLoggerCallback):
    """
    On epoch end, evaluate the model on `tasks`.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, tasks, prior, dist):
        self.tasks = tasks
        self.dist = dist
        self.preprocessing_config = prior.preprocessing_config
        self.preprocessor = prior.preprocessor
    

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        regressor = Regressor(model, self.dist, self.preprocessor, get_default_device())
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(f'diabetes dataset | avg r2 score {avg_score:.3f}',
              flush=True)
        

class SanityCheckLoggerCallback(ConsoleLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, prior, dist):
        self.prior_config = prior.prior_config
        self.dist = dist
        self.preprocessing_config = prior.preprocessing_config
        self.preprocessor = prior.preprocessor
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = ObservationalDataLoader(num_steps=10,
                                batch_size=1,
                                prior_config=self.prior_config,
                                preprocessing_config=self.preprocessing_config,
                                seed=42)
        regressor = Regressor(model, self.dist, self.preprocessor, get_default_device())
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
        print(f'synthetic data | avg r2 score {avg_score:.3f}',
              flush=True)