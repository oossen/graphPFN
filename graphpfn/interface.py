import torch
import torch.nn.functional as F
from numpy import ndarray
from pfns.bar_distribution import FullSupportBarDistribution

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import get_default_device
from nanotabpfn.interface import NanoTabPFNRegressor

from prior.preprocessing.preprocessing import Preprocessor


class Regressor(NanoTabPFNRegressor):
    """ 
    Scikit-learn like interface.
    This class converts arrays to torch tensors, adds batch dimensions as needed, and performs preprocessing.
    """
    def __init__(self, model: NanoTabPFNModel, dist: FullSupportBarDistribution, preprocessor: Preprocessor, device=get_default_device()):
        self.model = model.to(device)
        self.dist = dist
        self.preprocessor = preprocessor
        self.device = device

    def fit(self, X_train: ndarray, y_train: ndarray):
        """
        Stores X_train and y_train for later use. Computes target normalization. Builds normalized bar distribution from existing self.dist.
        """
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.X_train_tensor, self.y_train_tensor = self.preprocessor.fit(self.X_train_tensor, self.y_train_tensor)
        self.y_train_tensor = self.y_train_tensor.unsqueeze(-1) # now shape (1, N, 1)

    def predict(self, X_test: ndarray) -> ndarray:
        """
        Performs in-context learning using X_train and y_train. Predicts the means of the output distributions for X_test.
        """
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device).unsqueeze(0)
        X_test_tensor = self.preprocessor.process_x(torch.Tensor(X_test_tensor))
        X_tensor = torch.concatenate((self.X_train_tensor, X_test_tensor), dim=1)
        single_eval_pos = self.X_train_tensor.shape[1]

        with torch.no_grad():
            logits = self.model((X_tensor, self.y_train_tensor), single_eval_pos=single_eval_pos).squeeze(0)
            preds = self.dist.mean(logits)
            preds = self.preprocessor.unprocess_y(preds).squeeze(0)

        return preds.cpu().numpy()
