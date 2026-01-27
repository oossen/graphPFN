import numpy as np
import pandas as pd
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score

from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNRegressor
import tfmplayground.model, graphpfn.graph_prior_model, graphpfn.binary_mask_model, graphpfn.fallback_model, graphpfn.graph_prior_fallback_model, graphpfn.binary_mask_fallback_model, graphpfn.pos_encoding_model, graphpfn.transfer_model, graphpfn.transfer_model_binary, graphpfn.transfer_model, graphpfn.gcn_model


def init_model_from_state_dict_file(model_type: str, file_path: str):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
    if model_type == "pfn":
        model_class = tfmplayground.model.NanoTabPFNModel
    elif model_type == 'graph_prior':
        model_class = graphpfn.graph_prior_model.GraphPFNModel
    elif model_type == 'binary':
        model_class = graphpfn.binary_mask_model.GraphPFNModel
    elif model_type == 'fallback':
        model_class = graphpfn.fallback_model.GraphPFNModel
    elif model_type == 'graph_prior_fallback':
        model_class = graphpfn.graph_prior_fallback_model.GraphPFNModel
    elif model_type == 'binary_fallback':
        model_class = graphpfn.binary_mask_fallback_model.GraphPFNModel
    elif model_type == 'pos_encoding':
        model_class = graphpfn.pos_encoding_model.GraphPFNModel
    elif model_type == 'transfer_pe':
        model_class = graphpfn.transfer_model.GraphPFNModel
    elif model_type == 'transfer_binary':
        model_class = graphpfn.transfer_model_binary.GraphPFNModel
    elif model_type == 'transfer':
        model_class = graphpfn.transfer_model.GraphPFNModel
    elif model_type == 'gcn':
        model_class = graphpfn.gcn_model.GraphPFNModel
        
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model = model_class(**state_dict['architecture'])
    model.load_state_dict(state_dict['model'])
    return model


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x

def to_numeric(x):
    return x.apply(pd.to_numeric, errors='coerce').to_numpy()

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(to_numeric)), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean')) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ],
        sparse_threshold=0
    )
    return preprocessor


class Regressor(NanoTabPFNRegressor):
    """ scikit-learn like interface """
    def __init__(self, model, dist: FullSupportBarDistribution, device: str|torch.device|None = None):
        if device is None:
            device = get_default_device()
        self.model = model.to(device)
        self.device = device
        self.dist = dist

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

    def predict(self, X_test: np.ndarray, **kwargs):
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X_test_transformed = self.feature_preprocessor.transform(X_test)
        assert isinstance(X_test_transformed, np.ndarray) and isinstance(self.X_train, np.ndarray), "Preprocessing did not produce numpy arrays!"
        X = np.concatenate((self.X_train, X_test_transformed))
        y = self.y_train

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), **kwargs).squeeze(0)
            preds = self.dist.mean(logits)
            preds = preds.cpu().numpy()

        return preds
    
    def log_ppd(self, X_test: np.ndarray, y_test: np.ndarray, **kwargs):
        """
        Computes the log predictive probability density of the provided test targets.
        """
        X_test_transformed = self.feature_preprocessor.transform(X_test)
        assert isinstance(X_test_transformed, np.ndarray) and isinstance(self.X_train, np.ndarray), "Preprocessing did not produce numpy arrays!"
        X = np.concatenate((self.X_train, X_test_transformed))
        y = self.y_train
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), **kwargs)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=self.device)
            self.dist.to(device=self.device)
            # expand logits so that there is one for each input y
            logits = logits.view(1, 1, -1).expand(len(y_test), 1, -1)
            neg_log_probs = self.dist.forward(logits, y_test_tensor).squeeze(0)
        
        return -neg_log_probs.cpu().numpy()
    
    
def cross_validate(reg: NanoTabPFNRegressor, X: np.ndarray, y: np.ndarray, single_eval_pos: int, n_folds: int = 5, **kwargs):
    """
    Perform `n_fold`-fold cross validation using the model `reg` on the provided data.
    
    Parameters
    ----------
    reg : NanoTabPFNRegressor
        the model to evaluate
    X : np.ndarray (n_samples, n_features)
        the features (train and test split) of the input data
    y : np.ndarray (n_samples)
        the targets (train and test split)
    single_eval_pos : int
        number of train samples
    n_folds : int
        number of cross validation folds
    kwargs : Dict
        additional arguments for prediction (adjacency matrix, graph...)
    
    """
    cv = ShuffleSplit(n_splits=10*n_folds, train_size=single_eval_pos, test_size=len(X)-single_eval_pos, random_state=42)
    scores = []
    for train_index, test_index in cv.split(X):
        # break out of loop once we have enough valid splits
        if len(scores) >= n_folds:
            break
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # reject splits with constant features
        if np.any(np.ptp(X_train, axis=0) == 0):
            continue 
        
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test, **kwargs) 
        score = r2_score(y_test, predictions)
        scores.append(score)
    if len(scores) < n_folds:
        print(f"Warning: Only using {len(scores)} splits.")
    return np.mean(scores)