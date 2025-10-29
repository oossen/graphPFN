import numpy as np
import pandas as pd
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from nanotabpfn.utils import get_default_device
from nanotabpfn.interface import NanoTabPFNRegressor

import graphpfn.attention_model, graphpfn.additive_encoding_model


def init_model_from_state_dict_file(file_path):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    if state_dict['architecture']['num_graph_attention_heads'] is not None:
        model = graphpfn.attention_model.GraphPFNModel(
            num_attention_heads=state_dict['architecture']['num_attention_heads'],
            num_graph_attention_heads=state_dict['architecture']['num_graph_attention_heads'],
            embedding_size=state_dict['architecture']['embedding_size'],
            mlp_hidden_size=state_dict['architecture']['mlp_hidden_size'],
            num_layers=state_dict['architecture']['num_layers'],
            num_outputs=state_dict['architecture']['num_outputs'],
        )
    elif state_dict['architecture']['gcn_hidden_size'] is not None:
        model = graphpfn.additive_encoding_model.GraphPFNModel(
            num_attention_heads=state_dict['architecture']['num_attention_heads'],
            gcn_hidden_size=state_dict['architecture']['gcn_hidden_size'],
            embedding_size=state_dict['architecture']['embedding_size'],
            mlp_hidden_size=state_dict['architecture']['mlp_hidden_size'],
            num_layers=state_dict['architecture']['num_layers'],
            num_outputs=state_dict['architecture']['num_outputs'],
        )
    else:
        raise ValueError("No valid model can be built from this state dict!")
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

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X_test_transformed = self.feature_preprocessor.transform(X_test)
        assert isinstance(X_test_transformed, np.ndarray) and isinstance(self.X_train, np.ndarray), "Preprocessing did not produce numpy arrays!"
        X = np.concatenate((self.X_train, X_test_transformed))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), **kwargs).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * torch.tensor(np.std(self.y_train, ddof=1) + 1e-8, dtype=torch.float32, device=self.device) + torch.tensor(np.mean(self.y_train), dtype=torch.float32, device=self.device)

        return preds.cpu().numpy()