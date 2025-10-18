import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from sklearn.metrics import r2_score

def plot_point_clouds(X: torch.Tensor, filename: str):
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    n_pairs = len(pairs)
    n_cols = int((n_pairs) ** 0.5)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for ax, (p, q) in zip(axes, pairs):
        ax.scatter(X[:, p].numpy(), X[:, q].numpy(), s=5, c='red')
        ax.set_xlabel(f"feature {p}")
        ax.set_ylabel(f"feature {q}")

    for ax in axes[len(pairs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    
    
def plot_correlation(X: torch.Tensor, filename: str):
    corr = torch.corrcoef(X.T)
    plt.imshow(corr.abs() ** 0.5, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Correlation')
    plt.title('Feature Correlation Matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(filename, dpi=300)
    plt.close()
    
    
def plot_graph(g: nx.DiGraph, filename: str):
    nx.draw(g, with_labels=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    

def plot_r2(prior: DataLoader, filename: str):
    models = {}
    from nanotabpfn import NanoTabPFNRegressor
    models["nano_tabpfn"] = NanoTabPFNRegressor()
    from tabpfn import TabPFNRegressor
    models["tabpfn"] = TabPFNRegressor()
    from sklearn.ensemble import RandomForestRegressor
    models["rf_10"] = RandomForestRegressor(10)
    from sklearn.svm import SVR
    models["svr"] = SVR()
    from sklearn.linear_model import Ridge
    models["ridge"] = Ridge()
    
    scores = {model: [] for model in models}

    for data in prior:
        X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
        y_train = data['y'][0, :data['single_eval_pos'], :].cpu().numpy()
        X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
        y_test = data['y'][0, data['single_eval_pos']:, :].cpu().numpy()
        
        for name, model in models.items():
            model.fit(X_train, y_train.ravel())
            pred = model.predict(X_test)
            scores[name].append(r2_score(y_test.ravel(), pred))
            
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    for i, name in enumerate(models.keys()):
        axes[i].boxplot(scores[name], label=name, showfliers=False)
        axes[i].set_title(name)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()