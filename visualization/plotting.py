from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from sklearn.metrics import r2_score

def plot_point_clouds(X: torch.Tensor, y: torch.Tensor, filename: str, single_eval_pos=None):
    if single_eval_pos is None:
        # plot everything in the same color
        single_eval_pos = X.shape[0]
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    n_pairs = len(pairs)
    n_plots = n_pairs + X.shape[1] # pairs of features plus pairs involving the target
    n_cols = int((n_plots) ** 0.5)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes) # in case there is only one plot
    axes = axes.flatten()

    for i, (p, q) in enumerate(pairs):
        axes[i].scatter(X[:single_eval_pos, p].numpy(), X[:single_eval_pos, q].numpy(), s=5, c='red')
        axes[i].scatter(X[single_eval_pos:, p].numpy(), X[single_eval_pos:, q].numpy(), s=5, c='gray')
        axes[i].set_xlabel(f"feature {p}")
        axes[i].set_ylabel(f"feature {q}")
    for i in range(X.shape[1]):
        axes[i + n_pairs].scatter(X[:single_eval_pos, i].numpy(), y[:single_eval_pos].numpy(), s=5, c='blue')
        axes[i + n_pairs].scatter(X[single_eval_pos:, i].numpy(), y[single_eval_pos:].numpy(), s=5, c='gray')
        axes[i + n_pairs].set_xlabel(f"feature {i}")
        axes[i + n_pairs].set_ylabel(f"target")

    for ax in axes[n_plots:]:
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
    

def plot_prob_adj(prob_adj, filename: str):
    if not isinstance(prob_adj, list):
        prob_adj = [prob_adj]
        
    n_cols = len(prob_adj)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5)) # Adjust width based on N
    for i, ax in enumerate(axes):
        im = ax.imshow(prob_adj[i], cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_xlabel('Feature Index')
        if i == 0:
            ax.set_ylabel('Feature Index') # Only show Y label on the first plot to save space
        else:
            ax.set_yticks([]) # Hide Y ticks on subsequent plots
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Probability')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_graph(g: nx.DiGraph, filename: str):
    node_color = ['gray' if g.nodes[v].get('dropped', False) else 'blue' for v in g.nodes]
    edge_color = ['gray' if g.edges[e].get('contracted', False) else 'black' for e in g.edges]
    nx.draw(g, with_labels=True, node_color=node_color, edge_color=edge_color)
    plt.savefig(filename, dpi=300)
    plt.close()
    

def plot_r2(prior: DataLoader, filename: str):
    models = {}
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
    

def plot_scores(scores: Dict, filename: str):
    plt.grid(True)
    for name, values in scores.items():
        plt.scatter(range(len(values)), values, label=name)
    plt.xlabel("Index")
    plt.ylabel("R²")
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()