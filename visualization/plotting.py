from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from sklearn.metrics import r2_score

def plot_point_clouds(X: torch.Tensor, y: torch.Tensor, filename: str, single_eval_pos: int = 0, graph=None):
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    n_pairs = len(pairs)
    n_plots = n_pairs + X.shape[1] # pairs of features plus pairs involving the target
    n_cols = int((n_plots) ** 0.5)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes) # in case there is only one plot
    axes = axes.flatten()
    
    labels = list(graph.nodes) if graph is not None else [f"feature {i}" for i in range(X.shape[1])]
    labels.remove('y')

    for i, (p, q) in enumerate(pairs):
        if graph is not None and (graph.has_edge(labels[p], labels[q]) or graph.has_edge(labels[q], labels[p])):
            main_color = 'orange'
        else:
            main_color = 'red'
        axes[i].scatter(X[:single_eval_pos, p].numpy(), X[:single_eval_pos, q].numpy(), s=5, c=main_color)
        axes[i].scatter(X[single_eval_pos:, p].numpy(), X[single_eval_pos:, q].numpy(), s=5, c='gray')
        axes[i].set_xlabel(labels[p])
        axes[i].set_ylabel(labels[q])
    for i in range(X.shape[1]):
        if graph is not None and (graph.has_edge(labels[i], 'y') or graph.has_edge('y', labels[i])):
            main_color = 'cyan'
        else:
            main_color = 'blue'
        axes[i + n_pairs].scatter(X[:single_eval_pos, i].numpy(), y[:single_eval_pos].numpy(), s=5, c=main_color)
        axes[i + n_pairs].scatter(X[single_eval_pos:, i].numpy(), y[single_eval_pos:].numpy(), s=5, c='gray')
        axes[i + n_pairs].set_xlabel(labels[i])
        axes[i + n_pairs].set_ylabel('y')

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
    

def plot_prob_adj(prob_adj: torch.Tensor, filename: str):
    plt.imshow(prob_adj, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Probability')
    plt.title('Probabilistic adjacency matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(filename, dpi=300)
    plt.close()
    
    
def plot_graph(g: nx.Graph, filename: str, **drawing_style):
    node_color = ['gray' if g.nodes[v].get('hidden', False) else 'blue' for v in g.nodes]
    weights = [g[u][v].get('weight', 1.0) for u, v in g.edges]
    base_color = (0, 0, 0) 
    edge_color = [base_color + (w,) for w in weights]
    drawing_style.setdefault('with_labels', True) 
    nx.draw(g, node_color=node_color, edge_color=edge_color, **drawing_style)
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