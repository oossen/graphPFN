from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
import networkx as nx

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
        axes[i].scatter(X[:single_eval_pos, p].cpu().numpy(), X[:single_eval_pos, q].cpu().numpy(), s=5, c=main_color)
        axes[i].scatter(X[single_eval_pos:, p].cpu().numpy(), X[single_eval_pos:, q].cpu().numpy(), s=5, c='gray')
        axes[i].set_xlabel(labels[p])
        axes[i].set_ylabel(labels[q])
    for i in range(X.shape[1]):
        if graph is not None and (graph.has_edge(labels[i], 'y') or graph.has_edge('y', labels[i])):
            main_color = 'cyan'
        else:
            main_color = 'blue'
        axes[i + n_pairs].scatter(X[:single_eval_pos, i].cpu().numpy(), y[:single_eval_pos].cpu().numpy(), s=5, c=main_color)
        axes[i + n_pairs].scatter(X[single_eval_pos:, i].cpu().numpy(), y[single_eval_pos:].cpu().numpy(), s=5, c='gray')
        axes[i + n_pairs].set_xlabel(labels[i])
        axes[i + n_pairs].set_ylabel('y')

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    
    
def plot_correlation(X: torch.Tensor, filename: str):
    corr = torch.corrcoef(X.T).cpu()
    plt.imshow(corr.abs() ** 0.5, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Correlation')
    plt.title('Feature Correlation Matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(filename, dpi=300)
    plt.close()
    

def plot_adj(prob_adj: torch.Tensor, adj: Optional[torch.Tensor], filename: str):
    plt.imshow(prob_adj.detach().cpu(), cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Probability')
    if adj is not None:
        rows, cols = torch.where(adj == 1)
        plt.scatter(cols.cpu(), rows.cpu(), 
                    marker='x', 
                color='red', 
                s=20,          # Size of the cross
                linewidths=1)  # Thickness of the cross lines
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
    
def plot_likelihoods(bucket_mids: torch.Tensor, probs: torch.Tensor, true_y: float, filename: str):
    eps = 1e-3
    max_prob = torch.max(probs)
    mask = probs > eps * max_prob
    indices = torch.where(mask)[0]
    buffer = 1
    start_idx = max(0, indices[0] - buffer)
    end_idx = min(len(bucket_mids) - 1, indices[-1] + buffer)
    probs = probs[start_idx:end_idx+1].cpu()
    bucket_mids = bucket_mids[start_idx:end_idx+1].cpu()
    
    plt.plot(bucket_mids, probs, label="p(y)")
    plt.axvline(x=true_y, color='red', linestyle='--', linewidth=1)  # the true y-value
    plt.xlabel("y")
    plt.ylabel("p(y)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()