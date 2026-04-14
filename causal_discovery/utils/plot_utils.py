import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def plot_adjacency_heatmap(
    adjacency_matrix: np.ndarray,
    title: str = "Adjacency Matrix",
    path: str = "adjacency.png",
):
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        adjacency_matrix,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.xlabel("Feature", fontsize=14, fontweight="bold")
    plt.ylabel("Feature", fontsize=14, fontweight="bold")
    plt.title(f"{title}", fontsize=16, fontweight="bold", pad=15)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{path}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_graph_from_adjacency_matrix(
    adjacency_matrix: np.ndarray,
    title: str = "Graph",
    path: str = "graph.png",
):
    adjacency_matrix = (adjacency_matrix > 0).astype(int)
    graph = nx.DiGraph(adjacency_matrix)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.title(title)
    plt.savefig(path)
    plt.close()
