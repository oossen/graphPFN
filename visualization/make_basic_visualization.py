import os

from priors.basic_dataloader import BasicDataLoader
from visualization.plotting import plot_graph, plot_point_clouds, plot_r2
import networkx as nx


def make_basic(prior, output_dir: str):
    """
    A visualization suite that works for the basic linear dataloader.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, data in list(enumerate(prior))[:20]:
        single_eval_pos = data['single_eval_pos']
        X = data['x'][0]
        y = data['y'][0]
        g = data['graph_information']['graph']
        plot_graph(g, f"{output_dir}/graph_{i}.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{i}.png", single_eval_pos=single_eval_pos)    
    plot_r2(prior, f"{output_dir}/r2.png")
    

if __name__ == "__main__":
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    g_0 = nx.DiGraph()
    g_0.add_nodes_from([0, 1, 2, 3])
    g_0.add_edges_from([(0, 3)])
    g_1 = nx.DiGraph()
    g_1.add_nodes_from([0, 1, 2, 3])
    g_1.add_edges_from([(0, 3), (1, 3)])
    g_2 = nx.DiGraph()
    g_2.add_nodes_from([0, 1, 2, 3])
    g_2.add_edges_from([(0, 3), (1, 3), (2, 3)])
    graphs = [g_0, g_1, g_2]
    prior = BasicDataLoader(20, 1, graphs, 43)
    make_basic(prior, f"visualization/output/{datetime_str}")