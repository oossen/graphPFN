import os

from priors.linear_dataloader import LinearDataLoader
from visualization.plotting import plot_graph, plot_point_clouds, plot_r2


def make_basic(prior, output_dir: str):
    """
    A visualization suite that works for the basic linear dataloader.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, data in list(enumerate(prior))[:20]:
        X = data['x'][0]
        y = data['y'][0]
        g = data['graph_information']['graph']
        plot_graph(g, f"{output_dir}/graph_{i}.png")
        plot_point_clouds(X, y, f"{output_dir}/point_clouds_{i}.png")    
    plot_r2(prior, f"{output_dir}/r2.png")
    

if __name__ == "__main__":
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    prior = LinearDataLoader(100, 1, 42)
    make_basic(prior, f"visualization/output/{datetime_str}")