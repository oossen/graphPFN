from typing import Dict
import pandas as pd

from graphpfn.interface import cross_validate


def compare_all(prior, models: Dict, filename: str, baseline=None, graph_info_trafo=None):
    """
    Make plots of the performance of the given models against various dataset characteristics.
    If `baseline` is given, instead plot the improvment over the baseline model.
    If `graph_info_trafo` is given, it must be a function that transforms the graph information before adding it to the data frame.
    """
    rows = []
    for data in prior:
        # add sampled parameters to data frame
        sampled_params = data["graph_information"]["sampled_params"]
        row = {}
        for k, v in sampled_params.items():
            row[k] = v
        rows.append(row)
        # evaluate on model, select first and only batch
        X = data['x'][0].cpu().numpy()
        y = data['y'][0].cpu().numpy()
        single_eval_pos = data['single_eval_pos']
        graph_info = data['graph_information']
        if graph_info_trafo is not None:
            graph_info, graph_info_score = graph_info_trafo(graph_info)
            row['graph_info_score'] = graph_info_score
        for name, model in models.items():
            score = cross_validate(model, X, y, single_eval_pos, 5, **graph_info)
            row[name] = score
    df = pd.DataFrame(rows)
    if baseline is not None:
        baseline_values = df[baseline].copy()
        for name in models.keys():
            df[name] = df[name] - baseline_values
        df.drop(columns=[baseline], inplace=True)
    df.to_csv(f"{filename}/full_comparison.csv")
    return df