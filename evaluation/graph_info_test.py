from configs.default_configs import prior_config
from priors.observational_dataloader import ObservationalDataLoader


def print_percentage_overview(input_dict):
    all_tuples = set()
    for sub_dict in input_dict.values():
        all_tuples.update(sub_dict.keys())
    totals = {k: sum(v.values()) for k, v in input_dict.items()}
    
    for tpl in sorted(list(all_tuples), key=str):
        stats = []
        for dict_label, sub_dict in input_dict.items():
            count = sub_dict.get(tpl, 0)
            total = totals[dict_label]
            percentage = (count / total * 100) if total > 0 else 0
            stats.append(f"{percentage:.2f}% {dict_label}")
        print(f"{tpl}: {', '.join(stats)}")

stats = {}
for prob_adj_mode in ["binary", "beta", "uniform"]:
    prior_config["graph_config"]["prob_adj_mode"] = {"distribution": "categorical",
                                                    "distribution_parameters": {"choices": [prob_adj_mode], "probabilities": [1.0]}}
    prior_config["graph_config"]["num_nodes"] = {"value": 4}
    prior = ObservationalDataLoader(num_steps=1, batch_size=1, prior_config=prior_config, seed=42)
    graph_counts = prior._make_statistics(100000)
    stats[prob_adj_mode] = graph_counts

print_percentage_overview(stats)
