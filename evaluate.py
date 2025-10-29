import torch
from datetime import datetime
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from visualization.pointwise_evaluation import evaluate
from configs.default_configs import prior_config
from pfns.bar_distribution import FullSupportBarDistribution
import graphpfn.interface
import nanotabpfn.interface


prior = ObservationalDataLoader(num_steps=50,
                                batch_size=1,
                                prior_config=prior_config,
                                seed=42)

graph_model = graphpfn.interface.init_model_from_state_dict_file("graph_pfn_model.pth")
graph_buckets = torch.load("graph_pfn_dist.pth")
graph_dist = FullSupportBarDistribution(graph_buckets)

pfn_model = nanotabpfn.interface.init_model_from_state_dict_file("nano_tab_pfn_model.pth")
pfn_buckets = torch.load("nano_tab_pfn_dist.pth")
pfn_dist = FullSupportBarDistribution(pfn_buckets)

models = {"graph": graph_model, "pfn": pfn_model}
dists = {"graph": graph_dist, "pfn": pfn_dist}

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"evaluation/{datetime_str}"
evaluate(prior, models, dists, output_dir, 50)
