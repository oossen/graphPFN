from pfns.bar_distribution import FullSupportBarDistribution

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train
from nanotabpfn.utils import get_default_device, make_global_bucket_edges

from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import default_graph_config, default_dataset_config, default_preprocessing_config, default_scm_config

num_buckets = 100
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=num_buckets,
)

prior = ObservationalDataLoader(5, 10, default_graph_config, default_scm_config, default_preprocessing_config, default_dataset_config, seed=42)

device = get_default_device()
bucket_edges = make_global_bucket_edges(
    filename="50x3_1280k_regression.h5",   # just for testing!
    n_buckets=num_buckets,
    device=device,
)
dist = FullSupportBarDistribution(bucket_edges)

trained_model, loss = train(model=model, prior=prior, criterion=dist, epochs=2)