from pathlib import Path
import numpy as np
import pandas as pd
from causal_discovery.config import get_default_config
from causal_discovery.causal_explorer import CausalExplorer
from datetime import datetime
import os


seed = 42
tasks = ["fish_toxicity",
            "concrete_compressive_strength",
            "healthcare_insurance_expenses",
            "airfoil_self_noise",
            "used_fiat_500",
            "wine_quality",
            "miami_housing",
            "houses",
            "food_delivery_time",
            "physiochemical_protein",
            "diamonds",]
for task in tasks:
    df = pd.read_csv(f"icml_plots/input/{task}/data_preprocessed.csv")
    df = df.sample(n=min(len(df), 500), random_state=seed)
    config = get_default_config()
    rng = np.random.default_rng(seed)

    now = datetime.now()
    datetime_str = now.strftime("%m_%d_%H_%M")
    output_dir = f"icml_plots/input/{task}/oracle_causal_discovery"
    os.makedirs(output_dir, exist_ok=True)

    results = CausalExplorer.load_causal_results(
        results_path=Path(output_dir),
        data_df=df,
        causal_discovery_config=config,
        num_workers=0,
        rng=rng,
    )