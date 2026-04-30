import openml
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import pandas as pd

task_ids = [("fish_toxicity", 363698),
            ("concrete_compressive_strength", 363625),
            ("healthcare_insurance_expenses", 363675),
            ("airfoil_self_noise", 363612),
            ("used_fiat_500", 363615),
            ("wine_quality", 363708),
            ("miami_housing", 363686),
            ("houses", 363678),
            ("food_delivery_time", 363672),
            ("physiochemical_protein", 363693),
            ("diamonds", 363631),]
for name, id in task_ids:
    # Initialize task
    task = openml.tasks.get_task(id, download_splits=False)

    # Access dataset and load into DataFrame
    dataset = task.get_dataset()
    df, *_ = dataset.get_data(dataset_format="dataframe")
    
    target_col = task.target_name
    # Remove target from its current spot and add it to the end
    df[target_col] = df.pop(target_col)

    # Export
    out_dir = f"icml_plots/input/{name}"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"icml_plots/input/{name}/data.csv", index=False)
    
    # Preprocessed version for causal discovery
    df_transformed = df.copy()
    scaler = StandardScaler()
    encoder = OrdinalEncoder()
    for col in df_transformed.columns:
        # Check if column is numerical
        if not pd.api.types.is_numeric_dtype(df_transformed[col]):
            df_transformed[[col]] = encoder.fit_transform(df_transformed[[col]])
        df_transformed[[col]] = scaler.fit_transform(df_transformed[[col]])
            
    df_transformed.to_csv(f"icml_plots/input/{name}/data_preprocessed.csv", index=False)