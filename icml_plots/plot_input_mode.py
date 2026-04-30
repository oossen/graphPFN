import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

input_dir = "icml_plots/output/input_mode"
metric = "r2"
models = ["attention_binary", "attention_beta", "attention_uniform"]
model_names = {"attention_binary": "Models trained\n with binary type\n graph information",
               "attention_beta": "Models trained\n with beta type\n graph information",
               "attention_uniform": "Models trained\n with uniform type\n graph information"}
baseline_model = "baseline_beta"

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"icml_plots/output/input_mode"
os.makedirs(output_dir, exist_ok=True)
file_map = {
    'binary': f"{input_dir}/results_binary.csv",
    'beta': f"{input_dir}/results_beta.csv",
    'uniform': f"{input_dir}/results_uniform.csv"
}
all_processed_data = []

for eval_label, file_path in file_map.items():
    df = pd.read_csv(file_path)
    df = df[df['metric'] == metric].copy()
    
    baseline_df = df[df['model'] == baseline_model][['ds_id', 'value']]
    baseline_df = baseline_df.rename(columns={'value': 'baseline_value'})
    models_df = df[df['model'].isin(models)].copy()
    merged = models_df.merge(baseline_df, on='ds_id')
    merged['improvement'] = merged['value'] - merged['baseline_value']
    merged['eval_set'] = eval_label
    all_processed_data.append(merged)

final_df = pd.concat(all_processed_data)
final_df['model'] = final_df['model'].map(model_names)

plt.figure(figsize=(10, 7))
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0)
custom_palette = {
    'binary': "#052afa",    # Purple
    'beta': "#f50303",      # Orange
    'uniform': "#00c42a"   # Yellow
}

ax = sns.barplot(
    data=final_df,
    x='model',
    y='improvement',
    hue='eval_set',
    hue_order=['binary', 'beta', 'uniform'],
    order=[model_names[m] for m in models],
    palette=custom_palette,
    capsize=.05,
    err_kws={'linewidth': 1.5},
)

plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel('$R^2$ (improvement over baseline)')
plt.xlabel(None)
handles, _ = ax.get_legend_handles_labels()
new_labels = ['Evaluation on binary graph info', 'Evaluation on beta type graph info', 'Evaluation on uniform type graph info']

ax.legend(
    handles=handles, 
    labels=new_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.22),
    ncol=1,
    title=None,
    frameon=True
)

plt.tight_layout()
plt.savefig(f"{output_dir}/improvement_by_input_mode.png")