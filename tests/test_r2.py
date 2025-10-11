from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor

import matplotlib.pyplot as plt

from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import prior_config, preprocessing_config

dataloader = ObservationalDataLoader(100, 1, prior_config, preprocessing_config, seed=44)

from nanotabpfn import NanoTabPFNRegressor
nano_tabpfn = NanoTabPFNRegressor()
tabpfn = TabPFNRegressor()
tree = RandomForestRegressor()

nano_tabpfn_scores = []
tabpfn_scores = []
tree_scores = []

for data in dataloader:
    print(data['x'].shape, data['y'].shape, data['single_eval_pos'])
    X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
    y_train = data['y'][0, :data['single_eval_pos'], :].cpu().numpy()
    X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
    y_test = data['y'][0, data['single_eval_pos']:, :].cpu().numpy()
    
    nano_tabpfn.fit(X_train, y_train)
    pred = nano_tabpfn.predict(X_test)
    nano_tabpfn_scores.append(r2_score(y_test, pred))
    
    tabpfn.fit(X_train, y_train.ravel())
    pred = tabpfn.predict(X_test)
    tabpfn_scores.append(r2_score(y_test, pred))
    
    tree.fit(X_train, y_train.ravel())
    pred = tree.predict(X_test)
    tree_scores.append(r2_score(y_test.ravel(), pred))
    
# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
axes[0].boxplot(nano_tabpfn_scores, label="NanoTabPFN", showfliers=False)
axes[0].set_title('NanoTabPFN')
axes[1].boxplot(tabpfn_scores, label="TabPFN", showfliers=False)
axes[1].set_title('TabPFN')
axes[2].boxplot(tree_scores, label="Decision tree", showfliers=False)
axes[2].set_title('Decision tree')
plt.tight_layout()
plt.savefig("plots/r2", dpi=300)
plt.close()