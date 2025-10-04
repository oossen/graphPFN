from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor

from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from prior.configs.default_configs import default_graph_config, default_dataset_config, default_preprocessing_config, default_scm_config

dataloader = ObservationalDataLoader(3, 10, default_graph_config, default_scm_config, default_preprocessing_config, default_dataset_config, seed=43)

from nanotabpfn import NanoTabPFNRegressor
nano_reg = NanoTabPFNRegressor()
reg = TabPFNRegressor()

for data in dataloader:
    print(data['x'].shape, data['y'].shape, data['single_eval_pos'])
    X_train = data['x'][0, :data['single_eval_pos'], :].cpu().numpy()
    y_train = data['y'][0, :data['single_eval_pos'], :].cpu().numpy()
    X_test = data['x'][0, data['single_eval_pos']:, :].cpu().numpy()
    y_test = data['y'][0, data['single_eval_pos']:, :].cpu().numpy()
    
    nano_reg.fit(X_train, y_train)
    pred = nano_reg.predict(X_test)
    print(r2_score(y_test, pred))
    
    reg.fit(X_train, y_train.ravel())
    pred = reg.predict(X_test)
    print(r2_score(y_test, pred))
    