



def print_num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    

if __name__ == "__main__":
    from configs.default_configs import training_config
    print_num_parameters(training_config["model"]) #5247824
    from configs.attention_configs import training_config
    print_num_parameters(training_config["model"]) #5247824
    from configs.gcn_configs import training_config
    print_num_parameters(training_config["model"]) # 6731216
    from configs.attention_gcn_configs import training_config
    print_num_parameters(training_config["model"]) # 6731216