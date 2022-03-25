
def instantiate(config):
    config = config.copy()
    _class = config["_target_"]
    del config["_target_"]
    instance = _class(**config)
    
    return instance

def load_model_trainer(trainer_settings, db_train_data, db_test_data):
    pass 