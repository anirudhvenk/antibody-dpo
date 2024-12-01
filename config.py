import ml_collections

def create_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 5e-4
    config.batch_size = 64
    config.loss_type = "hinge"
    config.alpha = 1.0
    config.beta = 0.1

    config.train_data = "data/dpo/holdout/dpo_train_data.csv"
    config.eval_data = "data/dpo/holdout/dpo_val_data.csv"

    return config