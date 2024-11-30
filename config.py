import ml_collections

def create_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 5e-4
    config.batch_size = 64
    config.loss_type = "hinge"
    config.alpha = 0.5
    config.beta = 0.2

    config.data_dir = "data/dpo/top_100k.csv"

    return config