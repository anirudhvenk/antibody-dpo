import ml_collections

def create_config():
    config = ml_collections.ConfigDict()
    config.learning_rate = 1e-6
    config.batch_size = 64
    config.loss_type = "hinge"
    config.alpha = 0.2
    config.beta = 0.2

    config.data_dir = "data/dpo/top_100k.csv"
    config.save_path = ""

    return config