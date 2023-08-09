import yaml

DEFAULT_MODEL_PATH = {
    'layoutlmv3': 'microsoft/layoutlmv3-base',
    'lilt': 'SCUT-DLVCLab/lilt-infoxlm-base'
}


class Config:
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(config)
    if config.model.pretrained_path is None:
        config.model.pretrained_path = DEFAULT_MODEL_PATH[config.model.model_type]
    return config

