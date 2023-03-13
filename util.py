import yaml
import os, pathlib

def get_config():
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg_path = os.path.join(root_path, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg