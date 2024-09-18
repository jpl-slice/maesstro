import os

import yaml


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            ext = ".yaml" if len(os.path.splitext(cf)) == 1 else ""
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ext)
            if not os.path.exists(cf):
                cf = os.path.basename(cf)
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                cf = os.path.join(repo_root, "config", config_dir, cf + ext)
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg = dict(l, **cfg)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg
