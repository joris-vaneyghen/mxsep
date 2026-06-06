import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from mxsep.cfg import Config

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg : Config):
    cfg: Config = OmegaConf.merge(defaults, cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg_dict))


if __name__ == '__main__':
    run()