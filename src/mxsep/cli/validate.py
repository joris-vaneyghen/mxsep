import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from mxsep.cfg import Config

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../configs", config_name="separate")
def run(cfg : DictConfig):
    #todo
    pass




if __name__ == '__main__':
    run()