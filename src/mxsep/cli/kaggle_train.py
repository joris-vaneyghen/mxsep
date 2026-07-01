import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from mxsep import training
from mxsep.cfg import Config
from mxsep.kaggle import create_notebook
from mxsep.training import Trainer

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())

@hydra.main(version_base=None, config_path="../configs", config_name="config")
# @hydra.main(version_base=None,  config_name="config")
def run(cfg : Config):
    log.info("Info level message")

    cfg: Config = OmegaConf.merge(defaults, cfg)
    create_notebook(user="jorisvaneyghen", cmd="mxsep-train", cfg=cfg)



if __name__ == '__main__':
    run()