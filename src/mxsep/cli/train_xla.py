import logging

import hydra
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from mxsep.cfg import Config
from mxsep.training import XLATrainer
import torch.distributed as dist

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())

def setup():
    dist.init_process_group('xla', init_method='xla://')

def _mp_fn(index, cfg):
    setup()
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    trainer = XLATrainer(rank, world_size, cfg)
    trainer.train()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
# @hydra.main(version_base=None,  config_name="config")
def run(cfg: Config):
    log.info("Info level message")

    cfg: Config = OmegaConf.merge(defaults, cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg_dict["training"]["monitoring"]["wandb"]["api_key"] = "******"
    except (KeyError, TypeError):
        pass

    print(OmegaConf.to_yaml(cfg_dict))

    torch_xla.launch(_mp_fn, args=(cfg))


if __name__ == "__main__":
    run()
