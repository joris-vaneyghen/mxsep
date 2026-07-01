import logging
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from mxsep.cfg import Config
from mxsep.training import DDPTrainer

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    print(f"Using default backend: {backend}")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def run_proc(rank, world_size, cfg : Config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    trainer = DDPTrainer(rank, world_size, cfg)
    trainer.train()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
# @hydra.main(version_base=None,  config_name="config")
def run(cfg : Config):
    log.info("Info level message")

    cfg: Config = OmegaConf.merge(defaults, cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg_dict["training"]["monitoring"]["wandb"]["api_key"] = "******"
    except (KeyError, TypeError):
        pass
    
    print(OmegaConf.to_yaml(cfg_dict))

    world_size = torch.accelerator.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    mp.spawn(run_proc, args=(world_size,cfg, ), nprocs=world_size, join=True)



if __name__ == '__main__':
    run()