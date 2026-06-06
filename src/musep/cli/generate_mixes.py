import json
import logging
import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from musep.cfg import Config, DatasetConfig
from musep.data.random_mix import generate_mix, generate_validation_mix
from musep.data.random_mix.mix_generator import generate_validation_mix_orphans

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())

# Merge (missing keys get defaults)

def generate_train_jsonl(epoch:int, jsonl_path:Path, config:DatasetConfig):
    print(f"Generating jsonl at {jsonl_path}")
    directory = jsonl_path.parent
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(jsonl_path, 'w') as f:
        for mix in generate_mix(epoch, config):
            json_line = json.dumps(mix.to_dict())
            f.write(json_line + '\n')

def generate_validation_jsonl(jsonl_path:Path, config:DatasetConfig):
    print(f"Generating jsonl at {jsonl_path}")
    directory = jsonl_path.parent
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(jsonl_path, 'w') as f:
        for mix in generate_validation_mix(config):
            json_line = json.dumps(mix.to_dict())
            f.write(json_line + '\n')
        for mix in generate_validation_mix_orphans(config):
            json_line = json.dumps(mix.to_dict())
            f.write(json_line + '\n')


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg : Config):
    log.info("Info level message")
    cfg: Config = OmegaConf.merge(defaults, cfg)

    assert cfg.dataset.random_mix
    assert cfg.dataset.train.predefined_jsonl_path

    if cfg.dataset.train.predefined_jsonl_path.suffix == '.jsonl':
        jsonl_file =cfg.dataset.train.predefined_jsonl_path
        # assert that the file has jsonl extension
        generate_train_jsonl(0, jsonl_file, cfg.dataset)
    else:
        assert cfg.training
        assert cfg.training.epochs
        jsonl_dir =  cfg.dataset.train.predefined_jsonl_path
        for epoch in range(cfg.training.epochs):
            jsonl_file = jsonl_dir / f"{epoch}.jsonl"
            generate_train_jsonl(epoch, jsonl_file, cfg.dataset)


    if cfg.dataset.validation.predefined_jsonl_path.suffix == '.jsonl':
        jsonl_file =cfg.dataset.validation.predefined_jsonl_path
        # assert that the file has jsonl extension
        generate_validation_jsonl(jsonl_file, cfg.dataset)
    else:
        jsonl_dir =  cfg.dataset.validation.predefined_jsonl_path
        for epoch in range(cfg.training.epochs):
            jsonl_file = jsonl_dir / "0.jsonl"
            generate_validation_jsonl(jsonl_file, cfg.dataset)



if __name__ == '__main__':
    run()