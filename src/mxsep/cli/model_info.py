import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torchinfo import summary
from torchview import draw_graph

from mxsep.cfg import Config
from mxsep.models import MusicSourceSeparationModel

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg : Config):
    log.info("Info level message")
    cfg: Config = OmegaConf.merge(defaults, cfg)
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    print(OmegaConf.to_yaml(model_dict))

    model = MusicSourceSeparationModel(cfg.model)
    summary(model, input_size=(1, 2, cfg.model.segment_length), row_settings=("var_names",), col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=3, verbose=1)


    draw_graph(
        model,
        input_size=(1, 2, cfg.model.segment_length),
        expand_nested=True,
        save_graph=True,
        depth=3,
        filename="model",
        directory="./outputs",
    )

if __name__ == '__main__':
    run()