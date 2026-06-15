import logging
import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from mxsep.cfg import Config
from mxsep.inference import Separator

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="../configs", config_name="separate")
def run(cfg : DictConfig):

    # todo Test Time Augmentation (TTA). Adjust the "Overlap" and "Chunk Size" parameters.

    model_path = cfg.model_path
    device = cfg.device
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_ext = cfg.output_ext
    batch_size = cfg.batch_size
    nb_passes = cfg.nb_passes
    crossfade_duration = cfg.crossfade_duration

    if 'tta' in cfg:
        tta = cfg.tta # Test Time Augmentation (TTA) containing overlap, channel_inverse, polarity_inverse, phase optimizing, residual...
    else:
        tta = None
    separator = Separator(model_path, device=device, tta=tta, crossfade_duration=crossfade_duration, nb_passes=nb_passes, batch_size=batch_size, output_ext=output_ext)
    separator.separate_dir(input_dir, output_dir)

if __name__ == '__main__':
    run()