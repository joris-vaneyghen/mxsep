import logging
import os
import time

import hydra
import torch
import torchaudio
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from mxsep.cfg import Config
from mxsep.models import ISTFTModule
from mxsep.training import Trainer

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=Config)
OmegaConf.register_new_resolver("eval", eval)

defaults = OmegaConf.structured(Config())




def write(
    output_path: str,
    idx: int,
    mix: torch.Tensor,
    target_sources: torch.Tensor,
    sample_rate: int,
):
    """
    Write mix and source audio files to disk.

    Args:
        output_path: Base directory path for output files
        idx: Batch index (used for filename/directory)
        mix: Shape (batch_size, channels, samples)
        target_sources: Shape (batch_size, sources, channels, samples)
        sample_rate: Audio sample rate in Hz

    Note: This version creates subdirectories for each batch element.
    """
    batch_size = mix.shape[0]

    for batch_idx in range(batch_size):
        # Create subdirectory for this batch element
        batch_dir = os.path.join(output_path, f"{idx}_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)

        # Write mix file
        mix_audio = mix[batch_idx]  # Shape: (channels, samples)
        if mix_audio.dim() == 1:
            mix_audio = mix_audio.unsqueeze(0)

        mix_audio = torch.clamp(mix_audio, -1.0, 1.0)
        mix_filepath = os.path.join(batch_dir, "mix.mp3")
        torchaudio.save(mix_filepath, mix_audio.cpu(), sample_rate, format="mp3")

        # Write source files
        sources_audio = target_sources[batch_idx]  # Shape: (sources, channels, samples)
        num_sources = sources_audio.shape[0]

        for source_idx in range(num_sources):
            source_audio = sources_audio[source_idx]  # Shape: (channels, samples)
            if source_audio.dim() == 1:
                source_audio = source_audio.unsqueeze(0)

            source_audio = torch.clamp(source_audio, -1.0, 1.0)
            source_filepath = os.path.join(batch_dir, f"source_{source_idx}.mp3")
            torchaudio.save(
                source_filepath, source_audio.cpu(), sample_rate, format="mp3"
            )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: Config):
    log.info("Info level message")

    cfg: Config = OmegaConf.merge(defaults, cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg_dict))
    cfg.training.monitoring.wandb = None
    trainer = Trainer(cfg)

    out_dir = "./outputs"
    cnt_train = 0  # segments train
    cnt_validation = 0  # segments validation
    max_writes = 64

    # Measure train loader performance
    train_start_time = time.time()
    with torch.inference_mode():
        for idx, (x, y) in enumerate(trainer.train_loader):
            cnt_train += x.shape[0]
    train_end_time = time.time()

    # Measure validation loader performance
    val_start_time = time.time()
    if trainer.validation_loader:
        with torch.inference_mode():
            for idx, (x, y) in enumerate(trainer.validation_loader):
                cnt_validation += x.shape[0]
    val_end_time = time.time()

    sr = cfg.dataset.sample_rate
    segment_len = cfg.dataset.segment_length
    dur_train = cnt_train * segment_len / sr
    dur_validation = cnt_validation * segment_len / sr

    # Calculate performance metrics
    train_total_time = train_end_time - train_start_time
    val_total_time = val_end_time - val_start_time
    train_ms_per_segment = train_total_time * 1000 / cnt_train if cnt_train > 0 else 0
    val_ms_per_segment = (
        val_total_time * 1000 / cnt_validation if cnt_validation > 0 else 0
    )

    def format_duration(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

    print(f"train: {cnt_train} segments, {format_duration(dur_train)}")
    print(f"validation: {cnt_validation} segments, {format_duration(dur_validation)}")

    print(
        f"train load time: {train_total_time:.2f} seconds, {train_ms_per_segment:.2f} ms/segment"
    )
    print(
        f"validation load time: {val_total_time:.2f} seconds, {val_ms_per_segment:.2f} ms/segment"
    )

    if cfg.training.stft_device == "cpu":
        print("stft on cpu")
        istft = ISTFTModule(cfg.model.stft)

    # write segments to disk (for inspection)
    cnt = 0
    with torch.inference_mode():
        for idx, (x, y) in enumerate(trainer.train_loader):
            if cnt >= max_writes:
                break
            if cfg.training.stft_device == "cpu":
                x = istft(x)
                y = istft(y)
            write(out_dir + "/train", idx, x, y, cfg.dataset.sample_rate)
            cnt += x.shape[0]

    cnt = 0
    if trainer.validation_loader:
        with torch.inference_mode():
            for idx, (x, y) in enumerate(trainer.validation_loader):
                if cnt >= max_writes:
                    break
                if cfg.training.stft_device == "cpu":
                    x = istft(x)
                    y = istft(y)
                write(out_dir + "/validation", idx, x, y, cfg.dataset.sample_rate)
                cnt += x.shape[0]


if __name__ == "__main__":
    run()
