import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from musep.cfg import DatasetConfig, STFTConfig
from musep.data.random_mix import Mix, load_mix

import logging

from musep.models import STFTModule

logger = logging.getLogger(__name__)


class PredefinedMixDataset(Dataset):

    def __init__(self, config: DatasetConfig, split="train",  stft_cfg: STFTConfig = None):
        """
        Dataset class for loading mixes from JSONL files.
        """
        self.output_target_waveform = stft_cfg # and split == "validation"
        if split == 'train':
            assert config.train.predefined_jsonl_path
            self.jsonl_path: Path = config.train.predefined_jsonl_path
        elif split == 'validation':
            assert config.validation.predefined_jsonl_path
            self.jsonl_path: Path = config.validation.predefined_jsonl_path
        else:
            raise Exception(f"Unknown predefined split: {split}")

        self.sample_rate = config.sample_rate
        self.segment_length = config.segment_length
        self.nb_channels = config.channels
        self.target_sources_map = config.target_sources
        self.examples: List[Mix] = []

        if stft_cfg is not None:
            self.stft = STFTModule(stft_cfg)
        else:
            self.stft = None

        self._cached_epoch_numbers = None
        self._cached_epoch_files = None
        self._current_loaded_jsonl = None

        self.init_epoch()

    def init_epoch(self, epoch: int = 0):
        """
        Load mixes from JSONL file(s).

        Args:
            epoch: Epoch number used when jsonl_path is a directory

        Returns:
            List of loaded Mix objects

        Raises:
            FileNotFoundError: If the specified file/directory doesn't exist
            ValueError: If the path is neither a file nor directory
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            jsonl_file = self._resolve_jsonl_file(epoch)
            self._load_mixes_from_jsonl(jsonl_file)

        except Exception as e:
            logger.error(f"Failed to load mixes: {e}")
            raise

    def _resolve_jsonl_file(self, epoch: int) -> Path:
        """Resolve the JSONL file path with modulo cycling for directories."""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.jsonl_path}")

        if self.jsonl_path.is_file():
            return self.jsonl_path

        if self.jsonl_path.is_dir():
            epoch_numbers, file_paths = self._get_epoch_files()

            if not file_paths:
                raise FileNotFoundError(
                    f"No .jsonl files found in directory: {self.jsonl_path}"
                )

            # Apply modulo cycling
            num_files = len(file_paths)
            selected_idx = epoch % num_files
            selected_file = file_paths[selected_idx]
            selected_epoch = epoch_numbers[selected_idx] if epoch_numbers else selected_idx

            logger.info(
                f"Modulo cycling: requested_epoch={epoch} -> "
                f"selected_epoch={selected_epoch} (index={selected_idx}/{num_files}) -> "
                f"file={selected_file.name}"
            )

            return selected_file

        raise ValueError(f"Path is neither file nor directory: {self.jsonl_path}")

    def _get_epoch_files(self) -> tuple[List[int], List[Path]]:
        """
        Get sorted lists of available epoch numbers and their file paths.
        Caches results for performance.

        Returns:
            Tuple of (epoch_numbers, file_paths) sorted by epoch
        """
        if self._cached_epoch_numbers is not None and self._cached_epoch_files is not None:
            return self._cached_epoch_numbers, self._cached_epoch_files

        if not self.jsonl_path.is_dir():
            return [], []

        epoch_dict = {}
        for file_path in self.jsonl_path.glob("*.jsonl"):
            try:
                epoch_num = int(file_path.stem)
                epoch_dict[epoch_num] = file_path
            except ValueError:
                logger.warning(f"Skipping file with non-integer name: {file_path.name}")
                continue

        if not epoch_dict:
            # Fallback: use alphabetical order with indices
            files = sorted(self.jsonl_path.glob("*.jsonl"))
            epoch_numbers = list(range(len(files)))
            file_paths = files
        else:
            epoch_numbers = sorted(epoch_dict.keys())
            file_paths = [epoch_dict[ep] for ep in epoch_numbers]

        self._cached_epoch_numbers = epoch_numbers
        self._cached_epoch_files = file_paths

        return epoch_numbers, file_paths

    def _load_mixes_from_jsonl(self, jsonl_file):
        if self._current_loaded_jsonl == jsonl_file:
            return

        self._current_loaded_jsonl = jsonl_file
        self.examples: List[Mix] = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    mix = Mix.from_dict(json.loads(line))
                    self.examples.append(mix)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        mix_def: Mix = self.examples[idx]
        # model config target sources todo read from model config

        stems, mix = load_mix(mix_def, self.sample_rate, self.segment_length, self.nb_channels)

        target_sources = {key: [] for key in self.target_sources_map.keys()}

        for target_source, dataset_stems in self.target_sources_map.items():
            for stem_name, segment in stems:  # Fixed: properly unpack each stem
                if stem_name in dataset_stems:
                    target_sources[target_source].append(segment)

            if len(target_sources[target_source]) == 0:
                target_sources[target_source].append(
                    np.zeros((self.nb_channels, self.segment_length), dtype=np.float32))

        target_sources = [np.stack(segments).sum(axis=0) for segments in target_sources.values()]
        target_sources = np.stack(target_sources)  # Shape: (num_target_sources, nb_channels, segment_length)
        target_sources = torch.from_numpy(target_sources).float()
        mix = torch.from_numpy(mix).float()

        if self.stft:
            mix_spec = self.stft(mix.unsqueeze(0)).squeeze(0)
            target_sources_spec = self.stft(target_sources.unsqueeze(0)).squeeze(0)
            if self.output_target_waveform:
                return mix_spec, target_sources_spec, target_sources
            return mix_spec, target_sources_spec
        else:
            return mix, target_sources, 

