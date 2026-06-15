import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from mxsep.cfg import ModelConfig
from mxsep.data.utils import allowed_audio_extensions
from mxsep.models import MusicSourceSeparationModel

logger = logging.getLogger(__name__)


class StreamingChunkDataset(IterableDataset):
    """
    Iterable dataset that streams chunks from multiple audio files on the fly.
    For each file it loads the audio, splits it into overlapping chunks for every pass,
    and yields (metadata, chunk) sequentially. This avoids storing all chunks in memory.
    """
    def __init__(
        self,
        audio_files: List[Path],
        sample_rate: int,
        segment_length: int,
        crossfade_duration: int,   # in ms
        nb_passes: int,
    ):
        super().__init__()
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.crossfade_length = sample_rate * crossfade_duration // 1000
        self.nb_passes = nb_passes

    def _load_audio(self, path: Path) -> np.ndarray:
        """Load audio, resample if needed, return (channels, samples)."""
        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        if sr != self.sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
        return data

    def __iter__(self) -> Iterator[Tuple[dict, np.ndarray]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # For simplicity we use num_workers=0; if changed, handle partition.
            raise RuntimeError("StreamingChunkDataset does not support multiple workers")

        """Split audio files into overlapping chunks for processing"""
        for audio_file in self.audio_files:
            audio = self._load_audio(audio_file)
            total_length = audio.shape[-1]
            shift = (self.segment_length - self.crossfade_length) // self.nb_passes

            for pass_id in range(self.nb_passes):
                # Build all chunks for this pass (still one pass of one file in memory)
                pass_chunks: List[Tuple[dict, np.ndarray]] = []
                initial_offset = -shift * pass_id
                offset = initial_offset
                reached_end = False

                while not reached_end:
                    # Calculate chunk boundaries
                    chunk_start = max(0, offset)
                    chunk_end = min(offset + self.segment_length, total_length)

                    # Extract chunk from audio data
                    chunk = audio[
                        ..., chunk_start:chunk_end
                    ].copy()  # copy to avoid ref to full audio

                    pad_left = 0
                    pad_right = 0

                    # Pad if necessary
                    if offset < 0:
                        pad_left = abs(offset)

                    if offset + self.segment_length >= total_length:
                        pad_right = (offset + self.segment_length) - total_length
                        reached_end = True

                    if pad_left > 0 or pad_right > 0:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (pad_left, pad_right)),
                            mode="constant",
                            constant_values=0,
                        )

                    metadata = {
                        "input_audio_file": str(audio_file),
                        "pass_id": pass_id,
                        "total_length": total_length,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "pad_left": pad_left,
                        "pad_right": pad_right,
                        # Total number of chunks in this pass (same for every chunk of the pass)
                        "pass_total_chunks": None,  # filled below
                    }
                    pass_chunks.append((metadata, chunk))

                    offset += self.segment_length - self.crossfade_length

                # Now we know the total chunks for this pass, set it in metadata
                pass_len = len(pass_chunks)
                for meta, _ in pass_chunks:
                    meta["pass_total_chunks"] = pass_len

                yield from pass_chunks  # stream chunks of this pass

            # File completely yielded – audio data can be garbage collected


class Separator:
    """Main separation class with streaming, memory‑efficient processing."""

    def __init__(
        self,
        model_path,
        device="cuda",
        batch_size=4,
        tta=None,
        crossfade_duration=20,
        nb_passes=2,
        output_ext="mp3",
    ):
        self.model_path = model_path
        self.device = device
        self.tta = tta
        self.nb_passes = nb_passes
        self.crossfade_duration = crossfade_duration
        self.batch_size = batch_size
        self.output_ext = output_ext
        self._load_checkpoint()

    def separate_dir(self, input_path: Path, output_path: Path):
        # Recursively find all files with supported extensions
        audio_files = [f for f in input_path.rglob("*") if f.suffix[1:].lower() in allowed_audio_extensions]
        self.separate_files(audio_files, input_path, output_path)

    def separate_files(self, audio_input_files: List[Path], input_path: Path, output_path: Path):
        """Separate sources from multiple audio files with streaming."""
        self.model.eval()
        self.model.to(self.device)

        dataset = StreamingChunkDataset(
            audio_files=audio_input_files,
            sample_rate=self.config.sample_rate,
            segment_length=self.config.segment_length,
            crossfade_duration=self.crossfade_duration,
            nb_passes=self.nb_passes,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # required for correct streaming
            collate_fn=self._collate_chunks,
            pin_memory=True,
        )

        # Accumulators per file
        file_state: Dict[str, dict] = {}
        # Single background writer thread to avoid I/O contention
        writer_executor = ThreadPoolExecutor(max_workers=1)

        try:
            with torch.inference_mode():
                for batch_metadata, batch_chunks in dataloader:
                    batch_chunks = batch_chunks.to(self.device)

                    # Model inference: (batch, sources, channels, samples)
                    output = self.model(batch_chunks).cpu().numpy()

                    batch_size = batch_chunks.size(0)
                    for i in range(batch_size):
                        meta = batch_metadata[i]
                        sources = output[i]  # (sources, channels, samples)

                        fname = meta["input_audio_file"]
                        pass_id = meta["pass_id"]
                        source_idx_range = range(output.shape[1])  # all sources

                        # Initialize file state if not present
                        if fname not in file_state:
                            file_state[fname] = {
                                "pass_chunks": defaultdict(list),
                                "pass_expected": {},
                                "completed_passes": set(),
                                "source_sums": defaultdict(lambda: None),
                                "total_passes": self.nb_passes,
                            }

                        state = file_state[fname]
                        # Add chunk data to the corresponding pass
                        state["pass_chunks"][pass_id].append({
                            "chunk": sources,                     # (sources, channels, samples)
                            "chunk_start": meta["chunk_start"],
                            "chunk_end": meta["chunk_end"],
                            "pad_left": meta["pad_left"],
                            "pad_right": meta["pad_right"],
                            "total_length": meta["total_length"],
                        })

                        # If we haven't yet stored expected chunk count for this pass, do it now
                        if pass_id not in state["pass_expected"]:
                            state["pass_expected"][pass_id] = meta["pass_total_chunks"]

                        # Check if this pass is now complete
                        if len(state["pass_chunks"][pass_id]) == state["pass_expected"][pass_id]:
                            self._finalize_pass(fname, pass_id, state, source_idx_range)
                            logger.info(f"Finalized pass {pass_id} for {fname}")

                        # If all passes are complete, finalize the file
                        if len(state["completed_passes"]) == self.nb_passes:
                            self._finalize_file(
                                fname, state, input_path, output_path, writer_executor
                            )
                            del file_state[fname]  # free memory immediately

        finally:
            # Wait for all writes to finish
            writer_executor.shutdown(wait=True)

        logger.info("Separation completed.")

    
    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    
    def _determine_output_file(
        self,
        input_file: Path,
        input_path: Path,
        output_path: Path,
        source_name: str,
    ) -> Path:
        """Determine output file path"""
        file_name = input_file.stem
        output_file = output_path / input_file.relative_to(input_path).with_name(
            f"{file_name}_{source_name}.{self.output_ext}"
        )
        return output_file
    
    def _load_checkpoint(self):
        logger.info(f"Loading checkpoint from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        defaults = OmegaConf.structured(ModelConfig())
        self.config: ModelConfig = OmegaConf.merge(defaults, config)
        self.model = MusicSourceSeparationModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _collate_chunks(self, batch: List[Tuple[dict, np.ndarray]]) -> Tuple[List[dict], torch.Tensor]:
        """Custom collate to keep metadata as a list of dicts."""
        metadata_list, chunks = zip(*batch)
        return list(metadata_list), torch.from_numpy(np.stack(chunks, axis=0))

    def _finalize_pass(self, fname: str, pass_id: int, state: dict, source_idx_range):
        """Join all chunks of one pass and add to running source sums."""
        chunks_data = state["pass_chunks"].pop(pass_id)  # free pass chunks
        joined = self._join_chunks(chunks_data)  # shape: (sources, channels, total_length)

        for src_idx in source_idx_range:
            source_wave = joined[src_idx]  # (channels, total_length)
            if state["source_sums"][src_idx] is None:
                state["source_sums"][src_idx] = source_wave.copy()
            else:
                state["source_sums"][src_idx] += source_wave

        state["completed_passes"].add(pass_id)

    def _finalize_file(self, fname: str, state: dict, input_path: Path, output_path: Path, executor):
        """Average passes, submit write task, and clean up."""
        input_file = Path(fname)
        num_passes = state["total_passes"]

        # Average across passes for each source
        for src_idx, sum_wave in state["source_sums"].items():
            avg_wave = sum_wave / num_passes

            output_file = self._determine_output_file(
                input_file, input_path, output_path,
                self.config.target_sources[src_idx]
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Submit to background writer
            executor.submit(self._write_audio, output_file, avg_wave.T)

    def _write_audio(self, path: Path, data: np.ndarray):
        """Write audio in a worker thread."""
        sf.write(path, data, self.config.sample_rate)
        logger.debug(f"Written {path}")

    def _join_chunks(self, chunks_data: List[Dict]) -> np.ndarray:
        """
        Join overlapping chunks of one pass, returning (sources, channels, total_length).
        Each chunk in chunks_data already contains all sources.
        """
        if not chunks_data:
            return np.array([])

        # Get total length and number of channels
        total_length = chunks_data[0]["total_length"]
        num_sources = chunks_data[0]["chunk"].shape[0]
        num_channels = chunks_data[0]["chunk"].shape[1]

        # Initialize output array and weight array for crossfade
        output = np.zeros((num_sources, num_channels, total_length), dtype=np.float32)
        weights = np.zeros(total_length, dtype=np.float32)

        cross_fade_length = self.config.sample_rate * self.crossfade_duration // 1000

        # Create fade windows
        fade_in = np.linspace(0, 1, cross_fade_length)
        fade_out = np.linspace(1, 0, cross_fade_length)

        for chunk_data in chunks_data:
            chunk = chunk_data["chunk"]  # (sources, channels, samples)
            chunk_start = chunk_data["chunk_start"]
            chunk_end = chunk_data["chunk_end"]
            pad_left = chunk_data["pad_left"]
            pad_right = chunk_data["pad_right"]

            if pad_left > 0:
                chunk = chunk[:, :, pad_left:]
            if pad_right > 0:
                chunk = chunk[:, :, :-pad_right]

            # Calculate actual chunk length (excluding padding)
            actual_length = chunk_end - chunk_start

            assert actual_length == chunk.shape[-1]

            # Create window for this chunk
            window = np.ones(actual_length, dtype=np.float32)

            # Apply fade in at the beginning (if not the first chunk at position 0)
            if chunk_start > 0 and actual_length >= cross_fade_length:
                window[:cross_fade_length] = fade_in[
                    : min(cross_fade_length, actual_length)
                ]
            elif chunk_start > 0:
                window[:actual_length] = fade_in[:actual_length]

            # Apply fade out at the end (if not the last chunk)
            if chunk_end < total_length and actual_length >= cross_fade_length:
                window[-cross_fade_length:] = fade_out[
                    -min(cross_fade_length, actual_length) :
                ]
            elif chunk_end < total_length:
                window[-actual_length:] = fade_out[-actual_length:]

            # Apply window to all sources and channels
            output[:, :, chunk_start:chunk_end] += chunk * window[np.newaxis, np.newaxis, :]
            weights[chunk_start:chunk_end] += window

        # Normalize by weights (avoid division by zero)
        weights = np.maximum(weights, 1e-8)
        output /= weights[np.newaxis, np.newaxis, :]
        
        return output
