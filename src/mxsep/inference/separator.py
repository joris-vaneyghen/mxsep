from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from mxsep.cfg import ModelConfig
from mxsep.data.utils import allowed_audio_extensions
from mxsep.models import MusicSourceSeparationModel


class ChunkDataset(torch.utils.data.Dataset):
    """Dataset for chunks with metadata"""

    def __init__(self, chunks_with_metadata: List[Tuple[dict, np.ndarray]]):
        self.chunks_with_metadata = chunks_with_metadata
        self.metadata_list = [meta for meta, _ in chunks_with_metadata]
        self.chunks = [chunk for _, chunk in chunks_with_metadata]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.metadata_list[idx], torch.from_numpy(self.chunks[idx])


class Separator:
    """Main separation class"""

    def __init__(self, model_path, device='cuda', batch_size=4,  tta=None, crossfade_duration=20, nb_passes=2, output_ext="mp3"):
        self.model_path = model_path
        self.device = device
        self.tta = tta
        self.nb_passes = nb_passes
        self.crossfade_duration = crossfade_duration
        self.batch_size = batch_size
        self.output_ext = output_ext
        self._load_checkpoint()

    def separate_dir(self, input_path: Path, output_path: Path):
        audio_files = []

        # Recursively find all files with supported extensions
        for file in input_path.rglob("*"):
            if file.suffix[1:].lower()  in allowed_audio_extensions:
                audio_files.append(file)

        self.separate_files(audio_files, input_path, output_path)

    def separate_files(
        self, audio_input_files: List[Path], input_path: Path, output_path: Path
    ):
        """Separate sources from multiple audio files"""
        self.model.eval()
        self.model.to(self.device)

        # Collect all chunks with metadata
        all_chunks = list(self._split_chunks(audio_input_files))

        # Create dataset and dataloader
        dataset = ChunkDataset(all_chunks)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Dictionary to store separated sources per file and pass
        # Structure: {input_file: {pass_id: {source_idx: [chunks]}}}
        output_results  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Process all batches
        with torch.inference_mode():
            for batch_metadata, batch_chunks in dataloader:
                batch_chunks = batch_chunks.to(self.device)

                # Model inference
                sources = self.model(
                    batch_chunks
                )  # shape: (batch, sources, channels, samples)

                sources = sources.cpu().numpy()

                batch_size = batch_chunks.size(0)

                # Distribute sources back to respective files
                for i in range(batch_size):
                    input_file = batch_metadata["input_audio_file"][i]
                    pass_id = batch_metadata["pass_id"][i].item()
                    chunk_start = batch_metadata["chunk_start"][i].item()
                    chunk_end = batch_metadata["chunk_end"][i].item()
                    pad_left = batch_metadata["pad_left"][i].item()
                    pad_right = batch_metadata["pad_right"][i].item()
                    total_length = batch_metadata["total_length"][i].item()

                    for source_idx in range(sources.shape[1]):
                        source_chunk = sources[i, source_idx]
                        output_results[input_file][pass_id][source_idx].append(
                            {
                                "chunk": source_chunk,
                                "chunk_start": chunk_start,
                                "chunk_end": chunk_end,
                                "pad_left": pad_left,
                                "pad_right": pad_right,
                                "total_length": total_length,
                            }
                        )

        # Join chunks and average passes for each file
        for input_file in audio_input_files:
            # Get all passes for this file
            passes = output_results[str(input_file)]

            # Combine and average passes per source
            num_sources = len(self.config.target_sources)
            for source_idx in range(num_sources):
                pass_outputs = []

                for pass_id in sorted(passes.keys()):
                    chunks_data = passes[pass_id][source_idx]
                    pass_output = self._join_chunks(chunks_data)
                    pass_outputs.append(pass_output)

                # Average across passes
                if pass_outputs:
                    averaged_output = np.mean(pass_outputs, axis=0)

                    # Save the separated source
                    output_file = self._determine_output_file(
                        input_file,
                        input_path,
                        output_path,
                        self.config.target_sources[source_idx],
                    )
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(output_file, averaged_output.T, self.config.sample_rate)

    def _load_audio(self, path: Path) -> np.ndarray:
        """Load audio file and return as numpy array"""
        audiodata, sr = sf.read(
            path, dtype="float32"
        )

        # Convert to (channels, samples) format
        if audiodata.ndim == 1:
            audiodata = audiodata[np.newaxis, :]
        else:
            audiodata = audiodata.T

        if sr!= self.config.sample_rate:
            audiodata = librosa.core.resample(audiodata, orig_sr=sr, target_sr=self.config.sample_rate)
        return audiodata

    def _determine_output_file(
        self,
        input_file: Path,
        input_path: Path,
        output_path: Path,
        source_name: str,
    ) -> Path:
        """Determine output file path"""
        stem = input_file.stem
        output_file = output_path / input_file.relative_to(input_path).with_name(
            f"{stem}_{source_name}.{self.output_ext}"
        )
        stem = output_file.stem
        output_file = output_file.with_name(f"{stem}_{source_name}.{self.output_ext}")
        return output_file

    def _load_checkpoint(self):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        defaults = OmegaConf.structured(ModelConfig())
        self.config: ModelConfig = OmegaConf.merge(defaults, config)
        self.model = MusicSourceSeparationModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _split_chunks(self, audio_files: List[Path]):
        """Split audio files into overlapping chunks for processing"""
        cross_fade_length = self.config.sample_rate * self.crossfade_duration // 1000

        for audio_file in audio_files:
            audiodata = self._load_audio(audio_file)
            segment_length = self.config.segment_length
            total_length = audiodata.shape[-1]

            # Calculate shift between passes
            shift = (segment_length - cross_fade_length) // self.nb_passes

            for pass_id in range(self.nb_passes):
                # Calculate initial offset for this pass
                initial_offset = -shift * pass_id
                offset = initial_offset
                reached_end = False

                while not reached_end:
                    # Calculate chunk boundaries
                    chunk_start = max(0, offset)
                    chunk_end = min(offset + segment_length, total_length)

                    # Extract chunk from audio data
                    chunk = audiodata[..., chunk_start:chunk_end]

                    
                    pad_left = 0
                    pad_right = 0

                    # Pad if necessary
                    if offset < 0:
                        pad_left = abs(offset)
                        

                    if offset + segment_length >= total_length:
                        pad_right = (offset + segment_length) - total_length
                        reached_end = True

                    if pad_left > 0 or pad_right > 0:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (pad_left, pad_right)),
                            mode="constant",
                            constant_values=0,
                        )

                    metadata = {
                        'input_audio_file': str(audio_file),
                        'pass_id': pass_id,
                        'total_length': total_length,
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                        'pad_left': pad_left,
                        'pad_right': pad_right
                    }

                    yield metadata, chunk

                    # Move to next chunk position
                    offset += segment_length - cross_fade_length

    def _join_chunks(self, chunks_data: List[Dict]) -> np.ndarray:
        """Join overlapping chunks back together with crossfade"""
        if not chunks_data:
            return np.array([])

        # Get total length and number of channels
        total_length = chunks_data[0]["total_length"]
        num_channels = chunks_data[0]["chunk"].shape[0]

        # Initialize output array and weight array for crossfade
        output = np.zeros((num_channels, total_length), dtype=np.float32)
        weights = np.zeros(total_length, dtype=np.float32)

        cross_fade_length = self.config.sample_rate * self.crossfade_duration // 1000

        # Create fade windows
        fade_in = np.linspace(0, 1, cross_fade_length)
        fade_out = np.linspace(1, 0, cross_fade_length)

        for chunk_data in chunks_data:
            chunk = chunk_data["chunk"]
            chunk_start = chunk_data["chunk_start"]
            chunk_end = chunk_data["chunk_end"]
            pad_left = chunk_data["pad_left"]
            pad_right = chunk_data["pad_right"]

            if pad_left > 0:
                chunk = chunk[:, pad_left:]
            if pad_right > 0:
                chunk = chunk[:, :-pad_right]

            # Calculate actual chunk length (excluding padding)
            actual_length = chunk_end - chunk_start

            assert actual_length == chunk.shape[1]

            # Create window for this chunk
            window = np.ones(actual_length)

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

            # Add chunk to output with window
            output[:, chunk_start:chunk_end] += (
                chunk * window[np.newaxis, :]
            )
            weights[chunk_start:chunk_end] += window

        # Normalize by weights (avoid division by zero)
        weights = np.maximum(weights, 1e-8)
        output /= weights[np.newaxis, :]

        return output
