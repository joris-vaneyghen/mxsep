from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from audiomentations.core.transforms_interface import BaseWaveformTransform

from musep.data.augmentation import DynamicTransformFactory
from musep.data.random_mix import Mix, Augmentation


def load_mix(mix_def: Mix, sample_rate: int, segment_length: int, nb_channels: int) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    stems = []
    for segment_def in mix_def.segments:
        segment, original_sr = _load_audio_segment(
            segment_def.path,
            segment_def.offset,
            segment_def.length,

        )
        # todo handle nb channels
        segment = _apply_augmentations(segment, original_sr, sample_rate, segment_length,
                                      segment_def.augmentations)
        # append
        stems.append((segment_def.stem, segment))

    # Create mix (with potential mixing augmentations)
    mix = _create_mix(stems, mix_def, sample_rate)

    return stems, mix

def _load_audio_segment(path:str, offset:int, length:int)->Tuple[np.ndarray, int]:
    audio, sr = sf.read(
        path,
        start=offset,
        frames=length,
        dtype='float32'
    )
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=1)
    return audio.T, sr

def _create_mix(stems:List[Tuple[str, np.ndarray]], mix_def:Mix, sample_rate) -> np.ndarray:
    """Create mix from stem segments"""
    # Sum stems
    stems_segments = [segment for stem, segment in stems]
    mix = np.stack(stems_segments).sum(axis=0)

    # Apply mix-level augmentations
    if mix_def.mix_augmentations:
        mix = _apply_augmentations(mix, sample_rate, sample_rate, mix.shape[-1], mix_def.mix_augmentations)

    return mix


def _apply_augmentations(samples:np.ndarray, original_sr:int, target_sr:int, target_length:int, augmentations:List[Augmentation])->np.ndarray:
    """Apply defined augmentations"""
    sr = original_sr

    if augmentations:
        for augmentation in augmentations:
            transform: BaseWaveformTransform = DynamicTransformFactory.create(augmentation.transformer, augmentation.settings, augmentation.parameters)
            if transform:
                samples = transform(samples=samples, sample_rate=sr)
                if augmentation.transformer == 'Resample':
                    sr = augmentation.parameters["target_sample_rate"]

    if sr != target_sr:
        print(f"Warning: Resampling {sr} to {target_sr}")
        samples = librosa.core.resample(
            samples,
            orig_sr=sr,
            target_sr=target_sr,
        )
    return _ensure_length(samples, target_length)


def _ensure_length(samples:np.ndarray, target_length:int)->np.ndarray:
    # Early return if already correct length
    segment_length = samples.shape[-1]
    if target_length == segment_length:
        return samples

    if abs(target_length - segment_length) > 10:
        print(f"Warning: Adjusting length from {segment_length} to {target_length}")

    # Pad or crop along the last dimension
    if samples.ndim == 2:
        target_shape = (samples.shape[0], target_length)
    else:
        target_shape = (target_length,)

    result = np.zeros(shape=target_shape, dtype=samples.dtype)
    copy_length = min(target_length, segment_length )
    result[..., :copy_length] = samples[..., :copy_length]
    return result






