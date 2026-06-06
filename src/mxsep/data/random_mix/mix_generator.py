import math
import random
from itertools import zip_longest
from typing import Iterator, List, Dict, Generator, Set

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from mxsep.cfg import DatasetConfig, MixingStrategy
from mxsep.data.augmentation import DynamicTransformFactory
from mxsep.data.random_mix import Segment, Augmentation, Mix
from mxsep.data.random_mix.mix_loader import _load_audio_segment
from mxsep.data.utils import (
    iter_audio_files,
    iter_audio_files_orphans,
    iter_audio_files_per_song,
    is_silence,
)


def calculate_stem_selection(
    dataset_config: DatasetConfig, split: str = "train"
) -> Dict[str, List[float]]:
    """
    Calculate stem selection probabilities based on co-occurrence of non-silent segments.

    Args:
        dataset_config: Configuration for the dataset
        split: Dataset split to process (default: "train")

    Returns:
        Dictionary mapping stem names to probability lists

    Raises:
        ValueError: If no valid segments are found
    """
    # Use defaultdict to avoid KeyError for new indices
    segment_counts: Dict[int, Dict[str, int]] = {}
    stems_found: Set[str] = set()
    total_positions = 0
    position_idx = 0

    for audio_files in iter_audio_files_per_song(dataset_config, split=split):
        # Process each stem's segments and collect them by index
        all_stem_segments = [
            list(
                process_segments_(
                    audio_file["path"],
                    audio_file["stem"],
                    dataset_config,
                    keep_raw_segments=True,
                )
            )
            for audio_file in audio_files
        ]

        # Zip segments across stems (handling potential length mismatches)
        for segments_at_position in  zip_longest(*all_stem_segments, fillvalue=None):
            if position_idx not in segment_counts:
                segment_counts[position_idx] = {}

            for segment in segments_at_position:
                if segment is None:
                    continue

                try:
                    audio, sr = _load_audio_segment(
                        segment.path,
                        segment.offset,
                        segment.length,
                    )
                except Exception as e:
                    print(f"Failed to load segment {segment.path}: {e}")
                    continue

                if not is_silence(
                    audio, sr, dataset_config.random_mix.silence_threshold_dbfs
                ):
                    stem_name = segment.stem
                    segment_counts[position_idx][stem_name] = (
                        segment_counts[position_idx].get(stem_name, 0) + 1
                    )
                    stems_found.add(stem_name)
                # else:
                #     sf.write(
                #         f"./outputs/silence/skipped_silence_{audio_files[0]['song']}_{segment.stem}_{segment.offset}.wav",
                #         audio.T,
                #         sr,
                #     )

            if segment_counts[position_idx]:  # Only count non-empty positions
                total_positions += 1

            position_idx += 1

    if total_positions == 0:
        raise ValueError("No valid segments found in the dataset")

    # Calculate stem selection probabilities
    stem_selection = _calculate_selection_probabilities(
        segment_counts, stems_found, total_positions
    )

    return stem_selection


def _calculate_selection_probabilities(
    segment_counts: Dict[int, Dict[str, int]],
    stems_found: Set[str],
    total_positions: int,
) -> Dict[str, List[float]]:
    """
    Calculate hierarchical selection probabilities for each stem.

    The probabilities represent the chance of selecting a stem given that
    we want at least n simultaneous occurrences of that stem.
    """
    stem_selection: Dict[str, List[float]] = {}

    for stem in stems_found:
        probabilities = []
        min_occurrences = 1
        prev_probability = 1.0

        while True:
            # Count positions where this stem appears at least min_occurrences times
            count = sum(
                1
                for position_counts in segment_counts.values()
                if position_counts.get(stem, 0) >= min_occurrences
            )

            if count == 0:
                break

            current_probability = count / total_positions

            # Calculate conditional probability
            if prev_probability > 0:
                conditional_probability = current_probability / prev_probability
            else:
                conditional_probability = 0.0
                print(
                    f"Zero probability encountered for stem '{stem}' "
                    f"at occurrence level {min_occurrences}"
                )

            probabilities.append(conditional_probability)
            prev_probability = current_probability
            min_occurrences += 1

        stem_selection[stem] = probabilities

    return stem_selection

def determine_end_(
    pos, segment_length, total_length, min_segment_length, random_first_segment=False
):
    if pos == 0 and random_first_segment:
        # we choose a random length for the first segments this will add more randomness to the mix and avoid each epoch
        # always starting with the same segment length which can be beneficial for training
        first_segment_length = random.randint(min_segment_length, segment_length)
        return min(first_segment_length, total_length)
    else:
        return min(pos + segment_length, total_length)


def process_segments_(
    path, stem, dataset_config: DatasetConfig, keep_raw_segments=False
) -> Iterator[Segment]:
    """

    :param path: Audio file path
    :param stem: Stem name
    :param dataset_config: Configuration of the dataset
    :param keep_raw_segments: Default False, True to keep raw segments (no augmentations, silence removals or first position shifts, only pan last segment)
    :return: Segments
    """
    # print(f"Processing {path}")
    audiodata, sr = sf.read(path, dtype="float32")
    audiodata = audiodata.T
    segment_length = dataset_config.segment_length
    min_segment_length = dataset_config.random_mix.min_segment_length
    if sr != dataset_config.sample_rate:
        segment_length = int(segment_length * sr / dataset_config.sample_rate)
        min_segment_length = int(min_segment_length * sr / dataset_config.sample_rate)

    pos = 0
    total_length = len(audiodata[-1])
    while pos < (total_length - min_segment_length):
        target_length = segment_length
        end = determine_end_(
            pos,
            segment_length,
            total_length,
            min_segment_length,
            random_first_segment=not keep_raw_segments,
        )
        audio_segment = audiodata[..., pos:end]

        # skip silent segments (ignore if keep_raw_segments).
        if not keep_raw_segments and is_silence(audio_segment, sr, dataset_config.random_mix.silence_threshold_dbfs):
            pos = end
            continue

        segment = Segment(path=str(path), stem=stem, offset=pos, length=end - pos)

        if not keep_raw_segments:
            augmentations_for_stem = [
                augmentation
                for augmentation in dataset_config.random_mix.stem_augmentations
                if not augmentation.apply_only_to or stem in augmentation.apply_only_to
            ]

            for augmentation_config in augmentations_for_stem:
                init_params = (
                    OmegaConf.to_container(augmentation_config.settings)
                    | {"p": augmentation_config.p}
                    | OmegaConf.to_container(augmentation_config.randomize)
                )
                transform = DynamicTransformFactory.create(
                    augmentation_config.transformer, init_params
                )
                transform.randomize_parameters(audio_segment, sr)
                if transform.parameters["should_apply"]:
                    augmentation = Augmentation.create_from(
                        augmentation_config, transformer=transform
                    )
                    segment.augmentations.append(augmentation)
                    if augmentation_config.transformer == "TimeStretchPitchShift":
                        rate = transform.parameters["rate"]
                        target_length = math.ceil(target_length * rate)
                        if not (pos == 0 and end < target_length):
                            end = min(pos + target_length, total_length)
                            audio_segment = audiodata[..., pos:end]
                            segment.length = end - pos

        if segment.length < target_length:
            if pos == 0:
                padding_position = "start"
            else:
                padding_position = "end"
            padding_augm = Augmentation(
                transformer="AdjustDuration",
                settings={
                    "duration_samples": target_length,
                    "padding_position": padding_position,
                },
            )
            segment.augmentations.insert(0, padding_augm)

        yield segment
        pos = end


def generate_mix(epoch: int, dataset_config: DatasetConfig) -> Iterator[Mix]:
    """Generate a random mix from config"""
    if dataset_config.random_mix.stem_selection:
        stem_selection = dataset_config.random_mix.stem_selection
    else:
        stem_selection = calculate_stem_selection(dataset_config)
        dataset_config.random_mix.stem_selection = stem_selection
        print(f"Calculated stem selection: {stem_selection}")

    stems = stem_selection.keys()

    segments = {stem: [] for stem in stems}

    for result in iter_audio_files(dataset_config, split="train"):
        file_path = result["path"]
        stem = result["stem"]
        for segment in process_segments_(file_path, stem, dataset_config):
            segments[stem].append(segment)

    for stem in segments.keys():
        random.shuffle(segments[stem])

    indexes = {stem: 0 for stem in stems}
    segments_remaining = {stem: len(segments[stem]) for stem in stems}

    nb_mixes = 0

    while True:
        mix = Mix()

        for stem, probs in stem_selection.items():
            # Stop if we've exhausted all segments from this stem
            if (
                dataset_config.random_mix.mix_strategy
                == MixingStrategy.exhaust_all_and_drop_exhausted
                and segments_remaining[stem] == 0
            ):
                continue

            for prob in probs:
                if random.random() < prob:
                    idx = indexes[stem]
                    mix.segments.append(segments[stem][idx])

                    # Update index with modulo when stem segments is exhausted. common stems
                    if segments_remaining[stem] > 0:
                        indexes[stem] = (idx + 1) % len(segments[stem])
                        segments_remaining[stem] -= 1
                else:
                    break

        for augmentation_config in dataset_config.random_mix.mix_augmentations:
            init_params = (
                OmegaConf.to_container(augmentation_config.settings)
                | {"p": augmentation_config.p}
                | OmegaConf.to_container(augmentation_config.randomize)
            )

            transform = DynamicTransformFactory.create(
                augmentation_config.transformer, init_params
            )
            # create dummy segment of target_length to randomize parameters based on the augmentation config
            # todo create real mix
            dummy_mix = np.zeros((2, dataset_config.segment_length), dtype=np.float32)
            transform.randomize_parameters(dummy_mix, dataset_config.sample_rate)
            if transform.parameters["should_apply"]:
                augmentation = Augmentation.create_from(
                    augmentation_config, transformer=transform
                )
                mix.mix_augmentations.append(augmentation)

        yield mix
        nb_mixes += 1

        mix_strategy = dataset_config.random_mix.mix_strategy

        # Stop when all stem segments have been seen
        if (
            mix_strategy == MixingStrategy.exhaust_all_and_drop_exhausted
            or mix_strategy == MixingStrategy.exhaust_all_and_recycle_exhausted
        ):
            if all(remaining == 0 for remaining in segments_remaining.values()):
                break
        elif mix_strategy == MixingStrategy.stop_when_any_stem_exhausted:
            if any(remaining == 0 for remaining in segments_remaining.values()):
                break
        elif mix_strategy == MixingStrategy.fixed_number:
            if nb_mixes >= dataset_config.random_mix.fixed_number_of_mixes:
                break


def generate_validation_mix(dataset_config: DatasetConfig) -> Iterator[Mix]:
    """Generate a random mix from config"""

    for audio_files in iter_audio_files_per_song(dataset_config, split="validation"):
        # Process each stem's segments and collect them by index
        all_stem_segments = [
            list(
                process_segments_(
                    audio_file["path"],
                    audio_file["stem"],
                    dataset_config,
                    keep_raw_segments=True,
                )
            )
            for audio_file in audio_files
        ]

        # Zip segments across stems (handling potential length mismatches)
        for segments_at_position in zip_longest(*all_stem_segments, fillvalue=None):
            if any(segment is None for segment in segments_at_position):
                print(
                    f"Warning: Mismatch number between nb segments for stems in song {audio_files[0]['song']} "
                )
                segments_at_position = [
                    segment for segment in segments_at_position if segment is not None
                ]
            mix = Mix()
            mix.segments.extend(segments_at_position)
            yield mix


def generate_validation_mix_orphans(dataset_config: DatasetConfig) -> Iterator[Mix]:
    stems = dataset_config.random_mix.stem_selection.keys()
    segments = {stem: [] for stem in stems}

    for result in iter_audio_files_orphans(dataset_config, split="validation"):
        file_path = result["path"]
        stem = result["stem"]
        for segment in process_segments_(
            file_path, stem, dataset_config, keep_raw_segments=True
        ):
            segments[stem].append(segment)

    nb_mixes = min(len(segments[stem]) for stem in stems)

    for idx in range(nb_mixes):
        mix = Mix()
        for stem in stems:
            mix.segments.append(segments[stem][idx])

        yield mix
