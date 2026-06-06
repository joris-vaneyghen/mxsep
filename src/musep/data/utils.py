import csv
import os
import re
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict, Optional, Iterator

import numpy as np

from musep.cfg import DatasetConfig

allowed_audio_extensions = ['wav', 'mp3', 'flac', 'ogg', 'aiff', 'aac']


def is_silence(
    audio: np.ndarray,
    sr: int,
    silence_thresh_dbfs: float = -50.0,
    window_ms: int = 20,
    seek_step_ms: int = 10,
    min_non_silence_len_ms: int = 50,
) -> bool:
    """
    Determine whether an audio segment should be considered silent.

    Audio is considered NON-silent if there exists a contiguous
    non-silent region lasting at least min_non_silence_len_ms.

    Parameters
    ----------
    audio
        Shape (samples,) or (channels, samples).
        Floating-point audio normalized to [-1, 1].

    sr
        Sample rate in Hz.

    silence_thresh_dbfs
        RMS threshold in dBFS.

    window_ms
        RMS analysis window size.

    seek_step_ms
        Step between successive analysis windows.

    min_non_silence_len_ms
        Minimum duration of contiguous non-silent audio required
        to classify the segment as non-silent.

    Returns
    -------
    bool
        True if the segment is silent.
        False if a sufficiently long non-silent region exists.
    """

    # Normalize shape to (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    elif audio.ndim != 2:
        raise ValueError(
            "audio must have shape (samples,) or (channels, samples)"
        )

    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(
            "audio must contain floating-point samples normalized to [-1, 1]"
        )

    _, n_samples = audio.shape

    if n_samples == 0:
        return True

    silence_rms_thresh = 10.0 ** (silence_thresh_dbfs / 20.0)

    win = max(1, round(window_ms * sr / 1000))
    hop = max(1, round(seek_step_ms * sr / 1000))

    min_non_silence_samples = max(
        1,
        round(min_non_silence_len_ms * sr / 1000),
    )

    # Short clip: evaluate whole clip once
    if n_samples < win:
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        return rms <= silence_rms_thresh

    last_start = n_samples - win
    starts = range(0, last_start + 1, hop)

    if last_start % hop:
        starts = list(starts) + [last_start]

    current_non_silent_samples = 0

    for start in starts:
        window = audio[:, start:start + win]

        # RMS across all channels.
        # Avoids phase-cancellation bug from channel averaging.
        rms = np.sqrt(
            np.mean(window.astype(np.float64) ** 2)
        )

        if rms > silence_rms_thresh:
            current_non_silent_samples += hop

            if current_non_silent_samples >= min_non_silence_samples:
                return False
        else:
            current_non_silent_samples = 0

    return True

def iter_audio_files_pattern_(
    audio_files_pattern: str, stem: Optional[str] = None
) -> Iterator[Dict]:
    """
    Iterate audio files matching a pattern with stem extraction.

    Args:
        audio_files_pattern: Glob-like pattern with optional {stem} and/or {song} placeholders
                      e.g., '/path/to/{song}/{stem}/*.mp3', '/path/**/{song}/{stem}_*.wav'
        stem: Optional specific stem to filter. If None, matches all stems.

    Yields:
        Dictionary with 'path' (full path string), 'stem' (extracted stem), 'filename' (file name only)
        and optionally 'song' (extracted song) if {song} was in pattern

    """
    print(f"iter {audio_files_pattern}")

    has_song = "{song}" in audio_files_pattern
    has_stem = "{stem}" in audio_files_pattern

    # Convert to glob pattern for initial search
    glob_pattern = audio_files_pattern
    if has_stem and stem:
        glob_pattern = glob_pattern.replace("{stem}", stem)
    elif has_stem:
        glob_pattern = glob_pattern.replace("{stem}", "*")

    if has_song:
        glob_pattern = glob_pattern.replace("{song}", "*")

    # Build regex for extraction
    regex_pattern = re.escape(audio_files_pattern)
    if has_stem:
        regex_pattern = regex_pattern.replace(r"\{stem\}", "(?P<stem>[^/]*)")
    if has_song:
        regex_pattern = regex_pattern.replace(r"\{song\}", "(?P<song>[^/]*)")
    regex_pattern = regex_pattern.replace(r"\*\*", ".*")
    regex_pattern = regex_pattern.replace(r"\*", "[^/]*")
    regex = re.compile(regex_pattern)

    # Find all matching files
    for filepath in (
        Path().glob(glob_pattern)
        if not os.path.isabs(glob_pattern)
        else Path("/").glob(glob_pattern.lstrip("/"))
    ):
        filepath_str = str(filepath.absolute())
        match = regex.match(filepath_str)
        if match and filepath.is_file():
            result = {
                "path": filepath_str,
                "filename": filepath.name,
            }

            # Add stem if present in pattern
            if has_stem:
                result["stem"] = stem if stem else match.group("stem")

            # Add song if present in pattern
            if has_song:
                result["song"] = match.group("song")

            yield result

def iter_audio_files_csv_(csv_file: str|Path, stem: Optional[str] = None) -> Iterator[Dict]:
    """
    Iterate audio files matching a pattern with optional stem extraction.

    Args:
        csv_file: Path to CSV file containing audio file information. Expected columns include 'stem' and 'path'.
        stem: Optional specific stem to filter. If None, matches all stems.

    Yields:
        Dictionary with 'path' (full path string), 'stem' (extracted stem),
        and 'filename' (file name only)

    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        # Auto-detect delimiter (comma or tab)
        sample = f.read(1024)
        f.seek(0)
        delimiter = '\t' if '\t' in sample else ','

        reader = csv.DictReader(f, delimiter=delimiter)

        # Normalize column names (case-insensitive)
        fieldnames = {k.lower(): k for k in reader.fieldnames}

        instr_col = fieldnames.get('instrum', fieldnames.get('instrument', fieldnames.get('stem')))
        path_col = fieldnames.get('path', fieldnames.get('file', fieldnames.get('filepath')))
        song_col = fieldnames.get('song', fieldnames.get('song_id', fieldnames.get('track')))

        if not instr_col or not path_col:
            raise ValueError(
                f"CSV file {csv_file} missing required columns. Expected 'instrum' and 'path'")

        for row in reader:
            audio_file_stem = row[instr_col].strip()
            audio_file_path = row[path_col].strip()

            parent_path = Path(csv_file).parent
            full_path = Path.joinpath(parent_path, audio_file_path)

            # Verify file exists and has valid extension
            if (os.path.exists(full_path)
                    and Path(full_path).suffix[1:].lower() in allowed_audio_extensions):
                if stem is None or audio_file_stem == stem:
                    filepath_str = str(full_path.absolute())
                    result = {
                        'path': filepath_str,
                        'filename': full_path.name,
                        'stem': audio_file_stem,
                    }
                    if song_col:
                        result['song'] = row[song_col].strip()

                    yield result
            else:
                # Optionally log warning about missing files
                print(f"Warning: File not found or invalid extension: {full_path}")



def iter_audio_files_per_song(
    dataset_config: DatasetConfig, split: str = "train"
) -> Iterator[Dict]:
    """
    Group audio files by song and yield each group as a list.

    Args:
        dataset_config: Configuration for the dataset
        split: Dataset split to process ('train', 'validation', 'test')

    Yields:
        List of audio file dictionaries belonging to the same song
    """
    # Group by song, filtering out items without a song
    audio_files = iter_audio_files(dataset_config, split=split)
    files_with_song = (audio_file for audio_file in audio_files if audio_file.get("song"))

    # Group and yield
    for song, stem_audio_files in groupby(files_with_song, key=itemgetter("song")):
        yield list(stem_audio_files)


def iter_audio_files_orphans(
    dataset_config: DatasetConfig, split: str = "train", stem: Optional[str] = None
) -> Iterator[Dict]:
    """
    Yield audio files that are not associated with any song.

    Args:
        dataset_config: Configuration for the dataset
        split: Dataset split to process ('train', 'validation', 'test')
        stem: Optional stem filter for audio files

    Yields:
        Audio file dictionaries that have no 'song' field
    """
    for result in iter_audio_files(dataset_config, split=split, stem=stem):
        if "song" not in result:
            yield result
        

def iter_audio_files(dataset_config:DatasetConfig, split='train', stem: Optional[str] = None) -> Iterator[Dict]:
    """
    Iterate audio files defined in dataset_config with stem extraction.

    Args:
        dataset_config: configuration containing audio file patterns and/or CSV file paths
        stem: Optional specific stem to filter. If None, matches all stems.

    Yields:
        Dictionary with 'path' (full path string), 'stem' (extracted stem),
        and 'filename' (file name only)

    """
    audio_files_pattern = None
    audio_file_csv = None

    if split == 'train':
        audio_files_pattern = dataset_config.train.audio_files_pattern
        audio_file_csv = dataset_config.train.audio_file_csv
    elif split == 'validation':
        audio_files_pattern = dataset_config.validation.audio_files_pattern
        audio_file_csv = dataset_config.validation.audio_file_csv

    if audio_files_pattern:
        for audio_files_pattern in audio_files_pattern:
            print(f"Processing pattern: {audio_files_pattern} with stem: {stem}")
            yield from iter_audio_files_pattern_(audio_files_pattern, stem=stem)

    if audio_file_csv:
        for audio_file_csv in audio_file_csv:
            yield from iter_audio_files_csv_(audio_file_csv, stem=stem)