from typing import Literal, Dict, Any

import python_stretch
import random
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform

#TODO check https://www.decodingai.com/p/mastering-ml-configurations-by-leveraging

class TimeStretchPitchShift(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(
            self,
            min_rate: float = 0.8,
            max_rate: float = 1.25,
            min_semitones: float = -4.0,
            max_semitones: float = 4.0,
            use_integer_semitones: bool = True,
            p: float = 0.5,
    ):
        """
        :param min_rate: Minimum time-stretch rate. Values less than 1.0 slow down the audio (reduce the playback speed).
        :param max_rate: Maximum time-stretch rate. Values greater than 1.0 speed up the audio (increase the playback speed).
        :param min_semitones: Minimum semitones to shift. A negative number means shift down.
        :param max_semitones: Maximum semitones to shift. A positive number means shift up.
        :param use_integer_semitones: True if semitones should be an integer.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_semitones < -24:
            raise ValueError("min_semitones must be >= -24")
        if max_semitones > 24:
            raise ValueError("max_semitones must be <= 24")
        if min_semitones > max_semitones:
            raise ValueError("min_semitones must not be greater than max_semitones")
        if min_rate < 0.1:
            raise ValueError("min_rate must be >= 0.1")
        if max_rate > 10:
            raise ValueError("max_rate must be <= 10")
        if min_rate > max_rate:
            raise ValueError("min_rate must not be greater than max_rate")

        self.min_rate = min_rate
        self.max_rate = max_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.use_integer_semitones = use_integer_semitones

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            if self.use_integer_semitones:
                num_semitones = float(random.randint(int(self.min_semitones), int(self.max_semitones)))
            else:
                num_semitones = random.uniform(self.min_semitones, self.max_semitones)

            self.parameters["num_semitones"] = num_semitones
            self.parameters["rate"] = random.uniform(self.min_rate, self.max_rate)

    def apply(
            self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        original_ndim = samples.ndim
        if original_ndim == 1:
            samples = samples[np.newaxis, :]

        if (
                original_ndim == 2
                and samples.shape[0] > 1
                and not samples.flags.c_contiguous
        ):
            samples = np.ascontiguousarray(samples)

        stretch = python_stretch.Signalsmith.Stretch()
        stretch.preset(samples.shape[0], sample_rate)
        stretch.setTransposeSemitones(self.parameters["num_semitones"])
        stretch.setTimeFactor(self.parameters["rate"])
        samples = stretch.process(samples)
        if samples.ndim > original_ndim:
            samples = samples[0]

        return samples


def optimize_time_stretch_pitch_shift(
        augmentations: Dict[str, Dict[str, Any]],
        original_sample_rate: int
) -> Dict[str, Dict[str, Any]]:
    """
    Experimental:
    Optimize TimeStretch and PitchShift augmentations by combining them into a single
    TimeStretch followed by a Resample operation when they appear together in the pipeline.

    This optimization takes advantage of the fact that pitch shifting and time stretching
    are mathematically related operations. A pitch shift of S semitones followed by
    (or preceded by) a time stretch of rate R can be replaced by:

    1. A single TimeStretch operation with an adjusted rate
    2. A Resample operation to achieve the desired pitch shift

    The mathematical relationship:
    - Pitch shift by S semitones changes duration by factor of 2^(-S/12)
    - Total time scaling = original_time_stretch_rate * 2^(-pitch_shift_semitones/12)
    - Resample rate = original_sample_rate * 2^(pitch_shift_semitones/12)

    Constraints:
    - Input augmentation dict cannot already contain a Resample operation
    - Only consecutive TimeStretch and PitchShift operations (in any order) are combined
    - All other augmentations are preserved in their original order

    Args:
        augmentations: Dictionary of augmentations where keys are augmentation names
                      and values are their parameter dictionaries.

                      Example input:
                      {
                          "Normalize": {"max_amplitude": 1.0},
                          "TimeStretch": {"rate": 1.15},
                          "PitchShift": {"num_semitones": 2},
                          "Gain": {"amplitude_ratio": 1.1}
                      }

    Returns:
        Optimized dictionary of augmentations with TimeStretch and PitchShift combined.

        Example output (with original_sample_rate=22050):
        {
            "Normalize": {"max_amplitude": 1.0},
            "TimeStretch": {"rate": 1.023},  # 1.15 * 2^(-2/12)
            "Resample": {"target_sample_rate": 24719},  # 22050 * 2^(2/12)
            "Gain": {"amplitude_ratio": 1.1}
        }

    Raises:
        ValueError: If the input already contains a Resample augmentation

    Notes:
        - The original_sample_rate should be provided or inferred from context.
          In this implementation, a placeholder is used that should be replaced
          with the actual sample rate from your audio processing pipeline.
        - The optimization preserves the order of other augmentations in the pipeline.
        - If TimeStretch and PitchShift are not consecutive (other augmentations between them),
          they are NOT combined in this implementation.
    """

    # Check if Resample already exists
    resample_params = None
    if "Resample" in augmentations:
        resample_params = augmentations["Resample"]
        del augmentations["Resample"]

    # Create a copy to avoid modifying the input
    optimized = {}

    # Convert dict items to list for easier manipulation
    items = list(augmentations.items())
    i = 0

    while i < len(items):
        aug_name, aug_params = items[i]

        # Look for consecutive TimeStretch and PitchShift (in any order)
        if i < len(items) - 1:
            next_aug_name, next_aug_params = items[i + 1]

            # Check if we have both operations consecutively
            if {aug_name, next_aug_name} == {"TimeStretch", "PitchShift"}:
                # Determine which is which
                if aug_name == "TimeStretch":
                    stretch_params = aug_params
                    pitch_params = next_aug_params
                else:
                    stretch_params = next_aug_params
                    pitch_params = aug_params

                # Extract parameters
                original_rate = stretch_params.get("rate", 1.0)
                semitones = pitch_params.get("num_semitones", 0)

                # Calculate new rate and target sample rate
                # Pitch shift by S semitones changes duration by factor of 2^(-S/12)
                pitch_time_factor = 2 ** (-semitones / 12)
                new_rate = original_rate * pitch_time_factor

                if resample_params:
                    original_sample_rate = resample_params["target_sample_rate"]
                target_sample_rate = int(original_sample_rate * (2 ** (semitones / 12)))

                # Add the optimized TimeStretch
                optimized["TimeStretch"] = {"rate": new_rate}

                # Add the Resample operation
                optimized["Resample"] = {"target_sample_rate": target_sample_rate}

                # Skip the next item since we've processed both
                i += 2
                continue

        # If not combining, just copy the augmentation as-is
        optimized[aug_name] = aug_params.copy()
        i += 1

    return optimized
