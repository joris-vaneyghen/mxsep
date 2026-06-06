import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray

class ChannelSwap(BaseWaveformTransform):
    """
    Swap the left and right channels of a stereo audio signal. This transform is only applicable
    """

    supports_multichannel = True

    def __init__(self, p: float = 0.5):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        return samples[::-1, :]