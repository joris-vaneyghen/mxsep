import numpy as np
import torch


def calculate_sdr(pred: torch.Tensor, target: torch.Tensor, target_sources :list[str]) -> dict[str, float]:
    """
    Calculate the Signal-to-Distortion Ratio (SDR) between the predicted and target signals.

    Args:
        pred (numpy.ndarray): The predicted signal. shape = (batch_size, nb_sources, num_channels, num_samples)
        target (numpy.ndarray): The target signal. shape = (batch_size, nb_sources, num_channels, num_samples)
        target_sources list[str] : traget_sources (names) of the sources to calculate SDR for. nb_sources = len(target_sources)
    
    Returns:
        dict[str,float]: The calculated SDR value per source and mean SRD.
    
    """
    metrics = {}
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    sdr = _sdr(pred, target)
    if len(target_sources) > 1:
        sdr_per_source = np.mean(sdr, axis=1) # mean of batches (keep sources)
        for sdr, source in zip(sdr_per_source, target_sources):
            metrics['sdr_' + source] = sdr

    metrics['sdr'] = np.mean(sdr)

    return metrics

def _sdr(references: np.ndarray, estimates: np.ndarray) ->  np.ndarray:
    """
    Compute Signal-to-Distortion Ratio (SDR) for one or more audio tracks.

    SDR is a measure of how well the predicted source (estimate) matches the reference source.
    It is calculated as the ratio of the energy of the reference signal to the energy of the error (difference between reference and estimate).
    Return SDR in decibels (dB)
    Parameters:
    ----------
    references : np.ndarray
        A  numpy array of shape (..., num_channels, num_samples), where num_sources is the number of sources,
        num_channels is the number of channels (e.g., 1 for mono, 2 for stereo), and num_samples is the length of the audio signal.

    estimates : np.ndarray
        A numpy array of shape (...,  num_sources, num_channels, num_samples) representing the estimated sources.

    Returns:
    -------
    np.ndarray
        A 1D numpy array containing the SDR values for each source.
    """
    eps = 1e-8  # to avoid numerical errors
    num = np.sum(np.square(references), axis=(-2, -1))
    den = np.sum(np.square(references - estimates), axis=(-2, -1))
    num += eps
    den += eps
    return 10 * np.log10(num / den)



# def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor):
#     """
#     Calculate the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between the predicted and target signals.
#
#     Args:
#         pred (numpy.ndarray): The predicted signal.
#         target (numpy.ndarray): The target signal.
#
#     Returns:
#         float: The calculated SDR value.
#
#     """
#     return _sdr(pred, target)