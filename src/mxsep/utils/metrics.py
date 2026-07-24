import torch


def calculate_sdr(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Signal-to-Distortion Ratio (SDR) between the predicted and target signals.

    Args:
        pred (torch.Tensor): The predicted signal. shape = (batch_size, nb_sources, num_channels, num_samples)
        target (torch.Tensor): The target signal. shape = (batch_size, nb_sources, num_channels, num_samples)

    Returns:
        sdr torch.Tensor: The calculated SDR value per source shape = (nb_sources)
    """
    sdr = _sdr(pred, target) # shape = (batch_size, nb_sources)

    # Mean over batches, keep sources dimension
    sdr_per_source = torch.mean(sdr, dim=0)

    return sdr_per_source


def _sdr(references: torch.Tensor, estimates: torch.Tensor) -> torch.Tensor:
    """
    Compute Signal-to-Distortion Ratio (SDR) for one or more audio tracks.

    SDR is a measure of how well the predicted source (estimate) matches the reference source.
    It is calculated as the ratio of the energy of the reference signal to the energy of the error
    (difference between reference and estimate).
    Return SDR in decibels (dB)

    Parameters:
    ----------
    references : torch.Tensor
        A torch tensor of shape (..., num_channels, num_samples)
        num_channels is the number of channels (e.g., 1 for mono, 2 for stereo), and num_samples is the length of the audio signal.

    estimates : torch.Tensor
        A torch tensor of shape (..., num_channels, num_samples) representing the estimated sources.

    Returns:
    -------
    torch.Tensor
        A tensor containing the SDR values for each source.
    """
    eps = 1e-8  # to avoid numerical errors

    # Sum over the last two dimensions (channels and samples)
    num = torch.sum(torch.square(references), dim=(-2, -1))
    den = torch.sum(torch.square(references - estimates), dim=(-2, -1))

    num = num + eps
    den = den + eps

    return 10 * torch.log10(num / den)
