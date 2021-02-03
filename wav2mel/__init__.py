import numpy as np

from wav2mel.audio import TacotronSTFT


def create_stft(
    filter_length: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mel_channels: int = 80,
    sampling_rate: int = 22050,
    mel_fmin: float = 0.0,
    mel_fmax: float = 8000.0,
) -> TacotronSTFT:
    return TacotronSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        sampling_rate=sampling_rate,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    )


def wav2mel(wav_array: np.ndarray, stft: TacotronSTFT) -> np.ndarray:
    """Convert WAV audio array to mel spectrogram"""
    return stft.mel_spectrogram(np.expand_dims(wav_array, 0)).squeeze(0)
