import typing
from dataclasses import dataclass

import librosa
import numpy as np
from dataclasses_json import DataClassJsonMixin


@dataclass
class AudioSettings(DataClassJsonMixin):
    """Settings for wav <-> mel"""

    # STFT settings
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 256
    mel_channels: int = 80
    sample_rate: int = 22050
    mel_fmin: float = 0.0
    mel_fmax: typing.Optional[float] = 8000.0
    ref_level_db: float = 20.0
    spec_gain: float = 1.0

    # Normalization
    signal_norm: bool = True
    min_level_db: float = -100.0
    max_norm: float = 4.0
    clip_norm: bool = True
    symmetric_norm: bool = True

    def __post_init__(self):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2

        # Compute mel bases
        self._mel_basis = librosa.filters.mel(
            self.sample_rate,
            self.filter_length,
            n_mels=self.mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )

        self._inv_mel_basis = np.linalg.pinv(self._mel_basis)

    # -------------------------------------------------------------------------
    # Mel Spectrogram
    # -------------------------------------------------------------------------

    def wav2mel(
        self, wav: np.ndarray, trim_silence: bool = False, trim_db: float = 60.0
    ) -> np.ndarray:
        if trim_silence:
            wav = self.trim_silence(wav, trim_db=trim_db)

        linear = self.stft(wav)
        mel_amp = self.linear_to_mel(np.abs(linear))
        mel_db = self.amp_to_db(mel_amp)

        if self.signal_norm:
            mel_db = self.normalize(mel_db)

        return mel_db

    def mel2wav(
        self, mel_db: np.ndarray, num_iters: int = 60, power: float = 1.0
    ) -> np.ndarray:
        """Converts melspectrogram to waveform using Griffim-Lim"""
        if self.signal_norm:
            mel_db = self.denormalize(mel_db)

        mel_amp = self.db_to_amp(mel_db)
        linear = self.mel_to_linear(mel_amp) ** power

        return self.griffin_lim(linear, num_iters=num_iters)

    def linear_to_mel(self, linear: np.ndarray) -> np.ndarray:
        """Linear spectrogram to mel amp"""
        return np.dot(self._mel_basis, linear)

    def mel_to_linear(self, mel_amp: np.ndarray) -> np.ndarray:
        """Mel amp to linear spectrogram"""
        return np.maximum(1e-10, np.dot(self._inv_mel_basis, mel_amp))

    def amp_to_db(self, mel_amp: np.ndarray) -> np.ndarray:
        return self.spec_gain * np.log10(np.maximum(1e-5, mel_amp))

    def db_to_amp(self, mel_db: np.ndarray) -> np.ndarray:
        return np.power(10.0, mel_db / self.spec_gain)

    # -------------------------------------------------------------------------
    # STFT
    # -------------------------------------------------------------------------
    def stft(self, wav: np.ndarray) -> np.ndarray:
        """Waveform to linear spectrogram"""
        return librosa.stft(
            y=wav,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode="reflect",
        )

    def istft(self, linear: np.ndarray) -> np.ndarray:
        """Linear spectrogram to waveform"""
        return librosa.istft(
            linear, hop_length=self.hop_length, win_length=self.win_length
        )

    def griffin_lim(self, linear: np.ndarray, num_iters: int = 60) -> np.ndarray:
        """Linear spectrogram to waveform using Griffin-Lim"""
        angles = np.exp(2j * np.pi * np.random.rand(*linear.shape))
        linear_complex = np.abs(linear).astype(np.complex)
        audio = self.istft(linear_complex * angles)

        for _ in range(num_iters):
            angles = np.exp(1j * np.angle(self.stft(audio)))
            audio = self.istft(linear_complex * angles)

        return audio

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def normalize(self, wav: np.ndarray) -> np.ndarray:
        """Put values in [0, max_norm] or [-max_norm, max_norm]"""
        wav_norm = ((wav - self.ref_level_db) - self.min_level_db) / (
            -self.min_level_db
        )
        if self.symmetric_norm:
            # Symmetric norm
            wav_norm = ((2 * self.max_norm) * wav_norm) - self.max_norm
            if self.clip_norm:
                wav_norm = np.clip(wav_norm, -self.max_norm, self.max_norm)
        else:
            # Asymmetric norm
            wav_norm = self.max_norm * wav_norm
            if self.clip_norm:
                wav_norm = np.clip(wav_norm, 0, self.max_norm)

        return wav_norm

    def denormalize(self, audio: np.ndarray) -> np.ndarray:
        """Pull values out of [0, max_norm] or [-max_norm, max_norm]"""
        if self.symmetric_norm:
            # Symmetric norm
            if self.clip_norm:
                audio_denorm = np.clip(audio, -self.max_norm, self.max_norm)

            audio_denorm = (
                (audio_denorm + self.max_norm)
                * -self.min_level_db
                / (2 * self.max_norm)
            ) + self.min_level_db
        else:
            # Asymmetric norm
            if self.clip_norm:
                audio_denorm = np.clip(audio, 0, self.max_norm)

            audio_denorm = (
                audio_denorm * -self.min_level_db / self.max_norm
            ) + self.min_level_db

        audio_denorm += self.ref_level_db

        return audio_denorm

    # -------------------------------------------------------------------------
    # Silence Trimming
    # -------------------------------------------------------------------------

    def trim_silence(
        self,
        wav: np.ndarray,
        trim_db: float = 60.0,
        margin_sec: float = 0.01,
        keep_sec: float = 0.1,
    ):
        """
        Trim silent parts with a threshold and margin.
        Keep keep_sec seconds on either side of trimmed audio.
        """
        margin = int(self.sample_rate * margin_sec)
        wav = wav[margin:-margin]
        _, trim_index = librosa.effects.trim(
            wav,
            top_db=trim_db,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )

        keep_samples = int(self.sample_rate * keep_sec)
        trim_start, trim_end = (
            max(0, trim_index[0] - keep_samples),
            min(len(wav), trim_index[1] + keep_samples),
        )

        return wav[trim_start:trim_end]
