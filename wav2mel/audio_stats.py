#!/usr/bin/env python3
"""Computes audio statistics from mel spectrograms"""
import argparse
import json
import logging
import os
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from wav2mel.audio import TacotronSTFT

_LOGGER = logging.getLogger("wav2mel.audio_stats")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="wav2mel.audio_stats")
    parser.add_argument("--filter-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--mel-channels", type=int, default=80)
    parser.add_argument("--sampling-rate", type=int, default=22050)
    parser.add_argument("--mel-fmin", type=float, default=0.0)
    parser.add_argument("--mel-fmax", type=float, default=8000.0)
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Convert to paths
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------

    # Cached TacotronSTFT objects by audio settings
    stfts: typing.Dict[AudioSettings, TacotronSTFT] = {}

    if os.isatty(sys.stdin.fileno()):
        print("Reading JSON from stdin...", file=sys.stderr)

    # Read JSON objects from standard input.
    # Each object should have this structure:
    # {
    #   "id": "utterance id (used for output file name)",
    #   "audio": {
    #     "filter_length": length of filter,
    #     "hop_length": length of hop,
    #     "win_length": length of window,
    #     "mel_channels": number of mel channels,
    #     "sample_rate": sample rate of audio,
    #     "mel_fmin": min frequency for mel,
    #     "mel_fmax": max frequency for mel
    #   },
    #   "mel": [numpy array of shape (mel_channels, mel_windows)]
    # }
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                # Skip blank lines
                continue

            mel_obj = json.loads(line)

            # Load audio settings
            audio_obj = mel_obj.get("audio", {})
            audio_settings = AudioSettings(
                filter_length=audio_obj.get("filter_length", args.filter_length),
                hop_length=audio_obj.get("hop_length", args.hop_length),
                win_length=audio_obj.get("win_length", args.win_length),
                mel_channels=audio_obj.get("mel_channels", args.mel_channels),
                sampling_rate=audio_obj.get("sampling_rate", args.sampling_rate),
                mel_fmin=audio_obj.get("mel_fmin", args.mel_fmin),
                mel_fmax=audio_obj.get("mel_fmax", args.mel_fmax),
            )

            # Look up or create STFT
            taco_stft = stfts.get(audio_settings)
            if taco_stft is None:
                # Creat new STFT
                taco_stft = TacotronSTFT(
                    filter_length=audio_settings.filter_length,
                    hop_length=audio_settings.hop_length,
                    win_length=audio_settings.win_length,
                    n_mel_channels=audio_settings.mel_channels,
                    sampling_rate=audio_settings.sampling_rate,
                    mel_fmin=audio_settings.mel_fmin,
                    mel_fmax=audio_settings.mel_fmax,
                )
                stfts[audio_settings] = taco_stft

            # Run griffin-lim
            mel = np.array(mel_obj["mel"])
            _LOGGER.debug("Mel shape: %s", mel.shape)

            mel_decompress = taco_stft.spectral_de_normalize(
                np.expand_dims(mel, 0)
            ).squeeze(0)

            mel_decompress = mel_decompress.transpose()
            spec_from_mel = np.matmul(mel_decompress, taco_stft.mel_basis)
            spec_from_mel = np.expand_dims(spec_from_mel.transpose(), 0)
            spec_from_mel = spec_from_mel * args.mel_scaling

            signal = griffin_lim(
                spec_from_mel[:, :, :-1], taco_stft.stft_fn, n_iters=args.iterations
            ).squeeze(0)

            # Save WAV data
            utt_id = mel_obj.get("id", "")
            if not utt_id:
                # Use timestamp
                utt_id = str(time.time())

            wav_path = args.output_dir / (utt_id + ".wav")
            with open(wav_path, "wb") as wav_file:
                scipy.io.wavfile.write(wav_file, audio_settings.sampling_rate, signal)

            duration_sec = len(signal) / audio_settings.sampling_rate
            _LOGGER.debug(
                "Wrote %s (%s sample(s), %s second(s))",
                wav_path,
                len(signal),
                duration_sec,
            )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


@dataclass(eq=True, frozen=True)
class AudioSettings:
    """Settings needed for griffin-lim"""

    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 256
    mel_channels: int = 80
    sampling_rate: int = 22050
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0


# -----------------------------------------------------------------------------


def griffin_lim(magnitudes, stft_fn, n_iters=60):
    """Create audio signal from mel spectrogram"""
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.shape)))
    angles = angles.astype(np.float32)
    signal = stft_fn.inverse(magnitudes, angles)

    for _ in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles)

    return signal


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
