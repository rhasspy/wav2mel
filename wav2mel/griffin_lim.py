#!/usr/bin/env python3
"""Converts mel spectrograms to WAV audio using griffin-lim"""
import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from .audio import AudioSettings
from .utils import add_audio_settings

_LOGGER = logging.getLogger("wav2mel.griffin_lim")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="wav2mel.griffin_lim")
    parser.add_argument("--output-dir", help="Directory to write WAV files")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("mel", nargs="*", help="Mel spectrogram files (.npy)")
    parser.add_argument("--dtype", default="float32", help="numpy data type for mel")
    parser.add_argument(
        "--batch-dim", action="store_true", help="Spectrograms include batch dimension"
    )

    add_audio_settings(parser)

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
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------

    audio_settings = AudioSettings(
        # STFT
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        mel_channels=args.mel_channels,
        sample_rate=args.sample_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        ref_level_db=args.ref_level_db,
        spec_gain=args.spec_gain,
        #
        # Normalization
        signal_norm=not args.no_normalize,
        min_level_db=args.min_level_db,
        max_norm=args.max_norm,
        clip_norm=not args.no_clip_norm,
        symmetric_norm=not args.asymmetric_norm,
    )

    # -------------------------------------------------------------------------

    def process_mel(mel_db: np.ndarray, utt_id: str = ""):
        if args.batch_dim:
            mel_db = mel_db.squeeze(0)

        # Run griffin-lim
        _LOGGER.debug("Mel shape: %s", mel_db.shape)

        wav = audio_settings.mel2wav(mel_db, num_iters=args.iterations)
        duration_sec = len(wav) / audio_settings.sample_rate

        # Save WAV data
        if not utt_id:
            # Use timestamp
            utt_id = str(time.time())

        if args.output_dir:
            # Write to file
            wav_path = args.output_dir / (utt_id + ".wav")
            with open(wav_path, "wb") as wav_file:
                scipy.io.wavfile.write(wav_file, audio_settings.sample_rate, wav)

            _LOGGER.debug(
                "Wrote %s (%s sample(s), %s second(s))",
                wav_path,
                len(wav),
                duration_sec,
            )
        else:
            # Write to stdout
            with io.BytesIO() as wav_file:
                scipy.io.wavfile.write(wav_file, audio_settings.sample_rate, wav)
                sys.stdout.buffer.write(wav_file.getvalue())

            _LOGGER.debug("Wrote (%s sample(s), %s second(s))", len(wav), duration_sec)

    # -------------------------------------------------------------------------

    if args.mel:
        for mel_path in args.mel:
            # Load from numpy file
            _LOGGER.debug("Processing %s", mel_path)
            mel_db = np.load(mel_path, allow_pickle=True).astype(args.dtype)
            process_mel(mel_db, utt_id=Path(mel_path).stem)
    else:
        if os.isatty(sys.stdin.fileno()):
            print("Reading mel from stdin...", file=sys.stderr)

        # stdin is single numpy array
        with io.BytesIO(sys.stdin.buffer.read()) as mel_file:
            mel_db = np.load(mel_file, allow_pickle=True).astype(args.dtype)

        process_mel(mel_db)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
