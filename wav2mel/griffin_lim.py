#!/usr/bin/env python3
"""Converts JSONL mel spectrograms to WAV audio using griffin-lim"""
import argparse
import io
import json
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
    parser.add_argument("--output_dir", help="Directory to write WAV files")
    parser.add_argument("--iterations", type=int, default=60)

    parser.add_argument(
        "--numpy", action="store_true", help="Standard input is a single numpy file"
    )
    parser.add_argument(
        "--numpy-files",
        action="store_true",
        help="Input is a list of .npy files instead of JSONL",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Don't verify audio settings in JSONL"
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

    # Audio settings to verify in JSON object
    verify_props = [
        "filter_length",
        "hop_length",
        "win_length",
        "mel_channels",
        "sample_rate",
        "mel_fmin",
        "mel_fmax",
        "ref_level_db",
        "spec_gain",
        "signal_norm",
        "min_level_db",
        "max_norm",
        "clip_norm",
        "symmetric_norm",
    ]

    # -------------------------------------------------------------------------

    def process_mel(mel_db: np.ndarray, utt_id: str = ""):
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

    if os.isatty(sys.stdin.fileno()):
        if args.numpy:
            print("Reading numpy array from stdin...", file=sys.stderr)
        elif args.numpy_files:
            print("Reading numpy file names from stdin...", file=sys.stderr)
        else:
            print("Reading JSON from stdin...", file=sys.stderr)

    if args.numpy:
        # stdin is single numpy array
        mel_db = np.load(sys.stdin.buffer, allow_pickle=True).astype(np.float32)
        process_mel(mel_db)
        return

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
    #     "mel_fmax": max frequency for mel,
    #     "ref_level_db": threshold to discard audio,
    #     "spec_gain": gain in amp to db conversion,
    #
    #     "signal_norm": true if mel was normalized,
    #     "max_norm": range of normalization,
    #     "min_level_db": min db for normalization,
    #     "clip_norm": clip during normalization,
    #     "symmetric_norm": normalize in [-max_norm, max_norm] instead of [0, max_norm]
    #   },
    #   "mel": [numpy array of shape (mel_channels, mel_windows)]
    # }
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                # Skip blank lines
                continue

            utt_id = ""

            if args.numpy_files:
                # Load from numpy file
                mel_db = np.load(line, allow_pickle=True).astype(np.float32)
            else:
                # Load from JSONL
                mel_obj = json.loads(line)
                utt_id = mel_obj.get("id", "")

                # Verify audio settings
                if not args.no_verify:
                    audio_obj = mel_obj.get("audio", {})
                    for verify_prop in verify_props:
                        expected_value = getattr(audio_settings, verify_prop)
                        actual_value = audio_obj[verify_prop]
                        assert (
                            expected_value == actual_value
                        ), f"Mismatch for {verify_prop}, expected {expected_value} but got {actual_value}"

                mel_db = np.array(mel_obj["mel"], dtype=np.float32)

            process_mel(mel_db, utt_id)

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
