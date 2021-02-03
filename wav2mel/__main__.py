#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

from wav2mel import create_stft, wav2mel

_LOGGER = logging.getLogger("wav2mel")

# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog="wav2mel")
    parser.add_argument("wav", nargs="*", help="Path(s) to WAV file(s)")
    parser.add_argument("--output-dir", help="Path to output directory")

    # STFT settings
    parser.add_argument("--filter-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--mel-channels", type=int, default=80)
    parser.add_argument("--sampling-rate", type=int, default=22050)
    parser.add_argument("--mel-fmin", type=float, default=0.0)
    parser.add_argument("--mel-fmax", type=float, default=8000.0)

    # Normalization
    parser.add_argument("--max-wav-value", type=float, default=32768.0)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio normalization by max-wav-value",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    stft = create_stft(
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.mel_channels,
        sampling_rate=args.sampling_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
    )

    import scipy.io.wavfile
    import numpy as np

    if args.wav:
        # Convert to paths
        args.wav = [Path(p) for p in args.wav]

        if args.output_dir:
            args.output_dir = Path(args.output_dir)
        else:
            args.output_dir = Path.cwd()

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Process WAVs
        num_wavs = 0
        for wav_path in args.wav:
            _LOGGER.debug("Processing %s", wav_path)
            sample_rate, wav_array = scipy.io.wavfile.read(wav_path)
            assert (
                sample_rate == args.sampling_rate
            ), f"{wav_path} has sample rate {sample_rate}, expected {args.sampling_rate}"

            wav_array = wav_array.astype(np.float32)
            if not args.no_normalize:
                wav_array /= args.max_wav_value

            mel_array = wav2mel(wav_array, stft=stft)

            # Save mel
            mel_path = args.output_dir / ((wav_path.stem) + ".npy")
            with open(mel_path, "wb") as mel_file:
                np.save(mel_file, mel_array, allow_pickle=True)

            _LOGGER.debug("Wrote %s", mel_path)
            num_wavs += 1

        _LOGGER.info("Done. Wrote %s mel(s) to %s", num_wavs, args.output_dir)
    else:
        # Read from stdin, write to stdout
        if os.isatty(sys.stdin.fileno()):
            print("Reading WAV data from stdin...", file=sys.stderr)

        sample_rate, wav_array = scipy.io.wavfile.read(sys.stdin.buffer)
        assert (
            sample_rate == args.sampling_rate
        ), f"WAV has sample rate {sample_rate}, expected {args.sampling_rate}"

        wav_array = wav_array.astype(np.float32)
        if not args.no_normalize:
            wav_array /= args.max_wav_value

        mel_array = wav2mel(wav_array, stft=stft)

        # Write mel to stdout
        np.save(sys.stdout.buffer, mel_array, allow_pickle=True)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
