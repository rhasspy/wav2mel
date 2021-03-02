#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import librosa
import jsonlines
import numpy as np

from wav2mel import create_stft, wav2mel

_LOGGER = logging.getLogger("wav2mel")

# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog="wav2mel")
    parser.add_argument("wav", nargs="*", help="Path(s) to WAV file(s)")
    parser.add_argument("--id", default="", help="Set mel id when using stdin")
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Output numpy file(s) instead of JSONL (see --numpy-dir)",
    )
    parser.add_argument(
        "--numpy-batch-dimension",
        action="store_true",
        help="Include batch dimension in numpy arrays",
    )
    parser.add_argument("--numpy-dir", help="Directory to save numpy file(s)")

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

    if args.numpy:
        if args.numpy_dir:
            args.numpy_dir = Path(args.numpy_dir)
        elif args.wav:
            # Default to current directory
            args.numpy_dir = Path.cwd()

        if args.numpy_dir:
            args.numpy_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------

    stft = create_stft(
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.mel_channels,
        sampling_rate=args.sampling_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
    )

    # Outline a line of JSON for each input file
    writer = jsonlines.Writer(sys.stdout, flush=True)
    output_obj = {
        "id": args.id,
        "audio": {
            "filter_length": args.filter_length,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "mel_channels": args.mel_channels,
            "sample_rate": args.sampling_rate,
            "sample_bytes": 2,
            "samples": 0,
            "channels": 1,
            "mel_fmin": args.mel_fmin,
            "mel_fmax": args.mel_fmax,
            "normalized": not args.no_normalize,
        },
        "mel": [],
    }

    num_wavs = 0
    if args.wav:
        # Convert to paths
        args.wav = [Path(p) for p in args.wav]

        # Process WAVs
        for wav_path in args.wav:
            _LOGGER.debug("Processing %s", wav_path)
            wav_array, _ = librosa.load(wav_path, sr=args.sampling_rate).astype(
                np.float32
            )
            if not args.no_normalize:
                wav_array /= args.max_wav_value

            mel_array = wav2mel(wav_array, stft=stft)

            if args.numpy:
                if args.numpy_batch_dimension:
                    mel_array = np.expand_dims(mel_array, 0)

                # Save to numpy file
                mel_path = args.numpy_dir / ((wav_path.stem) + ".npy")
                np.save(mel_path, mel_array)
            else:
                # Output JSONL
                output_obj["id"] = wav_path.stem
                output_obj["mel"] = mel_array.tolist()
                output_obj["audio"]["samples"] = len(wav_array)

                writer.write(output_obj)

            num_wavs += 1
    else:
        # Read from stdin, write to stdout
        if os.isatty(sys.stdin.fileno()):
            print("Reading WAV data from stdin...", file=sys.stderr)

        wav_array, _ = librosa.load(sys.stdin.buffer, sr=args.sampling_rate).astype(
            np.float32
        )
        if not args.no_normalize:
            wav_array /= args.max_wav_value

        mel_array = wav2mel(wav_array, stft=stft)

        if args.numpy:
            if args.numpy_batch_dimension:
                mel_array = np.expand_dims(mel_array, 0)

            # Write numpy file
            if args.numpy_dir:
                mel_id = args.id or str(time.time())
                mel_path = args.numpy_dir / f"{mel_id}.npy"
                np.save(mel_path, mel_array)
            else:
                # Write to stdout
                np.save(sys.stdout.buffer, mel_array)
        else:
            output_obj["mel"] = mel_array.tolist()
            output_obj["audio"]["samples"] = len(wav_array)
            writer.write(output_obj)

        num_wavs += 1

    _LOGGER.info("Done (%s WAV file(s))", num_wavs)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
