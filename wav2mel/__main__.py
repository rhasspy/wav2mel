#!/usr/bin/env python3
import argparse
import dataclasses
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import jsonlines
import librosa
import numpy as np

from .audio import AudioSettings
from .utils import add_audio_settings

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
    parser.add_argument("--audio-config", help="Write JSON audio config to file")

    add_audio_settings(parser)

    # Silence trimming
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-db", type=float, default=60.0)

    # Miscellaneous
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

        # Disable numba logging
        logging.getLogger("numba").setLevel(logging.WARNING)
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

    if args.audio_config:
        args.audio_config = Path(args.audio_config)

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

    if args.audio_config:
        # Save audio config as JSON
        args.audio_config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.audio_config, "w") as config_file:
            json.dump(audio_settings.to_dict(), config_file, indent=4)

    # Outline a line of JSON for each input file
    writer = jsonlines.Writer(sys.stdout, flush=True)
    output_obj = {
        "id": args.id,
        "audio": {
            # Audio settings
            **dataclasses.asdict(audio_settings),
            "sample_bytes": 2,
            "samples": 0,
            "channels": 1,
            #
            # Silence trimming
            "silence_trimmed": args.trim_silence,
            "trim_db": args.trim_db,
        },
        "mel": [],
    }

    num_wavs = 0
    if args.wav:
        # Convert to paths
        args.wav = [Path(p) for p in args.wav]

        # Process WAVs
        for wav_path in args.wav:
            try:
                _LOGGER.debug("Processing %s", wav_path)
                wav_array, _ = librosa.load(wav_path, sr=args.sample_rate)
                wav_array = wav_array.astype(np.float32)

                mel_array = audio_settings.wav2mel(
                    wav_array, trim_silence=args.trim_silence, trim_db=args.trim_db
                )

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
            except Exception:
                _LOGGER.exception(wav_path)
    else:
        # Read from stdin, write to stdout
        if os.isatty(sys.stdin.fileno()):
            print("Reading WAV data from stdin...", file=sys.stderr)

        with io.BytesIO(sys.stdin.buffer.read()) as wav_file:
            wav_array, _ = librosa.load(wav_file, sr=args.sample_rate)

        wav_array = wav_array.astype(np.float32)

        mel_array = audio_settings.wav2mel(
            wav_array, trim_silence=args.trim_silence, trim_db=args.trim_db
        )

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
                with io.BytesIO() as np_file:
                    np.save(np_file, mel_array)
                    sys.stdout.buffer.write(np_file.getvalue())
        else:
            output_obj["mel"] = mel_array.tolist()
            output_obj["audio"]["samples"] = len(wav_array)
            writer.write(output_obj)

        num_wavs += 1

    _LOGGER.info("Done (%s WAV file(s))", num_wavs)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
