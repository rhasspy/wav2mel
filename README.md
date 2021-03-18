# wav2mel

Converts WAV audio to Mel spectrograms for use in machine learning systems like Tacotron2.

This library contains portions of the copy-pasted code you see everywhere for WAV to Mel conversion.

## Installation

```sh
$ pip install wav2mel
```

## Dependencies

* Python 3.6 or higher
* librosa, numpy, scipy, numba, jsonlines

## Format

`wav2mel` outputs [JSONL](https://jsonlines.org/): one line of JSON for each input file. Each line has a JSON object with the following format:

```json
{
  "id": "name of the input file without extension or empty if stdin",
  "audio": {
      "filter_length": length of filter (default: 1024),
      "hop_length": length of hop (default: 256),
      "win_length": length of window (default: 1024),
      "mel_channels": number of mel channels (default: 80),
      "sample_rate": sample rate of audio (default: 22050),
      "sample_bytes": number of bytes per sample (default: 2),
      "samples": number of WAV samples,
      "channels": number of channels in the audio (default: 1),
      "mel_fmin": min frequency for mel (default: 0),
      "mel_fmax": max frequency for mel (default: 8000),
      "ref_level_db": threshold to discard audio (default: 20),
      "spec_gain": gain in amp to db conversion (default: 1),
      
      "signal_norm": true if mel was normalized (default: true),
      "max_norm": range of normalization (default: 4),
      "min_level_db": min db for normalization (default: -100),
      "clip_norm": clip during normalization (default: true),
      "symmetric_norm": normalize in [-max_norm, max_norm] instead of [0, max_norm] (default: true)
  },
  "mel": [numpy array of shape (mel_channels, mel_windows)]
}
```

## Usage

You can convert a single WAV file from `.wav` to JSON as follows:

```sh
$ wav2mel < WAVE_FILE > JSON_FILE
```

Multiple WAV files can also be converted and saved to a compressed archive:

```sh
$ wav2mel WAVE_FILE [WAVE_FILE ...] | gzip --to-stdout > JSON_FILE.gz
```

Add `--numpy` to output a `.npy` file instead of JSONL. Use `--numpy-dir` for multiple WAV files.

See `wav2mel --help` for more options (filter/hop/window length, sample rate, etc.).

## With GNU Parallel

```sh
$ find /path/to/wavs -name '*.wav' -type f | parallel -X wav2mel | gzip -9 --to-stdout > JSON_FILE.gz
```

## Griffin-Lim

You can convert a mel spectrogram to WAV audio too:

```sh
$ griffim-lim /path/to/wavs/ < JSON_FILE
```

This will write WAV files with names `<id>.wav` where `<id>` is the value if the "id" field in each JSON object or a timestamp if not available.

Add `--numpy-files` to read `.npy` file names from stdin instead of JSONL:

```sh
$ find /path/to/npy -name '*.npy' | griffin-lim --numpy-files /path/to/wavs
```

See `griffin-lim --help` for more options.
