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
      "normalized": true if audio was normalized (default),
  },
  "mel": [numpy array of shape (mel_channels, )],
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

See `wav2mel --help` for more options (filter/hop/window length, sample rate, etc.).

## With GNU Parallel

```sh
$ find /path/to/wavs -name '*.wav' -type f | parallel -X wav2mel | gzip -9 --to-stdout > JSON_FILE.gz
```
