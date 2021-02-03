# wav2mel

Converts WAV audio to Mel spectrograms for use in machine learning systems like Tacotron2.

This library contains portions of the copy-pasted code you see everywhere for WAV to Mel conversion.

## Installation

```sh
$ pip install wav2mel
```

## Dependencies

* Python 3.6 or higher
* librosa, numpy, scipy, numba

## Usage

You can convert a single WAV file from `.wav` to `.npy` as follows:

```sh
$ wav2mel < WAVE_FILE > MEL_NPY_FILE
```

Multiple WAV files can also be converted and saved to a directory:

```sh
$ wav2mel --output-dir OUTPUT_DIRECTORY WAVE_FILE [WAVE_FILE ...]
```

Output files will have the same names as the input files, just with a `.npy` extension instead of `.wav`.

See `wav2mel --help` for more options (filter/hop/window length, sample rate, etc.).

## With GNU Parallel

```sh
$ find /path/to/wavs -name '*.wav' -type f | parallel -X wav2mel --output-dir /path/to/mels
```
