# wav2mel

Converts WAV audio [1] to Mel spectrograms for use in machine learning systems like Tacotron2.

This library contains portions of the copy-pasted code you see everywhere for WAV to Mel conversion.

[1] Or any audio format supported by [librosa](https://librosa.org) (which uses [soundfile](https://pysoundfile.readthedocs.io) and [audioread](https://github.com/beetbox/audioread)).

## Installation

```sh
pip install wav2mel
```

## Dependencies

* Python 3.6 or higher
* librosa, numpy, scipy, numba

## Format

`wav2mel` outputs [numpy save data](https://numpy.org/doc/stable/reference/generated/numpy.save.html): one `.npy` file each input file. 

## Usage

You can convert a single WAVE file from `.wav` to a mel spectrogram (`.npy`) as follows:

```sh
wav2mel < WAVE_FILE > NPY_FILE
```

Multiple WAVE files can also be converted and saved to a directory:

```sh
wav2mel --output-dir /path/to/mels WAVE_FILE [WAVE_FILE ...]
```

Each `.npy` file will be named after the corresponding `.wav` file.

See `wav2mel --help` for more options (filter/hop/window length, sample rate, etc.).

## With GNU Parallel

```sh
find /path/to/wavs -name '*.wav' -type f | parallel -X wav2mel --output-dir /path/to/mels
```

## mel2wav (Griffin-Lim)

You can convert a mel spectrogram to WAV audio too using [griffin-lim](https://paperswithcode.com/method/griffin-lim-algorithm):

```sh
mel2wav < NPY_FILE > WAVE_FILE
```

or

```sh
mel2wav --output-dir /path/to/wavs NPY_FILE [NPY_FILE ...]
```

See `mel2wav --help` for more options.
