"""Utility methods"""


def add_audio_settings(parser):
    """Add audio settings to argparse parser"""
    # STFT settings
    parser.add_argument("--filter-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--mel-channels", type=int, default=80)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--mel-fmin", type=float, default=0.0)
    parser.add_argument("--mel-fmax", type=float, default=8000.0)
    parser.add_argument("--spec-gain", type=float, default=1.0)

    # Normalization
    parser.add_argument("--ref-level-db", type=float, default=20.0)
    parser.add_argument("--min-level-db", type=float, default=-100.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-norm", type=float, default=4.0)
    parser.add_argument("--asymmetric-norm", action="store_true")
    parser.add_argument("--no-clip-norm", action="store_true")
