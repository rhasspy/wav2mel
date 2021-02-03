"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from librosa import stft  # pylint: disable=import-error


class STFT:
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
    ):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

    def transform(self, input_data):
        x = input_data
        real_part = []
        imag_part = []
        for y in x:
            y_ = stft(
                y, self.filter_length, self.hop_length, self.win_length, self.window
            )
            real_part.append(
                y_.real[None, :, :]  # pylint: disable=unsubscriptable-object
            )
            imag_part.append(
                y_.imag[None, :, :]  # pylint: disable=unsubscriptable-object
            )
        real_part = np.concatenate(real_part, 0)
        imag_part = np.concatenate(imag_part, 0)

        magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
        phase = np.arctan2(imag_part.data, real_part.data)

        return magnitude, phase
