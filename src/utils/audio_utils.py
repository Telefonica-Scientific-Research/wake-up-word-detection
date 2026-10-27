# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

def trim_silence(audio_signal):
    """
    Trim the silence of an audio signal. Considers silence samples which level is under a given threshold
    Args:
        audio_signal: Audio signal as numpy array
        silence_threshold: Threshold to determine silence or not
    Returns:
        Trimed signal and its lengh as a tuple
    """
    silence_threshold = audio_signal.max()*0.01
    start = 0
    end = len(audio_signal) - 1
    for i in range(start, len(audio_signal)):
        if (audio_signal[i] > silence_threshold):
            start = i - 1
            break
    for i in range(len(audio_signal)):
        if (audio_signal[end - i] > silence_threshold):
            end = len(audio_signal) - i + 1
            break
    return (audio_signal[start:end], len(audio_signal[start:end]))