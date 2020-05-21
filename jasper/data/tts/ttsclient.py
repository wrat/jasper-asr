"""
TTSClient Abstract Class
"""
from abc import ABC, abstractmethod


class TTSClient(ABC):
    """
    Base class for TTS
    """

    @abstractmethod
    def text_to_speech(self, text: str, num_channels: int, sample_rate: int,
                       audio_encoding) -> bytes:
        """
        convert text to bytes

        Arguments:
            text {[type]} -- text to convert
            channel {[type]} -- output audio bytes channel setting
            width {[type]} -- width of audio bytes
            rate {[type]} -- rare for audio bytes

        Returns:
            [type] -- [description]
        """
