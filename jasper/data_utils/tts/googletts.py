from logging import getLogger
from google.cloud import texttospeech

LOGGER = getLogger("googletts")


class GoogleTTS(object):
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def text_to_speech(self, text: str, params: dict) -> bytes:
        tts_input = texttospeech.types.SynthesisInput(ssml=text)
        voice = texttospeech.types.VoiceSelectionParams(
            language_code=params["language"], name=params["name"]
        )
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
            sample_rate_hertz=params["sample_rate"],
        )
        response = self.client.synthesize_speech(tts_input, voice, audio_config)
        audio_content = response.audio_content
        return audio_content

    @classmethod
    def voice_list(cls):
        """Lists the available voices."""

        client = cls().client

        # Performs the list voices request
        voices = client.list_voices()
        results = []
        for voice in voices.voices:
            supported_eng_langs = [
                lang for lang in voice.language_codes if lang[:2] == "en"
            ]
            if len(supported_eng_langs) > 0:
                lang = ",".join(supported_eng_langs)
            else:
                continue

            ssml_gender = texttospeech.enums.SsmlVoiceGender(voice.ssml_gender)
            results.append(
                {
                    "name": voice.name,
                    "language": lang,
                    "gender": ssml_gender.name,
                    "engine": "wavenet" if "Wav" in voice.name else "standard",
                    "sample_rate": voice.natural_sample_rate_hertz,
                }
            )
        return results
