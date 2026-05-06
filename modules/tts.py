"""
Text-to-speech using piper-tts >= 1.4.

piper-tts 1.4 changed the API: synthesize() now returns an Iterable[AudioChunk]
instead of writing directly to a wave.Wave_write object.
Each AudioChunk has audio_float_array (float32 numpy array) and sample_rate.
"""

import wave
import tempfile
import os
import numpy as np
from piper.voice import PiperVoice

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PIPER_MODEL


class TTS:
    def __init__(self):
        if not os.path.exists(PIPER_MODEL):
            raise FileNotFoundError(
                f"Piper model not found: {PIPER_MODEL}\n"
                "Run setup.sh to download the voice model."
            )
        print("Loading Piper TTS model...")
        self.voice = PiperVoice.load(PIPER_MODEL)
        print("Piper TTS ready.")

    def synthesize(self, text: str) -> str:
        """
        Convert text to speech using the piper-tts 1.4+ API.
        Returns the path to a temporary WAV file (caller must delete it).
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        chunks = list(self.voice.synthesize(text))

        sample_rate = chunks[0].sample_rate if chunks else 22050

        with wave.open(tmp_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(sample_rate)
            for chunk in chunks:
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

        return tmp_path
