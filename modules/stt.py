"""
Speech-to-text using faster-whisper.
"""

import numpy as np
from faster_whisper import WhisperModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE, WHISPER_INITIAL_PROMPT,
)


class STT:
    def __init__(self):
        print(f"Loading Whisper '{WHISPER_MODEL}' model...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("Whisper ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 audio array.
        Language is auto-detected (supports Chinese and English).
        Returns the transcribed text, or empty string if nothing detected.
        """
        if audio.size == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio,
            beam_size=WHISPER_BEAM_SIZE,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=WHISPER_INITIAL_PROMPT,
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
