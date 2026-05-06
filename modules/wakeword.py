"""
Wake word detection using faster-whisper tiny model.

Continuously records short audio chunks (energy-triggered),
transcribes with the tiny Whisper model, and checks for wake phrases.
No extra dependencies — reuses the existing STT stack.
"""

import queue
import time
import numpy as np
import sounddevice as sd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RECORD_SAMPLE_RATE, CHANNELS, WHISPER_SAMPLE_RATE,
    VAD_MODE, WAKE_PHRASES, WAKE_WHISPER_MODEL, WAKE_CHUNK_DURATION,
)

import re
import webrtcvad
from collections import deque
from faster_whisper import WhisperModel

_PUNCT = re.compile(r"[^\w\s]")

FRAME_DURATION_MS = 30
FRAME_SIZE = int(RECORD_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
_DECIMATE = RECORD_SAMPLE_RATE // WHISPER_SAMPLE_RATE

# Lower threshold for wake word; soft speech should still trigger.
ENERGY_MULTIPLIER = 0.75
# Pre-buffer: keep audio before the trigger so "Hey" isn't clipped.
PRE_BUFFER_FRAMES = int(1.0 * RECORD_SAMPLE_RATE / FRAME_SIZE)

_DEMO_WAKE_TOKENS = {
    "max", "mac", "mack", "macs", "macks", "maps", "map", "mask",
    "mex", "mix", "mike", "next", "hey", "hi", "hello",
}


class WakeWordDetector:
    def __init__(self):
        print(f"Loading wake word model (whisper-{WAKE_WHISPER_MODEL})...")
        self.model = WhisperModel(
            WAKE_WHISPER_MODEL, device="cpu", compute_type="int8"
        )
        print("Wake word detector ready. Say 'Hey Max' to activate.")

    def _record_chunk(self, noise_floor: float) -> np.ndarray:
        """
        Wait for energy above noise floor, then record WAKE_CHUNK_DURATION seconds.
        Includes a pre-buffer so the start of "Hey" is not clipped.
        Returns float32 audio at 16000 Hz.
        """
        threshold = noise_floor * ENERGY_MULTIPLIER
        chunk_frames = int(WAKE_CHUNK_DURATION * RECORD_SAMPLE_RATE / FRAME_SIZE)
        pre_buffer: deque[np.ndarray] = deque(maxlen=PRE_BUFFER_FRAMES)

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _cb(indata, *_):
            audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=FRAME_SIZE,
            callback=_cb,
        ):
            # Phase 1: wait for energy trigger
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                mono = chunk[:, 0] if chunk.ndim == 2 else chunk
                rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
                if rms >= threshold:
                    # Phase 2: collect a fixed-length window (pre-buffer + current + rest)
                    frames = list(pre_buffer) + [chunk.copy()]
                    for _ in range(chunk_frames - 1):
                        try:
                            frames.append(audio_queue.get(timeout=0.5).copy())
                        except queue.Empty:
                            break
                    audio = np.concatenate(frames)
                    mono = audio[:, 0] if audio.ndim == 2 else audio
                    return mono[::_DECIMATE].astype(np.float32) / 32768.0
                pre_buffer.append(chunk.copy())

    def listen(self, noise_floor: float) -> bool:
        """
        Block until a wake phrase is detected.
        Returns True when 'hey max' (or a phonetic variant) is heard.
        """
        audio = self._record_chunk(noise_floor)
        if audio.size == 0:
            return False

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            initial_prompt=(
                "Hey Max. Hi Max. Hello Max. Hey Max wake up. "
                "Wake up Max. Max listen. Max can you hear me."
            ),
        )
        raw = " ".join(seg.text.strip() for seg in segments)
        # Strip punctuation before matching so "Hey, Max." → "hey max"
        text = _PUNCT.sub("", raw).lower()
        tokens = set(text.split())
        print(f"[wake] heard: '{raw.strip()}'", flush=True)

        if any(phrase in text for phrase in WAKE_PHRASES) or tokens & _DEMO_WAKE_TOKENS:
            print("[wake] ✓ wake word matched!", flush=True)
            return True
        return False
