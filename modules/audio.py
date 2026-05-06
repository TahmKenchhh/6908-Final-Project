"""
Audio capture (VAD-based) and playback utilities.

Records at 48000 Hz (USB mic native rate).
Each frame is decimated to 16000 Hz for webrtcvad (most reliable rate).
Full audio is decimated to 16000 Hz before being passed to Whisper.

Double-gate speech detection:
  1. Energy gate  — RMS must be > noise_floor * ENERGY_MULTIPLIER
  2. webrtcvad    — frame content must look like speech
Both must be true to count as a speech frame.
"""

import queue
import subprocess
import threading
import time
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad  # installed via webrtcvad-wheels
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RECORD_SAMPLE_RATE, WHISPER_SAMPLE_RATE, CHANNELS, VAD_MODE,
    SILENCE_DURATION, MAX_RECORD_DURATION, PRE_SPEECH_BUFFER_S,
    AUDIO_OUTPUT_DEVICE, AUDIO_VOLUME,
)

FRAME_DURATION_MS = 30
FRAME_SIZE = int(RECORD_SAMPLE_RATE * FRAME_DURATION_MS / 1000)   # 1440 @ 48kHz
_DECIMATE = RECORD_SAMPLE_RATE // WHISPER_SAMPLE_RATE              # 3
VAD_FRAME_SIZE = FRAME_SIZE // _DECIMATE                           # 480 @ 16kHz

# Energy gate: RMS must be this many times above the measured noise floor
ENERGY_MULTIPLIER = 2.0
CALIBRATION_DURATION = 1.5  # seconds


def calibrate_noise_floor() -> float:
    """
    Sample ambient audio for a short period and return the RMS noise floor.
    Called once at startup before the main loop.
    """
    print("Calibrating noise floor, please stay quiet...", flush=True)
    n_frames = int(CALIBRATION_DURATION * RECORD_SAMPLE_RATE / FRAME_SIZE)
    chunks = []
    with sd.InputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=FRAME_SIZE,
    ) as stream:
        for _ in range(n_frames):
            chunk, _ = stream.read(FRAME_SIZE)
            chunks.append(chunk)
    audio = np.concatenate(chunks).astype(np.float32)
    noise_floor = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Noise floor: {noise_floor:.0f}  (speech threshold: {noise_floor * ENERGY_MULTIPLIER:.0f})", flush=True)
    return noise_floor


def _decimate_frame(frame_int16: np.ndarray) -> bytes:
    """Decimate a 48kHz int16 frame to 16kHz and return raw bytes for webrtcvad."""
    mono = frame_int16[:, 0] if frame_int16.ndim == 2 else frame_int16
    return mono[::_DECIMATE].astype(np.int16).tobytes()


def _is_speech(vad: webrtcvad.Vad, frame: np.ndarray, noise_floor: float) -> bool:
    """Return True only if energy gate AND webrtcvad both say speech."""
    mono = frame[:, 0] if frame.ndim == 2 else frame
    rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
    if rms < noise_floor * ENERGY_MULTIPLIER:
        return False
    try:
        return vad.is_speech(_decimate_frame(frame), WHISPER_SAMPLE_RATE)
    except Exception:
        return False


def record_speech(noise_floor: float) -> np.ndarray:
    """
    Block until speech is detected, record until silence, return 16kHz float32 audio.
    Returns an empty array if nothing meaningful was captured.

    Uses a callback + queue instead of stream.read() so that Ctrl+C (SIGINT)
    is never swallowed by a blocking C-extension call.
    """
    vad = webrtcvad.Vad(VAD_MODE)

    max_silence_frames = int(SILENCE_DURATION * 1000 / FRAME_DURATION_MS)
    max_frames = int(MAX_RECORD_DURATION * 1000 / FRAME_DURATION_MS)
    pre_buf_len = int(PRE_SPEECH_BUFFER_S * 1000 / FRAME_DURATION_MS)

    frames: list[np.ndarray] = []
    pre_buffer: deque[np.ndarray] = deque(maxlen=pre_buf_len)
    silence_count = 0
    speech_detected = False
    total_frames = 0

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, frame_count, time_info, status):
        audio_queue.put(indata.copy())

    print("Listening... (speak now)", flush=True)

    with sd.InputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=FRAME_SIZE,
        callback=_callback,
    ):
        while total_frames < max_frames:
            try:
                chunk = audio_queue.get(timeout=0.5)  # yields to Python every 500ms → Ctrl+C works
            except queue.Empty:
                continue

            is_sp = _is_speech(vad, chunk, noise_floor)

            if not speech_detected:
                pre_buffer.append(chunk.copy())
                if is_sp:
                    speech_detected = True
                    frames.extend(list(pre_buffer))
                    frames.append(chunk.copy())
                    print("Recording...", flush=True)
            else:
                frames.append(chunk.copy())
                if is_sp:
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= max_silence_frames:
                        break

            total_frames += 1

    if not frames:
        return np.array([], dtype=np.float32)

    audio_48k = np.concatenate(frames)
    mono = audio_48k[:, 0] if audio_48k.ndim == 2 else audio_48k
    return mono[::_DECIMATE].astype(np.float32) / 32768.0


def _kill_procs(*procs) -> None:
    for p in procs:
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    for p in procs:
        if p:
            try:
                p.wait(timeout=0.3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def play_audio(file_path: str, interrupt=None) -> None:
    """
    Play a WAV file via ffmpeg → aplay pipeline.
    If `interrupt` (threading.Event) is set during playback, kills the pipeline.
    """
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-i", file_path,
         "-af", f"volume={AUDIO_VOLUME},adelay=300|300",
         "-ac", "2", "-ar", "48000", "-f", "wav",
         "-loglevel", "quiet", "-"],
        stdout=subprocess.PIPE,
    )
    aplay = subprocess.Popen(
        ["aplay", "-D", AUDIO_OUTPUT_DEVICE, "-"],
        stdin=ffmpeg.stdout,
    )
    ffmpeg.stdout.close()
    while aplay.poll() is None:
        if interrupt is not None and interrupt.is_set():
            _kill_procs(aplay, ffmpeg)
            return
        time.sleep(0.05)


_ADELAY_S = 0.300  # must match adelay=300 in ffmpeg filter

# Mouth openness per espeak-ng phoneme (0=closed, 1=fully open)
_PHONEME_MOUTH: dict[str, float] = {
    # silence
    "_": 0.0,
    # bilabials — lips closed
    "p": 0.05, "b": 0.05, "m": 0.05,
    # labiodentals
    "f": 0.1, "v": 0.1,
    # alveolars
    "t": 0.2, "d": 0.2, "n": 0.2, "l": 0.2, "s": 0.2, "z": 0.2,
    # post-alveolars / palatals
    "S": 0.25, "Z": 0.25, "tS": 0.25, "dZ": 0.25,
    # velars
    "k": 0.3, "g": 0.3, "N": 0.3,
    # other consonants
    "h": 0.4, "r": 0.3, "w": 0.25, "j": 0.25,
    # front vowels (less open)
    "i": 0.3, "I": 0.3, "e": 0.5, "E": 0.55,
    # central vowels
    "@": 0.35, "3": 0.45, "V": 0.6,
    # back vowels
    "u": 0.3, "U": 0.3, "o": 0.65, "O": 0.7, "Q": 0.75,
    # open vowels
    "a": 0.95, "A": 0.9,
    # diphthongs
    "aI": 0.9, "aU": 0.9, "OI": 0.8, "eI": 0.65, "oU": 0.65,
    "I@": 0.5, "e@": 0.5, "U@": 0.45,
}


def _phoneme_timeline(text: str, duration_s: float) -> list[tuple[float, float]]:
    """
    Run espeak-ng -x on text, parse phonemes, distribute evenly across duration.
    Returns [(time_s, mouth_value), ...].
    """
    import re
    try:
        result = subprocess.run(
            ["espeak-ng", "-x", "-q", "--", text],
            capture_output=True, text=True, timeout=5,
        )
        raw = result.stdout
    except Exception:
        return []

    # Strip stress/tone markers, keep phoneme chars
    cleaned = re.sub(r"[',\"\r\n]+", " ", raw)
    phonemes: list[str] = []
    for token in cleaned.split():
        i = 0
        while i < len(token):
            two = token[i:i+2]
            if two in _PHONEME_MOUTH:
                phonemes.append(two)
                i += 2
            elif token[i] in _PHONEME_MOUTH:
                phonemes.append(token[i])
                i += 1
            else:
                i += 1

    if not phonemes:
        return []

    step = duration_s / len(phonemes)
    return [(i * step, _PHONEME_MOUTH[ph]) for i, ph in enumerate(phonemes)]


def play_audio_with_lipsync(file_path: str, display, text: str = "", interrupt=None) -> None:
    """
    Play file_path via HDMI while animating the character's mouth.
    Uses phoneme-based timing when text is provided, amplitude fallback otherwise.
    """
    # Get WAV duration
    try:
        with wave.open(file_path, "rb") as wf:
            wav_duration = wf.getnframes() / wf.getframerate()
    except Exception:
        wav_duration = 0.0

    # Build timeline: list of (time_s, mouth_value) relative to speech start
    if text and wav_duration > 0:
        timeline = _phoneme_timeline(text, wav_duration)
    else:
        timeline = []

    # Amplitude fallback
    if not timeline and wav_duration > 0:
        try:
            with wave.open(file_path, "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_ch > 1:
                samples = samples.reshape(-1, n_ch)[:, 0]
            fps = 30
            frame_len = max(1, sr // fps)
            n_frames = len(samples) // frame_len
            rms = np.array([
                float(np.sqrt(np.mean(samples[i*frame_len:(i+1)*frame_len] ** 2)))
                for i in range(n_frames)
            ])
            peak = rms.max() if rms.max() > 0 else 1.0
            timeline = [(i / fps, float(np.clip(rms[i] / peak, 0, 1)))
                        for i in range(n_frames)]
        except Exception:
            pass

    # Start playback in background thread
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-i", file_path,
         "-af", f"volume={AUDIO_VOLUME},adelay=300|300",
         "-ac", "2", "-ar", "48000", "-f", "wav",
         "-loglevel", "quiet", "-"],
        stdout=subprocess.PIPE,
    )
    aplay = subprocess.Popen(
        ["aplay", "-D", AUDIO_OUTPUT_DEVICE, "-"],
        stdin=ffmpeg.stdout,
    )
    ffmpeg.stdout.close()
    player_thread = threading.Thread(target=aplay.wait, daemon=True)
    player_thread.start()

    # Drive mouth values timed to audio (interruptible)
    def _interrupted() -> bool:
        return interrupt is not None and interrupt.is_set()

    # Sleep adelay in small chunks so we can react to interrupt
    end = time.monotonic() + _ADELAY_S
    while time.monotonic() < end:
        if _interrupted():
            _kill_procs(aplay, ffmpeg)
            display.send_mouth(0.0)
            return
        time.sleep(0.02)

    t_start = time.monotonic()
    for t_ph, mouth_val in timeline:
        if _interrupted():
            _kill_procs(aplay, ffmpeg)
            display.send_mouth(0.0)
            return
        target = t_start + t_ph
        wait = target - time.monotonic()
        if wait > 0:
            time.sleep(min(wait, 0.05))
            if _interrupted():
                _kill_procs(aplay, ffmpeg)
                display.send_mouth(0.0)
                return
            # finish remaining wait
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        display.send_mouth(mouth_val)

    display.send_mouth(0.0)
    # Wait for playback to finish, but bail on interrupt
    while aplay.poll() is None:
        if _interrupted():
            _kill_procs(aplay, ffmpeg)
            return
        time.sleep(0.05)
