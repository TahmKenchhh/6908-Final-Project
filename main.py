"""
Edge Voice Assistant — Max
  [idle]  Wake word detector listens for "Hey Max"
      ↓   wake phrase detected
  [active] Microphone → VAD → Whisper STT → Ollama LLM → Piper TTS → Speaker
      ↓   ACTIVE_TIMEOUT seconds of silence
  [idle]  back to wake word mode

UI buttons (sent from browser via WebSocket):
  interrupt  — stop current LLM generation / TTS
  vision     — immediately enter vision mode (camera snapshot + VLM)
"""

import os
import re
import sys
import signal
import time

os.environ["ORT_LOGGING_LEVEL"] = "4"

from modules.audio import calibrate_noise_floor, record_speech, play_audio, play_audio_with_lipsync
from modules.stt import STT
from modules.llm import LLM
from modules.tts import TTS
from modules.wakeword import WakeWordDetector
from modules.weather import get_weather_summary
from modules.camera import is_vision_query, capture_frame_b64
from modules.light import LightSensor, is_light_query
from modules.led import on_lux as led_on_lux, set_led, detect_led_command

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ACTIVE_TIMEOUT, DISPLAY_ENABLED, WEATHER_LAT, WEATHER_LON, WEATHER_CITY,
)

if DISPLAY_ENABLED:
    from modules.display import Display, State
else:
    class State:
        IDLE = LISTENING = THINKING = SPEAKING = None
    class Display:
        interrupt_event = type('E', (), {'is_set': lambda s: False, 'clear': lambda s: None, 'set': lambda s: None})()
        vision_trigger  = type('E', (), {'is_set': lambda s: False, 'clear': lambda s: None, 'set': lambda s: None})()
        def start(self): pass
        def set_state(self, s): pass
        def send_mouth(self, v): pass
        def clear_interrupt(self): pass

_SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s*')


def _play(path: str, text: str, display: Display) -> None:
    if DISPLAY_ENABLED:
        play_audio_with_lipsync(path, display, text, interrupt=display.interrupt_event)
    else:
        play_audio(path, interrupt=display.interrupt_event)


def _speak(text: str, tts: TTS, display: Display) -> None:
    text = text.strip()
    if not text:
        return
    display.set_state(State.SPEAKING)
    f = tts.synthesize(text)
    _play(f, text, display)
    try:
        os.unlink(f)
    except OSError:
        pass


def _do_vision(llm: LLM, tts: TTS, display: Display) -> None:
    """Capture a frame and describe it — used by both voice and button trigger."""
    print("[camera] capturing frame...")
    display.set_state(State.THINKING)
    image_b64 = capture_frame_b64()
    if not image_b64:
        print("[camera] capture failed")
        return

    print("Max: ", end="", flush=True)
    buffer = ""
    first_sentence = True
    tmp_files: list[str] = []
    accumulated = ""

    for chunk in llm.chat("What do you see? Describe briefly.", image_b64=image_b64):
        if display.interrupt_event.is_set():
            break
        print(chunk, end="", flush=True)
        buffer += chunk
        accumulated += chunk
        display.send_transcript("assistant", accumulated)
        parts = _SENTENCE_END.split(buffer)
        for sentence in parts[:-1]:
            s = sentence.strip()
            if s:
                if first_sentence:
                    display.set_state(State.SPEAKING)
                    first_sentence = False
                f = tts.synthesize(s)
                tmp_files.append(f)
                _play(f, s, display)
        buffer = parts[-1]

    print()
    if buffer.strip() and not display.interrupt_event.is_set():
        display.set_state(State.SPEAKING)
        f = tts.synthesize(buffer.strip())
        tmp_files.append(f)
        _play(f, buffer.strip(), display)

    for path in tmp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_turn(stt: STT, llm: LLM, tts: TTS, display: Display, noise_floor: float, light: "LightSensor | None" = None) -> bool:
    display.clear_interrupt()
    display.set_state(State.LISTENING)
    audio = record_speech(noise_floor)
    if audio.size == 0:
        return False

    display.set_state(State.THINKING)
    print("Transcribing...", end="\r")
    text = stt.transcribe(audio)
    if not text:
        print("(no speech detected)      ")
        return False
    print(f"You: {text}          ")
    display.send_transcript("user", text)

    if text.lower().strip() in {"clear", "clear history", "reset"}:
        llm.clear_history()
        return True

    # LED voice command — bypass LLM entirely
    led_cmd = detect_led_command(text)
    if led_cmd is not None:
        set_led(led_cmd)
        reply = "Light on." if led_cmd else "Light off."
        print(f"[led] {reply}")
        _speak(reply, tts, display)
        return True

    # Voice-triggered vision
    image_b64 = None
    if is_vision_query(text):
        print("[camera] capturing frame...")
        image_b64 = capture_frame_b64()
        if not image_b64:
            print("[camera] capture failed, continuing without image")

    context_parts = []
    weather = get_weather_summary(text, WEATHER_LAT, WEATHER_LON, WEATHER_CITY)
    if weather and not image_b64:
        print(f"[weather] {weather}")
        context_parts.append(f"Weather: {weather}")
    if light and is_light_query(text) and light.lux is not None:
        lux = light.lux
        desc = "very dark" if lux < 10 else "dark" if lux < 50 else "dim" if lux < 200 else "bright" if lux < 1000 else "very bright"
        context_parts.append(f"Ambient light: {lux} lux ({desc})")
        print(f"[light] {lux} lux ({desc})")
    if context_parts and not image_b64:
        query = f"[Sensor data: {'; '.join(context_parts)}]\nUser asked: {text}"
    else:
        query = text

    print("Max: ", end="", flush=True)
    buffer = ""
    first_sentence = True
    tmp_files: list[str] = []

    accumulated = ""
    for chunk in llm.chat(query, image_b64=image_b64):
        if display.interrupt_event.is_set():
            print("\n[interrupted]")
            break
        print(chunk, end="", flush=True)
        buffer += chunk
        accumulated += chunk
        display.send_transcript("assistant", accumulated)
        parts = _SENTENCE_END.split(buffer)
        for sentence in parts[:-1]:
            s = sentence.strip()
            if s:
                if first_sentence:
                    display.set_state(State.SPEAKING)
                    first_sentence = False
                f = tts.synthesize(s)
                tmp_files.append(f)
                if not display.interrupt_event.is_set():
                    _play(f, s, display)
        buffer = parts[-1]

    print()

    if buffer.strip() and not display.interrupt_event.is_set():
        display.set_state(State.SPEAKING)
        f = tts.synthesize(buffer.strip())
        tmp_files.append(f)
        _play(f, buffer.strip(), display)

    for path in tmp_files:
        try:
            os.unlink(path)
        except OSError:
            pass

    return True


def main() -> None:
    print("=== Edge Voice Assistant — Max ===")
    print("Initialising modules...\n")

    display = Display()
    display.start()
    display.set_state(State.THINKING)

    stt = STT()
    llm = LLM()
    tts = TTS()
    wake = WakeWordDetector()
    noise_floor = calibrate_noise_floor()

    import threading
    light = LightSensor(interval=1.0)
    _dark_event = threading.Event()
    _was_dark = False

    def _on_lux(lux: float):
        nonlocal _was_dark
        display.send_lux(lux)
        is_dark = lux < 20
        if is_dark and not _was_dark:   # falling edge: just became dark
            print(f"[light] fell below 20 lux ({lux}), setting dark event")
            _dark_event.set()
        _was_dark = is_dark

    light.start(on_update=_on_lux)

    display.set_state(State.IDLE)
    print('\nSay "Hey Max" to activate. Ctrl+C to quit.\n')

    def _handle_exit(sig, frame):
        print("\nGoodbye.")
        llm.unload()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_exit)

    while True:
        try:
            display.set_state(State.IDLE)
            display.clear_interrupt()

            # Lux fell below threshold — proactively ask about the light
            if _dark_event.is_set():
                print(f"[light] dark event fired! lux={light.lux}")
                _dark_event.clear()
                _speak("It's getting dark. Would you like me to turn on the light?", tts, display)
                display.set_state(State.LISTENING)
                answer = record_speech(noise_floor)
                if answer.size > 0:
                    ans_text = stt.transcribe(answer).lower()
                    print(f"[light] answer: {ans_text}")
                    if any(w in ans_text for w in ("yes", "yeah", "sure", "please", "ok", "yep")):
                        set_led(True)
                        _speak("Light on.", tts, display)
                    else:
                        _speak("Okay.", tts, display)
                continue

            # Camera overlay open in idle: skip wake word, just poll
            if display.camera_open.is_set():
                if display.vision_trigger.is_set():
                    display.vision_trigger.clear()
                    _do_vision(llm, tts, display)
                time.sleep(0.1)
                continue

            if not wake.listen(noise_floor):
                continue

            _speak("Yes?", tts, display)
            llm.clear_history()
            last_activity = time.time()

            while True:
                # Camera overlay open: pause voice, keep session alive
                if display.camera_open.is_set():
                    last_activity = time.time()  # prevent session timeout
                    if display.vision_trigger.is_set():
                        display.vision_trigger.clear()
                        _do_vision(llm, tts, display)
                    time.sleep(0.1)
                    continue

                # Dark event during active session — ask about light inline
                if _dark_event.is_set():
                    print(f"[light] dark event fired! lux={light.lux}")
                    _dark_event.clear()
                    _speak("It's getting dark. Would you like me to turn on the light?", tts, display)
                    display.set_state(State.LISTENING)
                    answer = record_speech(noise_floor)
                    if answer.size > 0:
                        ans_text = stt.transcribe(answer).lower()
                        print(f"[light] answer: {ans_text}")
                        if any(w in ans_text for w in ("yes", "yeah", "sure", "please", "ok", "yep")):
                            set_led(True)
                            _speak("Light on.", tts, display)
                        else:
                            _speak("Okay.", tts, display)
                    last_activity = time.time()
                    continue

                if display.interrupt_event.is_set():
                    print("[interrupted] ready for next command")
                    display.clear_interrupt()
                    last_activity = time.time()
                    display.set_state(State.LISTENING)
                if time.time() - last_activity > ACTIVE_TIMEOUT:
                    print("[idle] No activity, returning to wake word mode.")
                    break
                did_speak = run_turn(stt, llm, tts, display, noise_floor, light)
                if did_speak:
                    last_activity = time.time()

        except Exception as exc:
            print(f"\n[error] {exc}")


if __name__ == "__main__":
    main()
