import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Display ---
DISPLAY_ENABLED = True    # set True to launch Chromium avatar

# --- STT (faster-whisper) ---
WHISPER_MODEL = "base"          # tiny / base / small / medium
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"   # fastest on CPU
WHISPER_BEAM_SIZE = 1           # higher = more accurate, slower
WHISPER_INITIAL_PROMPT = (
    "Hey Max. What is the weather. Can you see. Tell me, please. "
    "What time is it. What do you see."
)

# --- LLM (Ollama) ---
OLLAMA_MODEL = "qwen3:1.7b"           # text-only local LLM
VISION_MODEL = "qwen3-vl:2b"          # 2B Qwen3 vision model, better quality than moondream
OLLAMA_HOST = "http://localhost:11434"

LLM_NUM_PREDICT = 35                   # hard cap on generated tokens per response
VISION_NUM_PREDICT = 300               # qwen3-vl uses ~200 tokens on thinking, then answers
SYSTEM_PROMPT = (
    "You are Max, a voice assistant on a Raspberry Pi 5. "
    "Start with the answer immediately. Use one short sentence. "
    "Answer in under 20 words. One short sentence, maybe two. "
    "Speak plainly. No preamble, no markdown, no lists, no explanations."
)

# Wake word config
WAKE_PHRASES = {
    # Demo-friendly broad triggers: very easy to wake, more false positives.
    "hey", "hi", "hello", "wake up", "ok", "okay",
    "hey max", "hey max wake up", "max wake up", "wake up max",
    "hi max", "hello max", "okay max", "ok max", "yo max",
    "hey there max", "come on max", "are you there max",
    "max are you there", "max can you hear me", "listen max",
    "max listen", "max please", "please max",
    "hay max", "hey macs", "hey mac", "hey mex", "hey mike",
    "hi mac", "hello mac", "ok mac", "wake up mac",
    "hey map", "hi map", "wake up map",
    "a max", "ey max", "he max", "hey maps", "hey mask",
    "hey mad", "hey man", "hey next", "hey mix",
    "max wake", "macs wake", "maps wake", "mask wake",
    "mac", "macs", "maps",
    "max",   # single-word fallback for when "hey" gets dropped
}
WAKE_WHISPER_MODEL = "tiny"   # lightweight model just for wake word detection
WAKE_CHUNK_DURATION = 4.0     # seconds of audio to transcribe per wake-word check
ACTIVE_TIMEOUT = 15           # seconds of silence before returning to wake mode

# --- TTS (piper) ---
PIPER_MODEL = os.path.join(BASE_DIR, "models", "hal.onnx")

# --- Audio output ---
AUDIO_OUTPUT_DEVICE = "hdmi:CARD=vc4hdmi0,DEV=0"  # HDMI 0 (change to vc4hdmi1 for HDMI 1)
AUDIO_VOLUME = 0.3  # 0.0 ~ 1.0, TTS playback volume

# --- LED (GPIO17) ---
LED_GPIO = 17
LED_LUX_THRESHOLD = 20    # lux below this → LED on, above → LED off

# --- Camera & Vision ---
CAMERA_DEVICE = 0                      # /dev/video0

# --- Gemini Vision (free tier: 1500 req/day) ---
# Get key at https://aistudio.google.com/apikey, then either:
#   export GEMINI_API_KEY="..."        (preferred)
# or set the value below directly.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = "gemini-2.5-flash"

# --- Weather (Open-Meteo, no API key) ---
WEATHER_CITY = "New York"
WEATHER_LAT  = 40.7128
WEATHER_LON  = -74.0060

# --- Audio ---
# Record at 48000 Hz (supported by USB mic + webrtcvad).
# Audio is decimated 3:1 to 16000 Hz before being passed to Whisper.
RECORD_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000   # Whisper requires 16000 Hz
CHANNELS = 1
VAD_MODE = 3                  # 0-3, higher = more aggressive noise filtering
SILENCE_DURATION = 1.5        # seconds of silence to stop recording
MAX_RECORD_DURATION = 30      # hard cap in seconds
PRE_SPEECH_BUFFER_S = 0.5     # seconds to keep before speech starts (avoid clipping)
