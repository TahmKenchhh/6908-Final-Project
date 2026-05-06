"""
LED controller via lgpio (RPi5 compatible).
Also provides voice command detection for turning the LED on/off.
"""

import re
import lgpio

_ON_RE  = re.compile(r"\b(turn on|switch on|light on|lights on|enable|activate)\b.{0,15}\b(light|lights|led|lamp)\b|\b(light|lights|led|lamp)\b.{0,10}\b(on)\b", re.I)
_OFF_RE = re.compile(r"\b(turn off|switch off|light off|lights off|disable|deactivate)\b.{0,15}\b(light|lights|led|lamp)\b|\b(light|lights|led|lamp)\b.{0,10}\b(off)\b", re.I)


def detect_led_command(text: str) -> bool | None:
    """
    Returns True to turn on, False to turn off, None if no LED command found.
    """
    if _ON_RE.search(text):
        return True
    if _OFF_RE.search(text):
        return False
    return None

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LED_GPIO, LED_LUX_THRESHOLD

_chip = None
_initialized = False


def _init():
    global _chip, _initialized
    if _initialized:
        return
    try:
        _chip = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(_chip, LED_GPIO)
        _initialized = True
    except Exception as e:
        print(f"[led] init failed: {e}")


def set_led(on: bool) -> None:
    _init()
    if _initialized:
        lgpio.gpio_write(_chip, LED_GPIO, 1 if on else 0)


def on_lux(lux: float) -> None:
    """Called by LightSensor callback — turn LED on when dark."""
    set_led(lux < LED_LUX_THRESHOLD)
