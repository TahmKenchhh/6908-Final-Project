"""
Camera capture for vision queries and live preview.
Singleton CameraManager keeps a single VideoCapture open so the HTTP
snapshot endpoint and voice-triggered vision share one device.
"""

import base64
import re
import threading
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_DEVICE

_VISION_KW = re.compile(
    r"\b(see|look|show|watch|describe|observe|what('s| is) (in front|around|that|this)|"
    r"what do you see|can you see|what am i|what are (you|we)|take a (look|picture|photo))\b",
    re.I,
)


def is_vision_query(text: str) -> bool:
    return bool(_VISION_KW.search(text))


class CameraManager:
    """Thread-safe shared camera instance."""
    _cap = None
    _lock = threading.Lock()

    @classmethod
    def _ensure_open(cls) -> bool:
        if cls._cap is None or not cls._cap.isOpened():
            cls._cap = cv2.VideoCapture(CAMERA_DEVICE)
        return cls._cap.isOpened()

    @classmethod
    def grab_frame(cls):
        with cls._lock:
            if not cls._ensure_open():
                return None
            cls._cap.grab()
            ret, frame = cls._cap.read()
            return frame if ret else None

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._cap is not None:
                cls._cap.release()
                cls._cap = None


def capture_frame_b64() -> str | None:
    """
    Grab a frame, downscale to ~384px for faster VLM inference,
    encode as JPEG and return base64 string.
    """
    frame = CameraManager.grab_frame()
    if frame is None:
        print(f"[camera] capture failed on device {CAMERA_DEVICE}")
        return None
    h, w = frame.shape[:2]
    if max(h, w) > 384:
        scale = 384 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def snapshot_jpeg(max_width: int = 640) -> bytes | None:
    """
    Grab a frame at higher resolution for the browser live preview.
    Returns raw JPEG bytes (no base64).
    """
    frame = CameraManager.grab_frame()
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()
