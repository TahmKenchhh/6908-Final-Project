"""
BH1750 ambient light sensor via I2C (address 0x23, bus 1).
Runs a background thread that reads lux every second and calls
an optional callback with the latest value.
"""

import re
import threading
import time

_LIGHT_KW = re.compile(
    r"\b(light|bright|dark|dim|lux|lamp|lights|illuminat|luminous|sunny|gloomy|visibility)\b",
    re.I,
)

def is_light_query(text: str) -> bool:
    return bool(_LIGHT_KW.search(text))

BH1750_ADDR    = 0x23
BH1750_BUS     = 1
_CONT_HRES     = 0x10   # continuous high-resolution mode (~1 lux resolution)
_READ_DELAY_S  = 0.18   # measurement time for high-res mode


class LightSensor:
    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._lux: float | None = None
        self._lock = threading.Lock()
        self._bus = None
        self._running = False

    def start(self, on_update=None) -> bool:
        """
        Start background reading. on_update(lux: float) called each cycle.
        Returns False if sensor can't be opened.
        """
        try:
            from smbus2 import SMBus
            self._bus = SMBus(BH1750_BUS)
            self._bus.write_byte(BH1750_ADDR, _CONT_HRES)
        except Exception as e:
            print(f"[light] BH1750 init failed: {e}")
            return False

        self._running = True
        t = threading.Thread(target=self._loop, args=(on_update,), daemon=True)
        t.start()
        print(f"[light] BH1750 ready on bus {BH1750_BUS} addr 0x{BH1750_ADDR:02X}")
        return True

    def _loop(self, on_update) -> None:
        while self._running:
            lux = self._read()
            if lux is not None:
                with self._lock:
                    self._lux = lux
                if on_update:
                    try:
                        on_update(lux)
                    except Exception:
                        pass
            time.sleep(self._interval)

    def _read(self) -> float | None:
        try:
            time.sleep(_READ_DELAY_S)
            data = self._bus.read_i2c_block_data(BH1750_ADDR, _CONT_HRES, 2)
            raw = (data[0] << 8) | data[1]
            return round(raw / 1.2, 1)
        except Exception:
            return None

    @property
    def lux(self) -> float | None:
        with self._lock:
            return self._lux
