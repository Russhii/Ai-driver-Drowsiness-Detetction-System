"""
pressure_sensor.py
==================
Steering wheel pressure sensor interface.

In real hardware  → reads from Arduino serial port (USB).
In simulation mode → generates realistic grip-pressure data
                     that slowly drops when 'drowsy' is simulated.

Matches the Guardian Alert paper approach:
  - Pressure threshold : 200 Pa
  - Below threshold + eyes closed → confirmed drowsy

Serial protocol (Arduino side):
  Arduino sends one integer per line: e.g.  "342\n"
  Values 0–1023 (10-bit ADC), mapped to 0–500 Pa.
"""

import time
import random
import math


# ══════════════════════════════════════════════════════════════════════════════
class PressureSensor:
    """
    Unified interface for real or simulated steering pressure.

    Parameters
    ----------
    port : str or None
        Serial port string e.g. 'COM3' or '/dev/ttyUSB0'.
        If None → simulation mode.
    threshold_pa : float
        Pressure below this value = driver losing grip.
    """

    PRESSURE_THRESHOLD = 200.0   # Pa — from Guardian Alert paper

    def __init__(self, port=None, baud=9600, threshold_pa=None):
        self.threshold = threshold_pa or self.PRESSURE_THRESHOLD
        self._sim_mode = True
        self._serial   = None
        self._last_val = 350.0
        self._t0       = time.time()

        if port:
            self._try_connect(port, baud)

    def _try_connect(self, port, baud):
        try:
            import serial
            self._serial  = serial.Serial(port, baud, timeout=1)
            self._sim_mode = False
            print(f"[PressureSensor] Connected to {port} @ {baud} baud")
        except Exception as e:
            print(f"[PressureSensor] Serial unavailable ({e}). Using simulation.")
            self._sim_mode = True

    # ── public API ──────────────────────────────────────────────────────────
    def read(self, simulated_drowsy=False):
        """
        Returns a PressureReading namedtuple.

        Parameters
        ----------
        simulated_drowsy : bool
            In sim mode, if True the pressure slowly drops to mimic
            a drowsy driver loosening their grip.
        """
        if not self._sim_mode and self._serial:
            return self._read_hardware()
        return self._read_simulated(simulated_drowsy)

    @property
    def mode(self):
        return "HARDWARE" if not self._sim_mode else "SIMULATION"

    # ── hardware read ───────────────────────────────────────────────────────
    def _read_hardware(self):
        try:
            line = self._serial.readline().decode("utf-8").strip()
            raw  = int(line)
            pa   = self._adc_to_pa(raw)
            return PressureReading(pa, self.threshold)
        except Exception:
            # On read error fall back to last known value
            return PressureReading(self._last_val, self.threshold)

    @staticmethod
    def _adc_to_pa(raw_10bit):
        """Map 0–1023 ADC reading → 0–500 Pa."""
        return (raw_10bit / 1023.0) * 500.0

    # ── simulation ──────────────────────────────────────────────────────────
    def _read_simulated(self, drowsy):
        t = time.time() - self._t0

        if drowsy:
            # Grip drops over ~5 seconds
            target = max(50.0, 350.0 - (t % 20) * 30)
        else:
            # Natural fluctuation around 320 Pa
            target = 320.0 + 40 * math.sin(t * 0.4) + random.gauss(0, 15)
            target = max(150.0, min(500.0, target))

        # Low-pass smooth
        self._last_val = 0.2 * target + 0.8 * self._last_val
        return PressureReading(self._last_val, self.threshold)

    def close(self):
        if self._serial:
            self._serial.close()


# ══════════════════════════════════════════════════════════════════════════════
class PressureReading:
    """Holds one pressure sensor reading with helper properties."""

    def __init__(self, value_pa: float, threshold_pa: float):
        self.value     = value_pa
        self.threshold = threshold_pa

    @property
    def grip_lost(self):
        """True when pressure drops below the safety threshold."""
        return self.value < self.threshold

    @property
    def grip_percent(self):
        """Normalised grip strength 0–100%."""
        return float(min(100.0, (self.value / 500.0) * 100.0))

    @property
    def status_label(self):
        return "GRIP LOST" if self.grip_lost else "GRIP OK"

    @property
    def bar_color_bgr(self):
        """Green → yellow → red as grip drops."""
        pct = self.grip_percent / 100.0
        if pct > 0.5:
            r = int((1 - pct) * 2 * 255)
            g = 200
        else:
            r = 220
            g = int(pct * 2 * 200)
        return (0, g, r)

    def __repr__(self):
        return (f"PressureReading(value={self.value:.1f} Pa, "
                f"threshold={self.threshold} Pa, grip_lost={self.grip_lost})")
