"""
alert_system.py
===============
Multi-level alert engine.

Level 0 — ALERT   : driver is fine, no action
Level 1 — WARNING : early signs, visual warning on screen
Level 2 — DROWSY  : confirmed drowsiness, loud beep + visual flash

Fusion rule (from Guardian Alert paper):
  Alert is triggered ONLY when drowsiness_score crosses threshold
  AND has been sustained for minimum duration.
  This prevents false alarms from brief blinks.

Also logs every alert event to logs/alert_log.csv
"""

import time
import csv
import os
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
class AlertLevel:
    ALERT   = 0
    WARNING = 1
    DROWSY  = 2

    LABELS = {0: "ALERT", 1: "WARNING", 2: "DROWSY"}
    COLORS_BGR = {
        0: (34, 139, 34),    # green
        1: (0, 165, 255),    # orange
        2: (0, 0, 220),      # red
    }


# ══════════════════════════════════════════════════════════════════════════════
class AlertSystem:
    """
    Manages alert state, timing, cooldown, and CSV logging.

    Parameters
    ----------
    log_path : str
        Path to the CSV log file.
    warning_hold_sec : float
        Seconds WARNING must be sustained before upgrading to DROWSY alert.
    cooldown_sec : float
        Seconds to wait before re-triggering a DROWSY alert.
    """

    def __init__(self,
                 log_path="logs/alert_log.csv",
                 warning_hold_sec=2.0,
                 cooldown_sec=8.0):

        self.log_path         = log_path
        self.warning_hold_sec = warning_hold_sec
        self.cooldown_sec     = cooldown_sec

        self._level           = AlertLevel.ALERT
        self._warning_start   = None
        self._last_alert_time = 0.0
        self._alert_count     = 0
        self._session_start   = time.time()
        self._beep_phase      = 0   # for software beep animation

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._init_log()

    # ── public API ─────────────────────────────────────────────────────────
    def update(self, detector_state: str, drowsiness_score: float):
        """
        Call once per frame with the detector's current state & score.
        Returns the current AlertLevel int.
        """
        now = time.time()

        if detector_state == "ALERT" or detector_state == "NO_FACE":
            self._level         = AlertLevel.ALERT
            self._warning_start = None
            return self._level

        if detector_state == "WARNING":
            if self._warning_start is None:
                self._warning_start = now
            held = now - self._warning_start
            if held >= self.warning_hold_sec:
                self._level = AlertLevel.WARNING
            else:
                self._level = AlertLevel.ALERT
            return self._level

        if detector_state == "DROWSY":
            self._level = AlertLevel.DROWSY
            # Trigger timed alert (with cooldown)
            if now - self._last_alert_time >= self.cooldown_sec:
                self._last_alert_time = now
                self._alert_count += 1
                self._log_event(now, drowsiness_score)
            return self._level

        return self._level

    @property
    def level(self):
        return self._level

    @property
    def label(self):
        return AlertLevel.LABELS[self._level]

    @property
    def color_bgr(self):
        return AlertLevel.COLORS_BGR[self._level]

    @property
    def alert_count(self):
        return self._alert_count

    @property
    def session_duration(self):
        return time.time() - self._session_start

    def beep_tick(self):
        """Returns True on every Nth call — used to flash/beep in the UI."""
        self._beep_phase = (self._beep_phase + 1) % 15
        return self._beep_phase == 0

    # ── logging ────────────────────────────────────────────────────────────
    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "elapsed_sec",
                                  "alert_level", "drowsiness_score"])

    def _log_event(self, timestamp, score):
        elapsed = timestamp - self._session_start
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                f"{elapsed:.1f}",
                AlertLevel.LABELS[self._level],
                f"{score:.3f}"
            ])

    def summary(self):
        mins = self.session_duration / 60
        return (f"Session: {mins:.1f} min | "
                f"Total alerts triggered: {self._alert_count}")
