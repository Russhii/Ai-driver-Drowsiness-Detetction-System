"""
hud.py
======
Head-Up Display renderer.

Draws all overlays onto the live camera frame:
  - Status banner (ALERT / WARNING / DROWSY)
  - Drowsiness score bar
  - Live EAR / MAR / Head-pitch indicators
  - MediaPipe landmark overlays (eye & mouth contours)
  - Steering pressure gauge
  - Alert flash effect
  - Session stats (time, alert count)
"""

import cv2
import numpy as np
import time


# ── colour palette (BGR) ────────────────────────────────────────────────────
C = {
    "green"      : (34, 180, 34),
    "orange"     : (0, 165, 255),
    "red"        : (0, 50, 220),
    "dark_red"   : (0, 0, 160),
    "white"      : (255, 255, 255),
    "black"      : (0, 0, 0),
    "gray"       : (160, 160, 160),
    "dark_gray"  : (50, 50, 50),
    "panel_bg"   : (20, 20, 20),
    "bar_bg"     : (60, 60, 60),
}

STATE_COLOR = {
    "ALERT"   : C["green"],
    "WARNING" : C["orange"],
    "DROWSY"  : C["red"],
    "NO_FACE" : C["gray"],
}


# ══════════════════════════════════════════════════════════════════════════════
class HUD:
    """
    All drawing is done in-place on the frame passed to render().
    """

    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, frame_w=640, frame_h=480):
        self.w = frame_w
        self.h = frame_h
        self._flash_on = False
        self._flash_counter = 0

    # ── main entry point ────────────────────────────────────────────────────
    def render(self, frame, detector, alert_system, pressure_reading):
        """
        Draw everything onto frame (in-place).
        Returns frame for convenience.
        """
        h, w = frame.shape[:2]
        self.w, self.h = w, h

        # 1. Flash overlay for DROWSY state
        self._draw_flash(frame, alert_system)

        # 2. Top status banner
        self._draw_status_banner(frame, detector, alert_system)

        # 3. Left panel: indicators
        self._draw_left_panel(frame, detector, pressure_reading)

        # 4. Bottom bar: session stats
        self._draw_bottom_bar(frame, alert_system)

        # 5. Drowsiness score bar (right side)
        self._draw_score_bar(frame, detector.drowsiness_score)

        return frame

    # ── status banner ───────────────────────────────────────────────────────
    def _draw_status_banner(self, frame, detector, alert_system):
        state = detector.state
        color = STATE_COLOR.get(state, C["gray"])

        # Banner rectangle
        cv2.rectangle(frame, (0, 0), (self.w, 52), C["panel_bg"], -1)
        cv2.rectangle(frame, (0, 0), (self.w, 52), color, 2)

        # State label
        label = f"  {state}"
        cv2.putText(frame, label, (8, 38),
                    self.FONT_BOLD, 1.15, color, 2, cv2.LINE_AA)

        # Score text
        score_txt = f"Score: {detector.drowsiness_score:.2f}"
        cv2.putText(frame, score_txt, (self.w - 190, 35),
                    self.FONT, 0.7, C["white"], 1, cv2.LINE_AA)

        # Alerts count
        alert_txt = f"Alerts: {alert_system.alert_count}"
        cv2.putText(frame, alert_txt, (self.w - 190, 15),
                    self.FONT, 0.5, C["gray"], 1, cv2.LINE_AA)

    # ── left indicator panel ────────────────────────────────────────────────
    def _draw_left_panel(self, frame, detector, pressure_reading):
        panel_x, panel_y = 8, 62
        panel_w, panel_h = 200, 180

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      C["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      C["dark_gray"], 1)

        ear = getattr(detector, 'smooth_ear', detector.eye_state.ear)
        mar = getattr(detector, 'smooth_mar', detector.mouth_state.mor)
        pitch = getattr(detector.head_pose, 'pitch_deg', 0.0)

        indicators = [
            ("EYES",
             f"EAR {ear:.2f}  {'CLOSED' if detector.eye_state.both_closed else 'OPEN'}",
             C["red"] if detector.eye_state.both_closed else C["green"]),

            ("YAWN",
             f"MAR {mar:.2f}  {'YAWNING' if detector.mouth_state.yawning else 'NORMAL'}",
             C["orange"] if detector.mouth_state.yawning else C["green"]),

            ("HEAD",
             f"{pitch:+.1f} deg  {'DROP' if detector.head_pose.tilted else 'OK'}",
             C["orange"] if detector.head_pose.tilted else C["green"]),

            ("GRIP",
             pressure_reading.status_label,
             C["red"] if pressure_reading.grip_lost else C["green"]),
        ]

        for i, (label, val, col) in enumerate(indicators):
            y = panel_y + 24 + i * 34
            cv2.putText(frame, label, (panel_x + 8, y),
                        self.FONT, 0.42, C["gray"], 1, cv2.LINE_AA)
            cv2.putText(frame, val, (panel_x + 56, y),
                        self.FONT, 0.44, col, 1, cv2.LINE_AA)

        # Mini grip bar
        bar_y = panel_y + panel_h - 18
        self._draw_mini_bar(frame, panel_x + 8, bar_y,
                            panel_w - 16, 10,
                            pressure_reading.grip_percent / 100.0,
                            pressure_reading.bar_color_bgr,
                            "GRIP")

        # ── Draw MediaPipe landmark overlays ─────────────────────────────────
        self._draw_landmarks(frame, detector)

    # ── drowsiness score bar ────────────────────────────────────────────────
    def _draw_score_bar(self, frame, score):
        bar_x = self.w - 28
        bar_top = 62
        bar_bot = self.h - 45
        bar_h = bar_bot - bar_top

        # Background track
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 18, bar_bot),
                      C["bar_bg"], -1)

        # Fill
        fill_h = int(bar_h * score)
        fill_y = bar_bot - fill_h
        if score < 0.35:
            col = C["green"]
        elif score < 0.55:
            col = C["orange"]
        else:
            col = C["red"]

        if fill_h > 0:
            cv2.rectangle(frame, (bar_x, fill_y), (bar_x + 18, bar_bot),
                          col, -1)

        # Label
        cv2.putText(frame, "SCORE", (bar_x - 5, bar_bot + 14),
                    self.FONT, 0.35, C["gray"], 1, cv2.LINE_AA)

        # Threshold lines
        warn_y  = bar_bot - int(bar_h * 0.30)
        drowsy_y = bar_bot - int(bar_h * 0.55)
        cv2.line(frame, (bar_x - 2, warn_y), (bar_x + 20, warn_y),
                 C["orange"], 1)
        cv2.line(frame, (bar_x - 2, drowsy_y), (bar_x + 20, drowsy_y),
                 C["red"], 1)

    # ── bottom stats bar ────────────────────────────────────────────────────
    def _draw_bottom_bar(self, frame, alert_system):
        bar_y = self.h - 32
        cv2.rectangle(frame, (0, bar_y), (self.w, self.h),
                      C["panel_bg"], -1)

        elapsed = int(alert_system.session_duration)
        mins, secs = divmod(elapsed, 60)
        time_txt = f"Session: {mins:02d}:{secs:02d}"
        cv2.putText(frame, time_txt, (10, self.h - 10),
                    self.FONT, 0.55, C["gray"], 1, cv2.LINE_AA)

        system_txt = "Driver Drowsiness Detection System v1.0"
        tw, _ = cv2.getTextSize(system_txt, self.FONT, 0.45, 1)[0], 0
        cv2.putText(frame, system_txt,
                    (self.w // 2 - 150, self.h - 10),
                    self.FONT, 0.45, C["gray"], 1, cv2.LINE_AA)

        fps_txt = f"{time.strftime('%H:%M:%S')}"
        cv2.putText(frame, fps_txt, (self.w - 90, self.h - 10),
                    self.FONT, 0.5, C["gray"], 1, cv2.LINE_AA)

    # ── DROWSY flash overlay ─────────────────────────────────────────────────
    def _draw_flash(self, frame, alert_system):
        if alert_system.level != 2:   # AlertLevel.DROWSY
            self._flash_counter = 0
            return

        self._flash_counter = (self._flash_counter + 1) % 30
        if self._flash_counter < 15:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Big centred warning text
        if self._flash_counter < 20:
            txt = "! DROWSY DRIVER !"
            (tw, th), _ = cv2.getTextSize(txt, self.FONT_BOLD, 1.5, 3)
            tx = (self.w - tw) // 2
            ty = self.h // 2 + 20
            cv2.putText(frame, txt, (tx + 2, ty + 2),
                        self.FONT_BOLD, 1.5, C["black"], 4, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx, ty),
                        self.FONT_BOLD, 1.5, C["white"], 3, cv2.LINE_AA)

    # ── utility ─────────────────────────────────────────────────────────────
    def _draw_landmarks(self, frame, detector):
        """Draw eye & mouth landmark dots from MediaPipe when available."""
        lm = getattr(detector, 'landmarks', None)
        if lm is None:
            return
        h, w = frame.shape[:2]
        try:
            from detector import _L_EYE, _R_EYE, _MOUTH_TOP, _MOUTH_BOTTOM, _MOUTH_HORIZ
        except ImportError:
            return

        eye_col   = C["red"] if detector.eye_state.both_closed else C["green"]
        mouth_col = C["orange"] if detector.mouth_state.yawning else (100, 220, 255)

        for idx in _L_EYE + _R_EYE:
            px = int(lm[idx].x * w)
            py = int(lm[idx].y * h)
            cv2.circle(frame, (px, py), 2, eye_col, -1, cv2.LINE_AA)

        for idx in _MOUTH_TOP + _MOUTH_BOTTOM + _MOUTH_HORIZ:
            px = int(lm[idx].x * w)
            py = int(lm[idx].y * h)
            cv2.circle(frame, (px, py), 2, mouth_col, -1, cv2.LINE_AA)

    def _draw_mini_bar(self, frame, x, y, w, h, ratio, color, label):
        cv2.rectangle(frame, (x, y), (x + w, y + h), C["bar_bg"], -1)
        fill_w = int(w * ratio)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), C["dark_gray"], 1)
