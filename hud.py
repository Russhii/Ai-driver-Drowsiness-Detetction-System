"""
hud.py  (v2.0 — larger HUD)
============================
Head-Up Display renderer with bigger text, larger panel, and
a wider score bar so everything is easy to read at a glance.
"""

import cv2
import numpy as np
import time

C = {
    "green"    : (34, 200, 34),
    "orange"   : (0, 165, 255),
    "red"      : (0, 50, 220),
    "dark_red" : (0, 0, 160),
    "white"    : (255, 255, 255),
    "black"    : (0, 0, 0),
    "gray"     : (180, 180, 180),
    "dark_gray": (70, 70, 70),
    "panel_bg" : (15, 15, 15),
    "bar_bg"   : (70, 70, 70),
}

STATE_COLOR = {
    "ALERT"  : C["green"],
    "WARNING": C["orange"],
    "DROWSY" : C["red"],
    "NO_FACE": C["gray"],
}


class HUD:
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, frame_w=640, frame_h=480):
        self.w = frame_w
        self.h = frame_h
        self._flash_counter = 0

    def render(self, frame, detector, alert_system, pressure_reading):
        h, w = frame.shape[:2]
        self.w, self.h = w, h

        self._draw_flash(frame, alert_system)
        self._draw_status_banner(frame, detector, alert_system)
        self._draw_left_panel(frame, detector, pressure_reading)
        self._draw_score_bar(frame, detector.drowsiness_score)
        self._draw_bottom_bar(frame, alert_system)
        return frame

    # ── status banner (taller) ───────────────────────────────────────────────
    def _draw_status_banner(self, frame, detector, alert_system):
        state = detector.state
        color = STATE_COLOR.get(state, C["gray"])

        BANNER_H = 72          # was 52
        cv2.rectangle(frame, (0, 0), (self.w, BANNER_H), C["panel_bg"], -1)
        cv2.rectangle(frame, (0, 0), (self.w, BANNER_H), color, 3)

        # State label — much bigger
        cv2.putText(frame, f"  {state}", (8, 52),
                    self.FONT_BOLD, 1.7, color, 3, cv2.LINE_AA)

        # Score — bigger
        cv2.putText(frame, f"Score: {detector.drowsiness_score:.2f}",
                    (self.w - 240, 52),
                    self.FONT, 1.0, C["white"], 2, cv2.LINE_AA)

        # Alerts count
        cv2.putText(frame, f"Alerts: {alert_system.alert_count}",
                    (self.w - 240, 22),
                    self.FONT, 0.65, C["gray"], 1, cv2.LINE_AA)

    # ── left panel (wider + taller rows) ────────────────────────────────────
    def _draw_left_panel(self, frame, detector, pressure_reading):
        px, py = 8, 80          # start below taller banner
        pw, ph = 290, 230       # was 200 x 180

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), C["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), C["dark_gray"], 1)

        ear   = getattr(detector, 'smooth_ear', detector.eye_state.ear)
        mar   = getattr(detector, 'smooth_mar', detector.mouth_state.mor)
        pitch = getattr(detector.head_pose, 'pitch_deg', 0.0)

        rows = [
            ("EYES",
             f"EAR {ear:.2f}   {'CLOSED' if detector.eye_state.both_closed else 'OPEN'}",
             C["red"] if detector.eye_state.both_closed else C["green"]),

            ("YAWN",
             f"MAR {mar:.2f}   {'YAWNING' if detector.mouth_state.yawning else 'NORMAL'}",
             C["orange"] if detector.mouth_state.yawning else C["green"]),

            ("HEAD",
             f"{pitch:+.1f} deg   {'DROP' if detector.head_pose.tilted else 'OK'}",
             C["orange"] if detector.head_pose.tilted else C["green"]),

            ("GRIP",
             pressure_reading.status_label,
             C["red"] if pressure_reading.grip_lost else C["green"]),
        ]

        ROW_H = 46              # was 34 — more breathing room
        for i, (label, val, col) in enumerate(rows):
            y = py + 38 + i * ROW_H

            # label key (gray, medium)
            cv2.putText(frame, label, (px + 10, y),
                        self.FONT, 0.60, C["gray"], 1, cv2.LINE_AA)

            # value (coloured, larger)
            cv2.putText(frame, val, (px + 78, y),
                        self.FONT, 0.65, col, 2, cv2.LINE_AA)

        # Grip bar (taller)
        bar_y = py + ph - 26
        self._draw_mini_bar(frame, px + 10, bar_y,
                            pw - 20, 14,
                            pressure_reading.grip_percent / 100.0,
                            pressure_reading.bar_color_bgr)

        self._draw_landmarks(frame, detector)

        # Calibration progress bar (shows during first 15 s)
        cal = getattr(detector, 'calibrator', None)
        if cal and not cal.is_done:
            prog = cal.progress
            bar_y2 = py + ph + 10
            cv2.putText(frame, f"Calibrating... {int(prog*100)}%",
                        (px + 10, bar_y2 - 4),
                        self.FONT, 0.55, C["orange"], 1, cv2.LINE_AA)
            cv2.rectangle(frame, (px + 10, bar_y2),
                          (px + pw - 10, bar_y2 + 10), C["bar_bg"], -1)
            fill = int((pw - 20) * prog)
            if fill > 0:
                cv2.rectangle(frame, (px + 10, bar_y2),
                              (px + 10 + fill, bar_y2 + 10), C["orange"], -1)

    # ── score bar (wider) ────────────────────────────────────────────────────
    def _draw_score_bar(self, frame, score):
        BAR_W  = 30             # was 18
        bar_x  = self.w - BAR_W - 10
        bar_top = 80
        bar_bot = self.h - 50
        bar_h   = bar_bot - bar_top

        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + BAR_W, bar_bot),
                      C["bar_bg"], -1)

        fill_h = int(bar_h * score)
        fill_y = bar_bot - fill_h
        col = C["green"] if score < 0.35 else (C["orange"] if score < 0.55 else C["red"])
        if fill_h > 0:
            cv2.rectangle(frame, (bar_x, fill_y), (bar_x + BAR_W, bar_bot), col, -1)

        # border
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + BAR_W, bar_bot),
                      C["dark_gray"], 1)

        cv2.putText(frame, "SCORE", (bar_x - 8, bar_bot + 18),
                    self.FONT, 0.45, C["gray"], 1, cv2.LINE_AA)

        # threshold lines
        warn_y   = bar_bot - int(bar_h * 0.30)
        drowsy_y = bar_bot - int(bar_h * 0.55)
        cv2.line(frame, (bar_x - 4, warn_y),   (bar_x + BAR_W + 4, warn_y),   C["orange"], 1)
        cv2.line(frame, (bar_x - 4, drowsy_y), (bar_x + BAR_W + 4, drowsy_y), C["red"],    1)

        # small % label
        cv2.putText(frame, f"{int(score*100)}%",
                    (bar_x - 2, fill_y - 6),
                    self.FONT, 0.45, col, 1, cv2.LINE_AA)

    # ── bottom bar ───────────────────────────────────────────────────────────
    def _draw_bottom_bar(self, frame, alert_system):
        BAR_H = 38              # was 32
        bar_y = self.h - BAR_H
        cv2.rectangle(frame, (0, bar_y), (self.w, self.h), C["panel_bg"], -1)

        elapsed = int(alert_system.session_duration)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}",
                    (12, self.h - 10),
                    self.FONT, 0.65, C["gray"], 1, cv2.LINE_AA)

        cv2.putText(frame, "Driver Drowsiness Detection System v2.0",
                    (self.w // 2 - 185, self.h - 10),
                    self.FONT, 0.55, C["gray"], 1, cv2.LINE_AA)

        cv2.putText(frame, time.strftime('%H:%M:%S'),
                    (self.w - 100, self.h - 10),
                    self.FONT, 0.60, C["gray"], 1, cv2.LINE_AA)

    # ── DROWSY flash ─────────────────────────────────────────────────────────
    def _draw_flash(self, frame, alert_system):
        if alert_system.level != 2:
            self._flash_counter = 0
            return

        self._flash_counter = (self._flash_counter + 1) % 30
        if self._flash_counter < 15:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        if self._flash_counter < 20:
            txt = "! DROWSY DRIVER !"
            (tw, _), _ = cv2.getTextSize(txt, self.FONT_BOLD, 2.0, 4), 0
            tx = (self.w - tw) // 2
            ty = self.h // 2 + 30
            cv2.putText(frame, txt, (tx + 2, ty + 2),
                        self.FONT_BOLD, 2.0, C["black"], 5, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx, ty),
                        self.FONT_BOLD, 2.0, C["white"], 4, cv2.LINE_AA)

    # ── landmarks ────────────────────────────────────────────────────────────
    def _draw_landmarks(self, frame, detector):
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
            cv2.circle(frame, (px, py), 3, eye_col, -1, cv2.LINE_AA)   # bigger dots

        for idx in _MOUTH_TOP + _MOUTH_BOTTOM + _MOUTH_HORIZ:
            px = int(lm[idx].x * w)
            py = int(lm[idx].y * h)
            cv2.circle(frame, (px, py), 3, mouth_col, -1, cv2.LINE_AA)

    # ── mini bar ─────────────────────────────────────────────────────────────
    def _draw_mini_bar(self, frame, x, y, w, h, ratio, color):
        cv2.rectangle(frame, (x, y), (x + w, y + h), C["bar_bg"], -1)
        fill_w = int(w * ratio)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), C["dark_gray"], 1)