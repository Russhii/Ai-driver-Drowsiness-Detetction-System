
import cv2
import time
import argparse
import sys
import os

# local modules
from detector       import DrowsinessDetector
from alert_system   import AlertSystem, AlertLevel
from pressure_sensor import PressureSensor
from hud            import HUD



def parse_args():
    p = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    p.add_argument("--source",          default="0",
                   help="Camera index (0,1…) or path to video file")
    p.add_argument("--serial-port",     default=None,
                   help="Arduino serial port e.g. COM3 or /dev/ttyUSB0")
    p.add_argument("--simulate-drowsy", action="store_true",
                   help="Simulate drowsy pressure reading (demo mode)")
    p.add_argument("--no-window",       action="store_true",
                   help="Headless — no display window (logs only)")
    p.add_argument("--width",           type=int, default=640)
    p.add_argument("--height",          type=int, default=480)
    p.add_argument("--log-dir",         default="logs")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
def open_capture(source_str, width, height):
    """Open VideoCapture from index or file path."""
    src = int(source_str) if source_str.isdigit() else source_str
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source_str}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    print("=" * 55)
    print("  Driver Drowsiness Detection System  v1.0")
    print("=" * 55)
    print(f"  Source       : {args.source}")
    print(f"  Serial port  : {args.serial_port or 'None (simulation)'}")
    print(f"  Log dir      : {args.log_dir}")
    print(f"  Show window  : {not args.no_window}")
    print("=" * 55)
    print("  Press Q to quit | R to reset | S to screenshot")
    print("  Press D to toggle simulated drowsy pressure")
    print("=" * 55)

    # ── Initialise modules ───────────────────────────────────────────────────
    cap      = open_capture(args.source, args.width, args.height)
    detector = DrowsinessDetector()
    alerts   = AlertSystem(log_path=os.path.join(args.log_dir, "alert_log.csv"))
    sensor   = PressureSensor(port=args.serial_port)
    hud      = HUD(args.width, args.height)

    sim_drowsy = args.simulate_drowsy
    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0
    screenshot_n = 0

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video files; end on dead feed
            if isinstance(args.source, str) and not args.source.isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[INFO] End of stream.")
            break

        frame_count += 1

        # ── FPS counter ──────────────────────────────────────────────────────
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - fps_timer + 1e-6)
            fps_timer = now

        # ── Detection ────────────────────────────────────────────────────────
        detector.process_frame(frame)

        # ── Pressure sensor ──────────────────────────────────────────────────
        pressure = sensor.read(simulated_drowsy=sim_drowsy)

        # ── Alert system (fusion logic) ──────────────────────────────────────
        # Upgrade to DROWSY only when BOTH vision detects drowsiness
        # AND grip is lost (multi-modal confirmation)
        effective_state = detector.state
        if detector.state in ("WARNING", "DROWSY") and pressure.grip_lost:
            effective_state = "DROWSY"   # confirmed by two modalities
        elif detector.state == "DROWSY" and not pressure.grip_lost:
            effective_state = "WARNING"  # vision says drowsy but grip is fine

        alerts.update(effective_state, detector.drowsiness_score)

        # Override detector.state for HUD display
        display_detector_state = effective_state

        # ── HUD render ───────────────────────────────────────────────────────
        # Temporarily override state for rendering
        original_state  = detector.state
        detector.state  = display_detector_state

        hud.render(frame, detector, alerts, pressure)
        detector.state = original_state   # restore

        # FPS overlay
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (args.width - 100, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                    1, cv2.LINE_AA)

        # Sensor mode label
        mode_col = (100, 200, 100) if sensor.mode == "HARDWARE" else (100, 100, 255)
        cv2.putText(frame, f"Sensor: {sensor.mode}",
                    (8, args.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_col, 1, cv2.LINE_AA)

        # ── Print to console when state changes ──────────────────────────────
        if frame_count % 15 == 0:
            bar = int(detector.drowsiness_score * 20)
            bar_str = "#" * bar + "-" * (20 - bar)
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"State: {alerts.label:<8} "
                  f"Score: [{bar_str}] {detector.drowsiness_score:.2f}  "
                  f"Grip: {pressure.value:.0f}Pa  "
                  f"Eyes: {'CLOSED' if detector.eye_state.both_closed else 'open  '}",
                  end="", flush=True)

        # ── Display ──────────────────────────────────────────────────────────
        if not args.no_window:
            cv2.imshow("Driver Drowsiness Detection System", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                detector.reset()
                print("\n[INFO] Detector reset.")
            elif key == ord("s"):
                screenshot_n += 1
                path = os.path.join(args.log_dir,
                                    f"screenshot_{screenshot_n:03d}.jpg")
                cv2.imwrite(path, frame)
                print(f"\n[INFO] Screenshot saved: {path}")
            elif key == ord("d"):
                sim_drowsy = not sim_drowsy
                print(f"\n[INFO] Simulated drowsy pressure: {sim_drowsy}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    print(f"\n\n[INFO] {alerts.summary()}")
    cap.release()
    sensor.close()
    if not args.no_window:
        cv2.destroyAllWindows()
    print("[INFO] Session complete. Log saved to:", alerts.log_path)


if __name__ == "__main__":
    main()
