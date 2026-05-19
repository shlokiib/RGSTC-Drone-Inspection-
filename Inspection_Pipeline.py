"""
Insulator Inspection Drone Script
==================================
Platform  : Raspberry Pi (mounted on drone)
Camera    : USB camera (cv2.VideoCapture)
Flight FC : Pixhawk connected via USB → MAVLink (dronekit)
ML Model  : YOLOv8-nano (ultralytics)

Modes
-----
AUTO   – Pixhawk follows a pre-planned mission (Mission Planner)
VISUAL – RPi takes over: centres insulator in frame, holds 8 m stand-off
"""

import cv2
import math
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  – edit these to match your hardware / mission requirements
# ─────────────────────────────────────────────────────────────────────────────

SERIAL_PORT          = "/dev/ttyACM0"   # Pixhawk USB port on RPi
BAUD_RATE            = 57600

MODEL_PATH           = "yolov8n_insulator.pt"   # your trained weights file
CAMERA_INDEX         = 0                         # USB camera index

FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
FRAME_CX             = FRAME_WIDTH  // 2
FRAME_CY             = FRAME_HEIGHT // 2

# ── Camera intrinsics (calibrate for your lens) ───────────────────────────────
FOCAL_LENGTH_PX      = 600.0    # focal length in pixels
INSULATOR_REAL_WIDTH = 0.30     # real-world width of insulator disc (metres)

# ── Stand-off / control parameters ────────────────────────────────────────────
SAFE_DISTANCE_M      = 8.0      # desired distance to insulator (metres)
DIST_TOLERANCE_M     = 0.5      # ±0.5 m before nudging fore/aft
CENTER_TOLERANCE_PX  = 30       # pixel dead-zone around frame centre
VELOCITY_STEP        = 0.4      # m/s increment per control tick (NED frame)
MAX_VELOCITY         = 1.5      # hard cap on any axis (m/s)

# ── Timing ────────────────────────────────────────────────────────────────────
CONTROL_HZ           = 10       # visual-control loop frequency
DETECTION_CONF       = 0.50     # YOLO confidence threshold
CLASS_ID_INSULATOR   = 0        # class index in your custom model

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("insulator_drone")


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single YOLO detection result for an insulator."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixel coords)
    confidence: float
    cx: int = field(init=False)
    cy: int = field(init=False)
    width_px: int = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.width_px = x2 - x1
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class ControlState:
    mode: str = "AUTO"           # "AUTO" or "VISUAL"
    target_locked: bool = False
    estimated_distance: float = 0.0
    last_detection: Optional[Detection] = None


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_distance(width_px: int) -> float:
    """
    Pinhole-camera distance estimate.

    distance = (real_width × focal_length) / pixel_width
    """
    if width_px <= 0:
        return float("inf")
    return (INSULATOR_REAL_WIDTH * FOCAL_LENGTH_PX) / width_px


# ─────────────────────────────────────────────────────────────────────────────
# MAVLINK VELOCITY COMMAND
# ─────────────────────────────────────────────────────────────────────────────

def send_ned_velocity(vehicle, vx: float, vy: float, vz: float, duration: float = 0.1):
    """
    Send a body-frame velocity setpoint via MAVLink SET_POSITION_TARGET_LOCAL_NED.

    NED convention:
        vx  → forward (+) / backward (-)
        vy  → right (+)   / left (-)
        vz  → down (+)    / up (-)   ← note: positive = descend
    """
    # clamp each axis
    vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx))
    vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy))
    vz = max(-MAX_VELOCITY, min(MAX_VELOCITY, vz))

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,                                   # time_boot_ms (not used)
        0, 0,                                # target system, component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # body-relative frame
        0b0000111111000111,                  # type_mask: only velocity
        0, 0, 0,                             # x, y, z (ignored)
        vx, vy, vz,                          # velocities (m/s)
        0, 0, 0,                             # accelerations (ignored)
        0, 0,                                # yaw, yaw_rate (ignored)
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def stop_drone(vehicle):
    """Bring all axes to zero velocity."""
    send_ned_velocity(vehicle, 0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL CONTROL LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity_command(detection: Detection) -> Tuple[float, float, float]:
    """
    Given the best insulator detection, compute (vx, vy, vz) in NED body-frame
    to:
      1. Centre the insulator horizontally  (vy)
      2. Centre the insulator vertically    (vz)
      3. Hold SAFE_DISTANCE_M stand-off     (vx)
    """
    err_x = detection.cx - FRAME_CX   # +ve → target is right of centre
    err_y = detection.cy - FRAME_CY   # +ve → target is below centre

    dist = estimate_distance(detection.width_px)

    # ── Lateral (left/right) ─────────────────────────────────────────────────
    if abs(err_x) > CENTER_TOLERANCE_PX:
        vy = VELOCITY_STEP * math.copysign(1, err_x)
    else:
        vy = 0.0

    # ── Vertical (up/down) – NED: vz positive = DOWN ─────────────────────────
    if abs(err_y) > CENTER_TOLERANCE_PX:
        vz = VELOCITY_STEP * math.copysign(1, err_y)   # below → move down
    else:
        vz = 0.0

    # ── Forward/backward – maintain 8 m stand-off ────────────────────────────
    dist_err = dist - SAFE_DISTANCE_M
    if abs(dist_err) > DIST_TOLERANCE_M:
        # positive dist_err → too far → move forward (positive vx)
        vx = VELOCITY_STEP * math.copysign(1, dist_err)
    else:
        vx = 0.0

    return vx, vy, vz


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def best_detection(results) -> Optional[Detection]:
    """
    From a YOLO results object return the highest-confidence insulator detection,
    or None if none found above threshold.
    """
    best: Optional[Detection] = None
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id != CLASS_ID_INSULATOR or conf < DETECTION_CONF:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det = Detection(bbox=(x1, y1, x2, y2), confidence=conf)
        if best is None or det.area > best.area:   # prefer largest (closest)
            best = det
    return best


# ─────────────────────────────────────────────────────────────────────────────
# OSD OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, state: ControlState, det: Optional[Detection]):
    """Draw bounding box, cross-hair, distance, and mode banner onto frame."""
    h, w = frame.shape[:2]

    # Cross-hair
    cv2.line(frame, (FRAME_CX - 20, FRAME_CY), (FRAME_CX + 20, FRAME_CY), (0, 255, 0), 1)
    cv2.line(frame, (FRAME_CX, FRAME_CY - 20), (FRAME_CX, FRAME_CY + 20), (0, 255, 0), 1)

    if det:
        x1, y1, x2, y2 = det.bbox
        colour = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(frame, (det.cx, det.cy), 5, colour, -1)

        dist = estimate_distance(det.width_px)
        label = f"Insulator  {det.confidence:.2f}  {dist:.1f} m"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Mode banner
    mode_colour = (0, 200, 255) if state.mode == "AUTO" else (0, 80, 255)
    cv2.rectangle(frame, (0, 0), (w, 28), mode_colour, -1)
    cv2.putText(frame, f"MODE: {state.mode}   dist={state.estimated_distance:.1f}m",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MODE SWITCHING
# ─────────────────────────────────────────────────────────────────────────────

def set_mode(vehicle, mode_name: str):
    """Change Pixhawk flight mode and block until confirmed (max 3 s)."""
    log.info("Requesting mode: %s", mode_name)
    vehicle.mode = VehicleMode(mode_name)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if vehicle.mode.name == mode_name:
            log.info("Mode confirmed: %s", mode_name)
            return True
        time.sleep(0.1)
    log.warning("Mode change to %s timed-out (current: %s)", mode_name, vehicle.mode.name)
    return False


def switch_to_visual(vehicle, state: ControlState):
    """Transition from AUTO mission to GUIDED (visual control)."""
    if state.mode == "VISUAL":
        return
    if set_mode(vehicle, "GUIDED"):
        state.mode = "VISUAL"
        log.info("Switched to VISUAL mode")


def switch_to_auto(vehicle, state: ControlState):
    """Return to the pre-planned AUTO mission."""
    if state.mode == "AUTO":
        return
    stop_drone(vehicle)
    if set_mode(vehicle, "AUTO"):
        state.mode = "AUTO"
        log.info("Switched to AUTO mission mode")


# ─────────────────────────────────────────────────────────────────────────────
# KEYBOARD HANDLER (runs in its own thread)
# ─────────────────────────────────────────────────────────────────────────────

class KeyHandler:
    """
    Non-blocking keyboard input.

    Keys:
      v  → switch to VISUAL mode (camera control)
      a  → switch to AUTO  mode (resume mission)
      q  → quit
    """
    def __init__(self):
        self.last_key: Optional[int] = None
        self._lock = threading.Lock()

    def update(self, key: int):
        with self._lock:
            self.last_key = key

    def consume(self) -> Optional[int]:
        with self._lock:
            k = self.last_key
            self.last_key = None
            return k


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading YOLOv8 model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    log.info("Connecting to Pixhawk on %s @ %d baud …", SERIAL_PORT, BAUD_RATE)
    vehicle = connect(SERIAL_PORT, baud=BAUD_RATE, wait_ready=True)
    log.info("Connected.  Mode=%s  Armed=%s", vehicle.mode.name, vehicle.armed)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        log.error("Cannot open camera index %d", CAMERA_INDEX)
        vehicle.close()
        return

    state    = ControlState()
    keys     = KeyHandler()
    tick     = 1.0 / CONTROL_HZ
    running  = True

    log.info("Starting main loop.  Press [v] VISUAL | [a] AUTO | [q] quit")

    try:
        while running:
            loop_start = time.time()

            # ── Grab frame ────────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed – skipping frame")
                time.sleep(0.05)
                continue

            # ── YOLO inference ────────────────────────────────────────────────
            results = model(frame, verbose=False)
            det     = best_detection(results)

            if det:
                state.last_detection    = det
                state.target_locked     = True
                state.estimated_distance = estimate_distance(det.width_px)
            else:
                state.target_locked      = False
                state.estimated_distance = 0.0

            # ── Visual control (GUIDED mode only) ─────────────────────────────
            if state.mode == "VISUAL":
                if det:
                    vx, vy, vz = compute_velocity_command(det)
                    send_ned_velocity(vehicle, vx, vy, vz)
                    log.debug("CMD  vx=%.2f  vy=%.2f  vz=%.2f  dist=%.2f m",
                              vx, vy, vz, state.estimated_distance)
                else:
                    # Target lost → hover
                    stop_drone(vehicle)
                    log.debug("Target lost – hovering")

            # ── OSD overlay ───────────────────────────────────────────────────
            frame = draw_overlay(frame, state, det)
            cv2.imshow("Insulator Inspection", frame)

            # ── Keyboard input ────────────────────────────────────────────────
            raw_key = cv2.waitKey(1) & 0xFF
            keys.update(raw_key)
            k = keys.consume()

            if k == ord("v"):
                switch_to_visual(vehicle, state)
            elif k == ord("a"):
                switch_to_auto(vehicle, state)
            elif k == ord("q"):
                log.info("Quit requested")
                running = False

            # ── Pace the loop ─────────────────────────────────────────────────
            elapsed = time.time() - loop_start
            sleep_t = max(0.0, tick - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – shutting down")
    finally:
        log.info("Stopping drone and closing connections …")
        stop_drone(vehicle)
        time.sleep(0.5)
        vehicle.close()
        cap.release()
        cv2.destroyAllWindows()
        log.info("Clean exit.")


if __name__ == "__main__":
    main()

