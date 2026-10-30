#!/usr/bin/env python3
"""EV3 square test with read-only encoder logging.

Keep this file beside your other EV3 programs. Run one square per video.
Set TURN_ANGLE to 90, 89, 88, -90 or -91 before each test.
CSV/JSON files are saved in the logs folder beside this script.
The log is encoder evidence, NOT a measurement of the robot's true floor pose.
The movement functions below retain the supplied motor commands and blocking.
Logging adds a small CPU/I/O load; it does not correct or synchronise motion.
"""
import csv
import json
import math
import os
import queue
import threading
import time
from datetime import datetime
from functools import wraps

from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent

WHEEL_RADIUS = 2.8       # cm; measured using a ruler, not motion-calibrated
WHEEL_DISTANCE = 12.0    # cm; wheel-centre separation
TURN_ANGLE = -91 #-90,+89,+90         # change to -91 for the preferred right-turn square
RUN_NOTE = ""           # e.g. video filename and surface; never invent results
LOG_PERIOD_S = 0.05      # desired interval; actual sample times are recorded

left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)


# Original motion calculations and motor commands, unchanged.
def drive_straight(distance_cm, speed=30):
    circumference = 2 * math.pi * WHEEL_RADIUS
    rotations = distance_cm / circumference
    degrees = rotations * 360
    left_motor.on_for_degrees(SpeedPercent(speed), degrees, block=False)
    right_motor.on_for_degrees(SpeedPercent(speed), degrees)


def turn(angle, speed=20):
    distance = math.pi * WHEEL_DISTANCE * abs(angle) / 360
    rotations = distance / (2 * math.pi * WHEEL_RADIUS)
    degrees = rotations * 360
    if angle > 0:
        left_motor.on_for_degrees(SpeedPercent(-speed), degrees, block=False)
        right_motor.on_for_degrees(SpeedPercent(speed), degrees)
    else:
        left_motor.on_for_degrees(SpeedPercent(speed), degrees, block=False)
        right_motor.on_for_degrees(SpeedPercent(-speed), degrees)


class EncoderLogger:
    """Background reads only; never writes a motor attribute or resets an encoder."""
    fields = [
        "row_type", "time_s", "read_duration_s", "sample_dt_s", "segment",
        "motion", "requested_cm", "requested_angle_deg", "speed_percent",
        "target_left_deg", "target_right_deg", "event",
        "left_position_counts", "right_position_counts",
        "left_delta_from_start_deg", "right_delta_from_start_deg",
        "left_speed_dps", "right_speed_dps", "left_state", "right_state",
        "odom_left_cm", "odom_right_cm", "odom_heading_deg", "detail"
    ]

    def __init__(self):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(folder, exist_ok=True)
        stem = "square_{:+g}_{}_{}".format(
            TURN_ANGLE, datetime.now().strftime("%Y%m%d_%H%M%S_%f"), os.getpid())
        self.path = os.path.join(folder, stem + ".csv")
        self.meta_path = os.path.join(folder, stem + ".json")
        self.cpr_l = left_motor.count_per_rot
        self.cpr_r = right_motor.count_per_rot
        self.initial_l = left_motor.position
        self.initial_r = right_motor.position
        max_l = left_motor.max_speed * 360.0 / self.cpr_l
        max_r = right_motor.max_speed * 360.0 / self.cpr_r
        metadata = {
            "run_note": RUN_NOTE, "turn_angle_deg": TURN_ANGLE,
            "wheel_radius_cm": WHEEL_RADIUS, "wheel_distance_cm": WHEEL_DISTANCE,
            "parameter_source": "ruler measurements; not motion calibration",
            "left_count_per_rot": self.cpr_l, "right_count_per_rot": self.cpr_r,
            "left_max_speed_dps": max_l, "right_max_speed_dps": max_r,
            "straight_speed_percent": 30, "turn_speed_percent": 20,
            "predicted_straight_cm_s": WHEEL_RADIUS * math.pi / 180 * .30 * (max_l + max_r) / 2,
            "predicted_turn_magnitude_deg_s": WHEEL_RADIUS / WHEEL_DISTANCE * .20 * (max_l + max_r),
            "requested_sample_period_s": LOG_PERIOD_S,
            "notes": [
                "Predicted speeds use rated maximum speed, not measured floor motion.",
                "Encoder heading assumes no slip, direct wheel drive, and positive counts forward.",
                "Event times mark Python calls; event rows have no encoder snapshot.",
                "Samples are sequential reads, not simultaneous measurements.",
                "segment_return means the original right-motor blocking call returned; it does not prove both wheels stopped."
            ],
            "manual_position_error_cm": None, "manual_heading_error_deg": None,
            "video_filename": None
        }
        with open(self.meta_path, "w") as stream:
            json.dump(metadata, stream, indent=2)
        self.stream = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fields)
        self.writer.writeheader()
        self.t0 = time.monotonic()
        self.context = {}
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.error = None
        self.last_sample_time = None
        self.segment = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.event("run_start")
        self.thread.start()

    def event(self, name, detail="", context=None):
        row = dict(self.context if context is None else context)
        row.update(row_type="event", time_s=time.monotonic() - self.t0,
                   event=name, detail=detail)
        self.events.put(row)

    def wrap(self, function, motion):
        @wraps(function)
        def tracked(value, speed=None):
            actual_speed = (30 if motion == "straight" else 20) if speed is None else speed
            self.segment += 1
            if motion == "straight":
                beta = value / (2 * math.pi * WHEEL_RADIUS) * 360
                # The supplied function's default positive speed is assumed.
                tl = tr = beta if actual_speed >= 0 else -beta
            else:
                beta = WHEEL_DISTANCE * abs(value) / (2 * WHEEL_RADIUS)
                tl, tr = (-beta, beta) if value > 0 else (beta, -beta)
                if actual_speed < 0:
                    tl, tr = -tl, -tr
            ctx = dict(segment=self.segment, motion=motion,
                       requested_cm=value if motion == "straight" else "",
                       requested_angle_deg=value if motion == "turn" else "",
                       speed_percent=actual_speed, target_left_deg=tl, target_right_deg=tr)
            self.context = ctx
            self.event("segment_start", context=ctx)
            try:
                return function(value, actual_speed)
            finally:
                self.event("segment_return", context=ctx)
        return tracked

    def _drain_events(self):
        while True:
            try:
                self.writer.writerow(self.events.get_nowait())
            except queue.Empty:
                return

    def _sample(self):
        ctx = dict(self.context)
        start = time.monotonic()
        pl, pr = left_motor.position, right_motor.position
        vl = left_motor.speed * 360.0 / self.cpr_l
        vr = right_motor.speed * 360.0 / self.cpr_r
        sl, sr = "|".join(left_motor.state), "|".join(right_motor.state)
        finish = time.monotonic()
        dl = (pl - self.initial_l) * 360.0 / self.cpr_l
        dr = (pr - self.initial_r) * 360.0 / self.cpr_r
        dist_l = math.radians(dl) * WHEEL_RADIUS
        dist_r = math.radians(dr) * WHEEL_RADIUS
        ctx.update(row_type="sample", time_s=start - self.t0,
                   read_duration_s=finish - start,
                   sample_dt_s="" if self.last_sample_time is None else start - self.last_sample_time,
                   left_position_counts=pl, right_position_counts=pr,
                   left_delta_from_start_deg=dl, right_delta_from_start_deg=dr,
                   left_speed_dps=vl, right_speed_dps=vr,
                   left_state=sl, right_state=sr, odom_left_cm=dist_l,
                   odom_right_cm=dist_r,
                   odom_heading_deg=math.degrees((dist_r - dist_l) / WHEEL_DISTANCE))
        self.last_sample_time = start
        self.writer.writerow(ctx)

    def _worker(self):
        try:
            last_flush = time.monotonic()
            while not self.stop_event.is_set():
                tick = time.monotonic()
                self._drain_events()
                self._sample()
                if tick - last_flush >= 1:
                    self.stream.flush()
                    last_flush = tick
                self.stop_event.wait(max(0, LOG_PERIOD_S - (time.monotonic() - tick)))
            self._drain_events()
            self._sample()
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.stream.flush()
            self.stream.close()

    def close(self):
        self.event("run_end")
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Logger did not finish; CSV may be incomplete")
        if self.error:
            raise RuntimeError("Encoder logging failed: " + self.error)
        print("CSV: " + self.path)
        print("Metadata: " + self.meta_path)


def main():
    logger = EncoderLogger()
    logged_straight = logger.wrap(drive_straight, "straight")
    logged_turn = logger.wrap(turn, "turn")
    print("Starting in 2 seconds...")
    try:
        time.sleep(2)  # same actual delay as the original code
        # Same four straight segments and turns; only the chosen angle varies.
        logged_straight(50)
        logged_turn(TURN_ANGLE)
        logged_straight(50)
        logged_turn(TURN_ANGLE)
        logged_straight(50)
        logged_turn(TURN_ANGLE)
        logged_straight(50)
        logged_turn(TURN_ANGLE)
        logger.event("sequence_return")
    except BaseException as exc:
        logger.event("run_exception", repr(exc))
        raise
    finally:
        try:
            left_motor.off()
        finally:
            try:
                right_motor.off()
            finally:
                logger.event("motors_off_called")
                logger.close()
    print("Square completed!")


if __name__ == "__main__":
    main()
