"""

Loads the ONNX policy exported by export_policy.py
and drives the robot from live camera + sensor input.

"""

import argparse
import time

import numpy as np
import cv2
import onnxruntime as ort

from jetbot import Robot, Camera

IMG_SIZE = 64
MAX_OBSTACLE_RANGE_M = 0.60
MAX_CLIFF_RANGE_M = 0.15
FALLBACK_CLIFF_VALUE = 0.2 

CRITICAL_OBSTACLE_CM = 5.0
CONTROL_HZ = 8

ACTIONS = ["forward", "left", "right", "backward", "stop"]


def read_obstacle_cm(adc_channel) -> float:
    voltage = adc_channel.voltage
    if voltage <= 0.01:
        return MAX_OBSTACLE_RANGE_M * 100  # out of range
    distance_cm = 27.86 * (voltage ** -1.15)
    return float(np.clip(distance_cm, 4.0, MAX_OBSTACLE_RANGE_M * 100))


def read_cliff_normalized(cliff_sensor=None) -> float:
    if cliff_sensor is None:
        return FALLBACK_CLIFF_VALUE
    distance_m = cliff_sensor.read_distance_m()
    return float(np.clip(distance_m / MAX_CLIFF_RANGE_M, 0.0, 1.0))


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    chw = np.transpose(frame_resized, (2, 0, 1)).astype(np.uint8)
    return chw[None, ...]  # add batch dim


def apply_safety_override(action: int, obstacle_cm: float, cliff_norm: float) -> int:
    if cliff_norm >= 0.9:
        return ACTIONS.index("backward")
    if obstacle_cm <= CRITICAL_OBSTACLE_CM:
        return ACTIONS.index("stop")
    return action


def run(policy_path: str, use_adc: bool, use_cliff_sensor: bool):
    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    robot = Robot()
    camera = Camera.instance(width=224, height=224)

    adc_channel = None
    if use_adc:
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        adc_channel = AnalogIn(ads, ADS.P0)

    cliff_sensor = None
    if use_cliff_sensor:
        raise NotImplementedError(
            "Wire up your downward-facing cliff sensor and implement a small "
            "wrapper with a read_distance_m() method here, then pass it into "
            "read_cliff_normalized() below."
        )

    period = 1.0 / CONTROL_HZ
    print("Starting control loop. Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.time()

            frame = camera.value
            image_input = preprocess_frame(frame)

            obstacle_cm = read_obstacle_cm(adc_channel) if adc_channel else MAX_OBSTACLE_RANGE_M * 100
            obstacle_norm = np.array([[np.clip(obstacle_cm / (MAX_OBSTACLE_RANGE_M * 100), 0.0, 1.0)]], dtype=np.float32)
            cliff_norm_value = read_cliff_normalized(cliff_sensor)
            cliff_norm = np.array([[cliff_norm_value]], dtype=np.float32)

            logits = sess.run(
                None,
                {"image": image_input, "obstacle_dist": obstacle_norm, "cliff_dist": cliff_norm},
            )[0]
            action = int(np.argmax(logits))
            action = apply_safety_override(action, obstacle_cm, cliff_norm_value)

            drive(robot, action)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()
        camera.stop()
        print("Stopped.")


def drive(robot: "Robot", action: int, speed: float = 0.3):
    name = ACTIONS[action]
    if name == "forward":
        robot.forward(speed)
    elif name == "left":
        robot.left(speed)
    elif name == "right":
        robot.right(speed)
    elif name == "backward":
        robot.backward(speed)
    elif name == "stop":
        robot.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="policy.onnx")
    parser.add_argument("--use-adc", action="store_true", default=True,
                         help="Read the forward obstacle sensor through an ADS1115 ADC (default on).")
    parser.add_argument("--use-cliff-sensor", action="store_true", default=False,
                         help="Only pass this once you've wired up a second, downward-facing sensor "
                              "and implemented it in run() -- see the NotImplementedError above.")
    args = parser.parse_args()
    run(args.policy, args.use_adc, args.use_cliff_sensor)


if __name__ == "__main__":
    main()
