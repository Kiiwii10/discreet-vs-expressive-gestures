import argparse
import re
import time

import serial

from input_providers import (
    ActionType,
    GestureMappedInputProvider,
    KeyboardInputProvider,
    load_gesture_map,
    load_gesture_source,
)

# Usage:
#   python control.py COM3
#   python control.py /dev/ttyUSB0
#   python control.py COM3 --keys
#   python control.py COM3 --input gestures --gesture-source stdin

BAUD = 115200
HOME_PAN = 80
HOME_TILT = 120


def open_port(port: str) -> serial.Serial:
    ser = serial.Serial(port, BAUD, timeout=0.5)
    time.sleep(2.0)  # allow board reset on connect
    # read a bit of startup text
    try:
        print(ser.read(1024).decode(errors="ignore"), end="")
    except Exception:
        pass
    return ser


def send_line(ser: serial.Serial, line: str) -> str:
    if not line.endswith("\n"):
        line += "\n"
    ser.write(line.encode("ascii"))
    ser.flush()
    # read one response line (best-effort)
    resp = ser.readline().decode(errors="ignore").strip()
    return resp


def clamp_angle(value: int) -> int:
    if value < 0:
        return 0
    if value > 180:
        return 180
    return value


def parse_set_response(resp: str) -> tuple[int | None, int | None]:
    match = re.search(r"\bP=(\d+)\s+T=(\d+)\b", resp)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def run_command_mode(ser: serial.Serial) -> None:
    print("Connected. Examples:")
    print("  P120 T70")
    print(f"  HOME (moves to {HOME_PAN}/{HOME_TILT})")
    print("  STATUS")
    print(f"  CENTER {HOME_PAN} {HOME_TILT}")
    print("Type commands, Ctrl+C to quit.\n")

    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue
            resp = send_line(ser, cmd)
            if resp:
                print(resp)
    except KeyboardInterrupt:
        print("\nExiting... sending HOME")
        try:
            print(send_line(ser, "HOME"))
        except Exception:
            pass


def run_input_mode(ser: serial.Serial, provider) -> None:
    current_pan = HOME_PAN
    current_tilt = HOME_TILT

    intro_lines = provider.intro_lines()
    for line in intro_lines:
        print(line)
    if intro_lines:
        print("")

    try:
        resp = send_line(ser, "HOME")
        if resp:
            pan, tilt = parse_set_response(resp)
            if pan is not None and tilt is not None:
                current_pan, current_tilt = pan, tilt
    except Exception:
        pass

    try:
        provider.open()
        for action in provider.iter_actions():
            if action.type == ActionType.QUIT:
                break
            if action.type == ActionType.SET_STEP:
                if action.step is not None:
                    print(f"Step: {action.step}")
                continue
            if action.type == ActionType.COMMAND:
                resp = send_line(ser, action.command or "")
                if resp:
                    print(resp)
                    pan, tilt = parse_set_response(resp)
                    if pan is not None and tilt is not None:
                        current_pan, current_tilt = pan, tilt
                if (action.command or "").strip().upper() == "HOME":
                    current_pan, current_tilt = HOME_PAN, HOME_TILT
                continue
            if action.type == ActionType.DELTA:
                current_pan = clamp_angle(current_pan + action.pan_delta)
                current_tilt = clamp_angle(current_tilt + action.tilt_delta)

                resp = send_line(ser, f"P{current_pan} T{current_tilt}")
                if resp:
                    print(resp)
                    pan, tilt = parse_set_response(resp)
                    if pan is not None and tilt is not None:
                        current_pan, current_tilt = pan, tilt
    except KeyboardInterrupt:
        print("\nExiting... sending HOME")
        try:
            print(send_line(ser, "HOME"))
        except Exception:
            pass
    except RuntimeError as exc:
        print(str(exc))
    finally:
        try:
            provider.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Control pan/tilt over serial.")
    parser.add_argument("port", help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--keys", action="store_true", help="Use keyboard control mode (legacy)")
    parser.add_argument(
        "--input",
        choices=["command", "keyboard", "gestures"],
        default=None,
        help="Input mode: command (default), keyboard, or gestures",
    )
    parser.add_argument("--step", type=int, default=4, help="Keyboard/gesture step size in degrees")
    parser.add_argument(
        "--gesture-source",
        default="stdin",
        help="Gesture source: stdin or module:Class (Python)",
    )
    parser.add_argument("--gesture-map", help="Path to JSON file mapping gestures to actions")
    parser.add_argument(
        "--gesture-args",
        help="JSON object with init kwargs for the gesture source",
    )
    args = parser.parse_args()

    ser = open_port(args.port)
    try:
        mode = args.input
        if mode is None:
            mode = "keyboard" if args.keys else "command"

        if mode == "command":
            run_command_mode(ser)
            return

        if mode == "keyboard":
            provider = KeyboardInputProvider(step=max(1, args.step))
            run_input_mode(ser, provider)
            return

        if mode == "gestures":
            try:
                gesture_map = load_gesture_map(args.gesture_map)
                source = load_gesture_source(args.gesture_source, args.gesture_args)
            except (ValueError, TypeError) as exc:
                print(f"Gesture setup error: {exc}")
                return
            provider = GestureMappedInputProvider(
                source,
                gesture_map,
                default_step=max(1, args.step),
            )
            run_input_mode(ser, provider)
            return
    finally:
        ser.close()


if __name__ == "__main__":
    main()
