from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Iterable


class ActionType(str, Enum):
    DELTA = "delta"
    COMMAND = "command"
    SET_STEP = "set_step"
    QUIT = "quit"


@dataclass(frozen=True)
class Action:
    type: ActionType
    pan_delta: int = 0
    tilt_delta: int = 0
    command: str | None = None
    step: int | None = None


class InputProvider:
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def iter_actions(self) -> Iterable[Action]:
        raise NotImplementedError

    def intro_lines(self) -> list[str]:
        return []


class KeyboardInputProvider(InputProvider):
    def __init__(self, step: int) -> None:
        self.step = step
        self._readkey = None
        self._key = None

    def open(self) -> None:
        try:
            from readchar import key, readkey
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "Keyboard mode needs the 'readchar' package. Install with: pip install -r requirements.txt"
            ) from exc
        self._readkey = readkey
        self._key = key

    def iter_actions(self) -> Iterable[Action]:
        if self._readkey is None or self._key is None:
            raise RuntimeError("Keyboard provider not initialized. Call open() first.")

        while True:
            key_pressed = self._readkey()
            if key_pressed in ("q", "Q"):
                yield Action(ActionType.QUIT)
                return
            if key_pressed in ("h", "H"):
                yield Action(ActionType.COMMAND, command="HOME")
                continue
            if key_pressed in ("s", "S"):
                yield Action(ActionType.COMMAND, command="STATUS")
                continue
            if key_pressed in ("+", "="):
                self.step = min(self.step + 1, 20)
                yield Action(ActionType.SET_STEP, step=self.step)
                continue
            if key_pressed in ("-", "_"):
                self.step = max(self.step - 1, 1)
                yield Action(ActionType.SET_STEP, step=self.step)
                continue

            if key_pressed == self._key.UP:
                yield Action(ActionType.DELTA, tilt_delta=self.step)
            elif key_pressed == self._key.DOWN:
                yield Action(ActionType.DELTA, tilt_delta=-self.step)
            elif key_pressed == self._key.LEFT:
                yield Action(ActionType.DELTA, pan_delta=-self.step)
            elif key_pressed == self._key.RIGHT:
                yield Action(ActionType.DELTA, pan_delta=self.step)

    def intro_lines(self) -> list[str]:
        return [
            "Keyboard mode (press Q to quit)",
            "  UP/DOWN    : tilt + / -",
            "  LEFT/RIGHT : pan  - / +",
            "  H     : HOME",
            "  S     : STATUS",
            f"  + / - : adjust step (current: {self.step})",
        ]


class GestureSource:
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def iter_gestures(self) -> Iterable[str]:
        raise NotImplementedError


class StdinGestureSource(GestureSource):
    def iter_gestures(self) -> Iterable[str]:
        while True:
            try:
                line = input().strip()
            except EOFError:
                return
            if line:
                yield line


class GestureMappedInputProvider(InputProvider):
    def __init__(
        self,
        source: GestureSource,
        gesture_map: dict[str, object],
        default_step: int,
        *,
        quiet: bool = False,
    ) -> None:
        self.source = source
        self.gesture_map = gesture_map
        self.default_step = default_step
        self.quiet = quiet

    def open(self) -> None:
        self.source.open()

    def close(self) -> None:
        self.source.close()

    def iter_actions(self) -> Iterable[Action]:
        for gesture in self.source.iter_gestures():
            raw = self.gesture_map.get(gesture)
            if raw is None:
                if not self.quiet:
                    print(f"Unmapped gesture: {gesture}")
                continue
            action = action_from_mapping(raw, self.default_step)
            if action is None:
                if not self.quiet:
                    print(f"Ignoring invalid mapping for gesture: {gesture}")
                continue
            if action.type == ActionType.SET_STEP and action.step is not None:
                self.default_step = action.step
            yield action

    def intro_lines(self) -> list[str]:
        return [
            "Gesture mode (type gestures, one per line if using stdin)",
            f"Loaded {len(self.gesture_map)} gesture mappings.",
        ]


DEFAULT_GESTURE_MAP: dict[str, object] = {
    "swipe_left": {"type": "delta", "pan_delta": "-step", "tilt_delta": 0},
    "swipe_right": {"type": "delta", "pan_delta": "step", "tilt_delta": 0},
    "swipe_up": {"type": "delta", "pan_delta": 0, "tilt_delta": "step"},
    "swipe_down": {"type": "delta", "pan_delta": 0, "tilt_delta": "-step"},
    "double_tap": {"type": "command", "command": "HOME"},
    "hold": {"type": "command", "command": "STATUS"},
}


def load_gesture_map(path: str | None) -> dict[str, object]:
    if not path:
        return dict(DEFAULT_GESTURE_MAP)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Gesture map JSON must be an object.")
    return data


def load_gesture_source(spec: str, args_json: str | None) -> GestureSource:
    if spec.lower() == "stdin":
        return StdinGestureSource()
    cls = load_class(spec)
    kwargs = {}
    if args_json:
        kwargs = json.loads(args_json)
        if not isinstance(kwargs, dict):
            raise ValueError("Gesture args must decode to a JSON object.")
    source = cls(**kwargs)
    if not isinstance(source, GestureSource) and not hasattr(source, "iter_gestures"):
        raise TypeError("Gesture source must implement iter_gestures().")
    return source


def load_class(spec: str) -> type:
    if ":" not in spec:
        raise ValueError("Expected module:Class for gesture source.")
    module_name, class_name = spec.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def action_from_mapping(raw: object, default_step: int) -> Action | None:
    if isinstance(raw, str):
        return Action(ActionType.COMMAND, command=raw)
    if not isinstance(raw, dict):
        return None

    type_value = str(raw.get("type", "")).lower()
    if type_value == "":
        if "command" in raw or "cmd" in raw:
            return Action(ActionType.COMMAND, command=str(raw.get("command") or raw.get("cmd")))
        return None

    if type_value == ActionType.DELTA.value:
        pan_value = raw.get("pan_delta", raw.get("pan", 0))
        tilt_value = raw.get("tilt_delta", raw.get("tilt", 0))
        pan_delta = resolve_int(pan_value, default_step)
        tilt_delta = resolve_int(tilt_value, default_step)
        return Action(ActionType.DELTA, pan_delta=pan_delta, tilt_delta=tilt_delta)
    if type_value == ActionType.COMMAND.value:
        command = raw.get("command") or raw.get("cmd")
        if command is None:
            return None
        return Action(ActionType.COMMAND, command=str(command))
    if type_value == ActionType.SET_STEP.value:
        step_value = raw.get("step", default_step)
        return Action(ActionType.SET_STEP, step=resolve_int(step_value, default_step))
    if type_value == ActionType.QUIT.value:
        return Action(ActionType.QUIT)
    return None


def resolve_int(value: object, default_step: int) -> int:
    if isinstance(value, str) and value.lower() in ("step", "$step", "default_step"):
        return int(default_step)
    if isinstance(value, str) and value.lower() in ("-step", "-$step", "-default_step"):
        return -int(default_step)
    return int(value)
