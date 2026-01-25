from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from serial.tools import list_ports

from control import clamp_angle, open_port, parse_set_response, send_line


PRESETS: dict[str, dict[str, Any]] = {
    "discreet": {
        "label": "Discreet",
        "mode": "gestures",
        "step": 3,
        "gesture_map": {
            "swipe_left": {"type": "delta", "pan_delta": "-step", "tilt_delta": 0},
            "swipe_right": {"type": "delta", "pan_delta": "step", "tilt_delta": 0},
            "swipe_up": {"type": "delta", "pan_delta": 0, "tilt_delta": "step"},
            "swipe_down": {"type": "delta", "pan_delta": 0, "tilt_delta": "-step"},
            "double_tap": {"type": "command", "command": "HOME"},
            "hold": {"type": "command", "command": "STATUS"},
        },
    },
    "expressive": {
        "label": "Expressive",
        "mode": "gestures",
        "step": 3,
        "gesture_map": {
            "swipe_left": {"type": "delta", "pan_delta": "-step", "tilt_delta": 0},
            "swipe_right": {"type": "delta", "pan_delta": "step", "tilt_delta": 0},
            "swipe_up": {"type": "delta", "pan_delta": 0, "tilt_delta": "step"},
            "swipe_down": {"type": "delta", "pan_delta": 0, "tilt_delta": "-step"},
            "circle": {"type": "command", "command": "CENTER"},
            "double_tap": {"type": "command", "command": "HOME"},
            "hold": {"type": "command", "command": "STATUS"},
        },
    },
}

TEST_ALLOWED = ["discreet", "expressive"]
TEST_DEFAULT = "discreet"
FREE_DEFAULT = "discreet"
DEFAULT_PORT = "COM4"
DEFAULT_TOLERANCE = 5
TEST_TOLERANCE = 5
CLICK_LED_SECONDS = 0.25
HOME_PAN = 80
HOME_TILT = 120
DEFAULT_DEVICE_MODE = "real"  # "real" or "simulated"
TEST_RUNS_DIR = Path("test-runs")
QUESTIONS_FILENAME = "questions.json"


@dataclass
class TestSession:
    active: bool = False
    target_index: int = 0
    started_at: float | None = None
    started_at_wall: float | None = None
    target_started_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    gestures_current: int = 0
    gestures_total: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    saved: bool = False
    last_result_path: str | None = None
    user_id: int | None = None
    user_id_text: str | None = None


@dataclass
class SessionState:
    active: bool = False
    stage: str | None = None
    started_at: float | None = None
    started_at_wall: float | None = None
    ended_at_wall: float | None = None
    session_id: str | None = None
    file_path: str | None = None
    user_id_text: str | None = None
    user_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pre_study: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    end_reason: str | None = None
    pending_block_index: int | None = None


@dataclass
class RobotState:
    port: str = DEFAULT_PORT
    device_mode: str = DEFAULT_DEVICE_MODE
    pan: int = HOME_PAN
    tilt: int = HOME_TILT
    connected: bool = False
    last_connect_error: str | None = None
    preset: str = FREE_DEFAULT
    input_mode: str = PRESETS[FREE_DEFAULT]["mode"]
    step: int = PRESETS[FREE_DEFAULT]["step"]
    gesture_map: dict[str, Any] = field(default_factory=lambda: dict(PRESETS[FREE_DEFAULT]["gesture_map"]))
    test_plan: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"pan": 180, "tilt": 0},
            {"pan": 0, "tilt": 0},
            {"pan": 60, "tilt": 140},
            {"pan": 20, "tilt": 60},
            {"pan": 150, "tilt": 150},
        ]
    )
    tolerance: int = DEFAULT_TOLERANCE
    test: TestSession = field(default_factory=TestSession)
    session: SessionState = field(default_factory=SessionState)


class ConnectRequest(BaseModel):
    port: str | None = None


class DeviceModeRequest(BaseModel):
    mode: str


class CommandRequest(BaseModel):
    command: str


class SetRequest(BaseModel):
    pan: int
    tilt: int


class TestPlanRequest(BaseModel):
    targets: list[dict[str, int]]
    tolerance: int = DEFAULT_TOLERANCE


class GestureRequest(BaseModel):
    name: str
    kind: str | None = None


class PresetRequest(BaseModel):
    name: str
    context: str | None = None
    user_id: str | None = None


class StepRequest(BaseModel):
    step: int


class TestStartRequest(BaseModel):
    user_id: str | None = None


class TestStopRequest(BaseModel):
    reason: str | None = None


class SessionStartRequest(BaseModel):
    user_id: str
    environment: str
    condition_order: str | None = None


class SessionStatusRequest(BaseModel):
    user_id: str


class SessionCleanupRequest(BaseModel):
    user_id: str


class PreStudyRequest(BaseModel):
    age: int | None = None
    gender: str | None = None
    handedness: str | None = None
    experience: list[str] | None = None
    baseline: dict[str, int] | None = None


class BlockQuestionnaireRequest(BaseModel):
    block_index: int | None = None
    skipped: bool = False
    answers: dict[str, Any] | None = None


class FinalQuestionnaireRequest(BaseModel):
    skipped: bool = False
    answers: dict[str, Any] | None = None


class SessionEndRequest(BaseModel):
    reason: str | None = None
    stage: str | None = None


state = RobotState()
state_lock = asyncio.Lock()
connect_lock = asyncio.Lock()
serial_handle = None
ws_clients: set[WebSocket] = set()
motion_task: asyncio.Task | None = None
auto_connect_task: asyncio.Task | None = None


class SimulatedSerial:
    def __init__(self) -> None:
        self._closed = False
        self.last_write: bytes = b""
        self.write_count: int = 0

    def write(self, data: bytes) -> int:
        if self._closed:
            raise RuntimeError("Simulated serial is closed.")
        self.last_write = bytes(data)
        self.write_count += 1
        return len(data)

    def close(self) -> None:
        self._closed = True

@dataclass
class MotionState:
    pan_dir: int = 0
    tilt_dir: int = 0
    pan_f: float = float(HOME_PAN)
    tilt_f: float = float(HOME_TILT)
    last_sent_pan: int = HOME_PAN
    last_sent_tilt: int = HOME_TILT
    last_tick: float = 0.0
    last_broadcast: float = 0.0
    last_test_check: float = 0.0


motion = MotionState()


@dataclass
class LedState:
    left_click_until: float = 0.0
    right_click_until: float = 0.0
    last_sent: tuple[int, int, int, int, int, int] | None = None


leds = LedState()

app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    global auto_connect_task
    if auto_connect_task is None or auto_connect_task.done():
        auto_connect_task = asyncio.create_task(auto_connect_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global serial_handle
    if serial_handle is not None:
        try:
            serial_handle.close()
        except Exception:
            pass
        serial_handle = None
        state.connected = False


def preset_payload(name: str) -> dict[str, Any]:
    preset = PRESETS[name]
    return {
        "preset": name,
        "label": preset["label"],
        "mode": preset["mode"],
        "step": preset["step"],
        "gesture_map": preset["gesture_map"],
    }


def apply_preset(name: str) -> None:
    preset = PRESETS[name]
    state.preset = name
    state.input_mode = preset["mode"]
    state.step = preset["step"]
    state.gesture_map = dict(preset["gesture_map"])


def normalize_user_id_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text if text else None


def parse_numeric_user_id(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def allowed_presets_for_user_text(text: str | None) -> list[str]:
    cleaned = normalize_user_id_text(text)
    if cleaned is None:
        return []
    return ["discreet", "expressive"]


def is_valid_user_id(user_id: int | None) -> bool:
    return user_id is not None and user_id > 0


def should_save_results() -> bool:
    # Save for all test runs, as long as we have a user_id_text.
    return normalize_user_id_text(state.test.user_id_text) is not None


def session_expected_preset(stage: str | None) -> str | None:
    if stage in ("discreet_intro", "discreet_test", "discreet_questions"):
        return "discreet"
    if stage in ("expressive_intro", "expressive_test", "expressive_questions"):
        return "expressive"
    return None


def session_order_first_second(order_value: str | None) -> tuple[str, str]:
    """
    Map condition order to (first, second) presets.
    "order_discreet" (or contains "discreet"): discreet first, expressive second.
    "order_expressive" (or contains "expressive"): expressive first, discreet second.
    Any other value falls back to discreet first.
    """
    if order_value:
        text = order_value.strip().lower()
        if "expressive" in text:
            return "expressive", "discreet"
    return "discreet", "expressive"


def build_session_summary() -> dict[str, Any]:
    session = state.session
    if not session.active and not session.file_path:
        return {"active": False}
    return {
        "active": session.active,
        "stage": session.stage,
        "user_id": session.user_id_text,
        "environment": session.metadata.get("environment"),
        "condition_order": session.metadata.get("condition_order"),
        "block_count": len(session.blocks),
        "pending_block_index": session.pending_block_index,
        "expected_preset": session_expected_preset(session.stage),
    }


def build_state_payload() -> dict[str, Any]:
    return {
        "type": "state",
        "connected": state.connected,
        "device_mode": state.device_mode,
        "port": state.port,
        "last_connect_error": state.last_connect_error,
        "pan": state.pan,
        "tilt": state.tilt,
        "preset": state.preset,
        "input_mode": state.input_mode,
        "step": state.step,
        "gesture_map": state.gesture_map,
    }


def build_test_payload() -> dict[str, Any]:
    elapsed = 0.0
    if state.test.active and state.test.target_started_at is not None:
        elapsed = max(0.0, time.monotonic() - state.test.target_started_at)
    return {
        "type": "test",
        "active": state.test.active,
        "user_id": state.test.user_id_text,
        "target_index": state.test.target_index,
        "targets": state.test_plan,
        "tolerance": state.tolerance,
        "gestures_current": state.test.gestures_current,
        "gestures_total": state.test.gestures_total,
        "results": state.test.results,
        "current_elapsed": elapsed,
        "session": build_session_summary(),
    }


def step_to_speed_dps(step: int) -> float:
    # Mirror the motion loop mapping so reports match behavior.
    speed = 45.0 * (max(1, min(20, int(step))) / 5.0)
    return float(max(10.0, min(180.0, speed)))


def test_event_time(now: float | None = None) -> float | None:
    if not state.test.active or state.test.started_at is None:
        return None
    t = (time.monotonic() if now is None else now) - state.test.started_at
    return round(max(0.0, t), 3)


def ensure_results_dir() -> Path:
    path = TEST_RUNS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def environment_key(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    if "private" in text:
        return "private"
    if "public" in text:
        return "public"
    return None


def scenario_key(preset: str, env_key: str) -> str:
    return f"{preset}_{env_key}"


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_existing_session(user_id_text: str | None) -> tuple[dict[str, Any], Path] | None:
    """Return most recent saved data + folder for this user_id_text, if any."""
    if not user_id_text:
        return None
    folder = ensure_results_dir()
    prefix = f"{safe_user_id_for_filename(user_id_text)}_"
    candidates = [p for p in folder.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None

    def score(path: Path) -> float:
        qpath = path / QUESTIONS_FILENAME
        if qpath.exists():
            return qpath.stat().st_mtime
        return path.stat().st_mtime

    candidates.sort(key=score, reverse=True)
    chosen = candidates[0]
    questions_path = chosen / QUESTIONS_FILENAME
    return load_json_file(questions_path), chosen


def safe_user_id_for_filename(user_id_text: str | None) -> str:
    if not user_id_text:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", user_id_text.strip())
    cleaned = cleaned.strip("_") or "unknown"
    return cleaned


def iso_time(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def ensure_user_dir(user_id_text: str | None) -> Path:
    if user_id_text and state.session.file_path and normalize_user_id_text(state.session.user_id_text) == normalize_user_id_text(user_id_text):
        return Path(state.session.file_path).parent
    existing = find_existing_session(user_id_text)
    if existing:
        _, folder = existing
        if user_id_text and normalize_user_id_text(state.session.user_id_text) == normalize_user_id_text(user_id_text):
            state.session.file_path = str(folder / QUESTIONS_FILENAME)
            state.session.session_id = folder.name
        return folder
    folder = ensure_results_dir()
    uid = safe_user_id_for_filename(user_id_text)
    suffix = uuid.uuid4().hex
    path = folder / f"{uid}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    if user_id_text and normalize_user_id_text(state.session.user_id_text) == normalize_user_id_text(user_id_text):
        state.session.file_path = str(path / QUESTIONS_FILENAME)
        state.session.session_id = path.name
    return path


def ensure_session_file() -> Path:
    if state.session.file_path:
        return Path(state.session.file_path)
    folder = ensure_user_dir(state.session.user_id_text)
    path = folder / QUESTIONS_FILENAME
    state.session.file_path = str(path)
    state.session.session_id = folder.name
    return path


def block_questionnaire_completed(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    if entry.get("skipped"):
        return True
    answers = entry.get("answers") or entry.get("questionnaire") or {}
    return bool(answers)


def final_questionnaire_completed(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    if entry.get("skipped"):
        return True
    answers = entry.get("answers") or entry.get("questionnaire") or entry
    return bool(answers)


def build_blocks_for_environment(
    user_dir: Path,
    questions: dict[str, Any],
    env_key: str,
    environment: str,
    condition_order: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    blocks: list[dict[str, Any]] = []
    scenario_to_index: dict[str, int] = {}
    blocks_data = questions.get("blocks", {})
    for preset in ("discreet", "expressive"):
        result_path = user_dir / f"{preset}_{env_key}.json"
        if not result_path.exists():
            continue
        block = load_json_file(result_path)
        block["result_file"] = str(result_path)
        block["preset"] = block.get("preset") or preset
        block["preset_label"] = block.get("preset_label") or PRESETS.get(preset, {}).get("label")
        block["block_label"] = f"{block.get('preset_label', preset)} - {environment}"
        block["session"] = block.get("session") or {
            "stage": None,
            "environment": environment,
            "condition_order": condition_order,
        }
        scenario = scenario_key(preset, env_key)
        entry = blocks_data.get(scenario)
        if entry:
            block["questionnaire"] = entry.get("answers") or entry.get("questionnaire") or {}
            block["questionnaire_skipped"] = bool(entry.get("skipped"))
            block["questionnaire_abandoned"] = False
            block["questionnaire_completed_at"] = entry.get("completed_at")
        block["block_index"] = len(blocks)
        blocks.append(block)
        scenario_to_index[scenario] = block["block_index"]
    return blocks, scenario_to_index


def determine_resume_stage(user_dir: Path, questions: dict[str, Any], env_key: str, condition_order: str | None) -> tuple[str, str | None]:
    prestudy_done = bool(questions.get("prestudy")) or bool(questions.get("baseline"))
    if not prestudy_done:
        return "prestudy", None
    first, second = session_order_first_second(condition_order)
    for preset in (first, second):
        scenario = scenario_key(preset, env_key)
        result_path = user_dir / f"{preset}_{env_key}.json"
        if not result_path.exists():
            return f"{preset}_intro", None
        if not block_questionnaire_completed(questions.get("blocks", {}).get(scenario)):
            return f"{preset}_questions", scenario
    if not final_questionnaire_completed(questions.get("final", {}).get(env_key)):
        return "final", None
    return "done", None


def environment_completed(user_dir: Path, questions: dict[str, Any], env_key: str) -> bool:
    blocks = questions.get("blocks", {})
    for preset in ("discreet", "expressive"):
        result_path = user_dir / f"{preset}_{env_key}.json"
        if not result_path.exists():
            return False
        scenario = scenario_key(preset, env_key)
        if not block_questionnaire_completed(blocks.get(scenario)):
            return False
    if not final_questionnaire_completed(questions.get("final", {}).get(env_key)):
        return False
    return True


def cleanup_user_dirs(user_id_text: str) -> int:
    folder = ensure_results_dir()
    prefix = f"{safe_user_id_for_filename(user_id_text)}_"
    removed = 0
    for path in folder.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def reset_test_state() -> None:
    state.test = TestSession()
    motion.pan_dir = 0
    motion.tilt_dir = 0
    motion.last_tick = 0.0
    motion.last_broadcast = 0.0
    motion.last_test_check = 0.0


def reset_session_state() -> None:
    state.session = SessionState()


def build_session_payload() -> dict[str, Any]:
    session = state.session
    return {
        "session_id": session.session_id,
        "active": session.active,
        "stage": session.stage,
        "started_at": iso_time(session.started_at_wall),
        "ended_at": iso_time(session.ended_at_wall),
        "end_reason": session.end_reason,
        "user_id_text": session.user_id_text,
        "user_id": session.user_id,
        "metadata": session.metadata,
        "pre_study": session.pre_study,
        "baseline": session.baseline,
        "blocks": session.blocks,
        "final": session.final,
    }


def update_questions_from_session(questions: dict[str, Any], session: SessionState) -> dict[str, Any]:
    if session.user_id_text:
        questions["user_id_text"] = session.user_id_text
    if session.user_id is not None:
        questions["user_id"] = session.user_id

    env_value = session.metadata.get("environment")
    env_key = environment_key(env_value)
    if env_key:
        consent = questions.setdefault("consent", {})
        consent_entry = consent.setdefault(env_key, {})
        consent_entry["environment"] = env_value
        if session.metadata.get("condition_order"):
            consent_entry["condition_order"] = session.metadata.get("condition_order")
        consent_entry["saved_at"] = iso_time(time.time())

    if session.pre_study:
        questions["prestudy"] = session.pre_study
    if session.baseline:
        questions["baseline"] = session.baseline

    if session.blocks:
        blocks = questions.setdefault("blocks", {})
        for block in session.blocks:
            preset = block.get("preset")
            block_env = block.get("session", {}).get("environment") or env_value
            block_env_key = environment_key(block_env)
            if not preset or not block_env_key:
                continue
            key = scenario_key(preset, block_env_key)
            entry = blocks.setdefault(key, {})
            if block.get("questionnaire") or block.get("questionnaire_skipped"):
                entry["answers"] = block.get("questionnaire") or {}
                entry["skipped"] = bool(block.get("questionnaire_skipped"))
                entry["completed_at"] = block.get("questionnaire_completed_at")
            entry["preset"] = preset
            entry["environment"] = block_env

    if session.final:
        final_by_env = questions.setdefault("final", {})
        if env_key:
            entry = {"answers": session.final}
            if "skipped" in session.final:
                entry["skipped"] = bool(session.final.get("skipped"))
            if session.ended_at_wall:
                entry["completed_at"] = iso_time(session.ended_at_wall)
            final_by_env[env_key] = entry
    return questions


def save_session_snapshot() -> dict[str, Any]:
    payload = build_session_payload()
    if not state.session.file_path and not state.session.active:
        return payload
    path = ensure_session_file()
    questions = load_json_file(path)
    questions = update_questions_from_session(questions, state.session)
    save_json_file(path, questions)
    return payload


def build_test_results_payload(reason: str) -> dict[str, Any]:
    now = time.time()
    elapsed_total = None
    if state.test.started_at is not None:
        elapsed_total = max(0.0, time.monotonic() - state.test.started_at)
    current_elapsed = None
    if state.test.target_started_at is not None:
        current_elapsed = max(0.0, time.monotonic() - state.test.target_started_at)
    payload: dict[str, Any] = {
        "user_id": state.test.user_id,
        "user_id_text": state.test.user_id_text,
        "preset": state.preset,
        "preset_label": PRESETS.get(state.preset, {}).get("label"),
        "input_mode": state.input_mode,
        "step": state.step,
        "speed_dps": round(step_to_speed_dps(state.step), 3),
        "port": state.port,
        "started_at": iso_time(state.test.started_at_wall or now),
        "ended_at": iso_time(now),
        "reason": reason,
        "targets": state.test_plan,
        "tolerance": state.tolerance,
        "results": state.test.results,
        "current_target_index": state.test.target_index,
        "current_target_elapsed": None if current_elapsed is None else round(current_elapsed, 3),
        "gestures_total": state.test.gestures_total,
        "events": state.test.events,
        "elapsed_total": None if elapsed_total is None else round(elapsed_total, 3),
    }
    if state.session.active:
        payload["session"] = {
            "session_id": state.session.session_id,
            "stage": state.session.stage,
            "environment": state.session.metadata.get("environment"),
            "condition_order": state.session.metadata.get("condition_order"),
        }
    return payload


def save_test_results(reason: str) -> tuple[dict[str, Any], Path | None]:
    payload = build_test_results_payload(reason)
    if not should_save_results():
        state.test.saved = True
        return payload, None
    env_key = environment_key(state.session.metadata.get("environment")) or "unknown"
    folder = ensure_user_dir(state.test.user_id_text)
    path = folder / f"{state.preset}_{env_key}.json"
    save_json_file(path, payload)
    state.test.saved = True
    state.test.last_result_path = str(path)
    return payload, path


def finalize_test_block(reason: str) -> dict[str, Any]:
    payload, path = save_test_results(reason)
    if state.session.active:
        block = dict(payload)
        env_value = state.session.metadata.get("environment")
        env_key = environment_key(env_value)
        block["result_file"] = str(path) if path else None
        block["environment"] = env_value
        block["questionnaire"] = {}
        block["questionnaire_skipped"] = False
        block["questionnaire_abandoned"] = False
        block["questionnaire_completed_at"] = None
        block["block_label"] = f"{block.get('preset_label', block.get('preset'))} - {env_value}"
        block_index = None
        for idx, existing in enumerate(state.session.blocks):
            existing_env = existing.get("session", {}).get("environment") or existing.get("environment")
            if existing.get("preset") == block.get("preset") and environment_key(existing_env) == env_key:
                block_index = idx
                break
        if block_index is None:
            block_index = len(state.session.blocks)
            block["block_index"] = block_index
            state.session.blocks.append(block)
        else:
            block["block_index"] = block_index
            state.session.blocks[block_index] = block
        state.session.pending_block_index = block_index
        preset = block.get("preset")
        if preset == "expressive":
            state.session.stage = "expressive_questions"
        else:
            state.session.stage = "discreet_questions"
        save_session_snapshot()
    return payload


def update_block_questionnaire(block_index: int | None, answers: dict[str, Any] | None, skipped: bool) -> dict[str, Any]:
    if not state.session.blocks:
        env_value = state.session.metadata.get("environment")
        env_key = environment_key(env_value)
        if env_key and state.session.user_id_text:
            user_dir = ensure_user_dir(state.session.user_id_text)
            questions = load_json_file(user_dir / QUESTIONS_FILENAME)
            blocks, scenario_to_index = build_blocks_for_environment(
                user_dir,
                questions,
                env_key,
                env_value,
                state.session.metadata.get("condition_order"),
            )
            state.session.blocks = blocks
            preset = session_expected_preset(state.session.stage) or state.preset
            scenario = scenario_key(preset, env_key)
            state.session.pending_block_index = scenario_to_index.get(scenario)
    if not state.session.blocks:
        raise HTTPException(status_code=400, detail="No completed block available.")
    index = block_index
    if index is None:
        index = state.session.pending_block_index
    if index is None:
        index = len(state.session.blocks) - 1
    if index < 0 or index >= len(state.session.blocks):
        raise HTTPException(status_code=400, detail="Invalid block index.")
    block = state.session.blocks[index]
    block["questionnaire"] = answers or {}
    block["questionnaire_skipped"] = bool(skipped)
    block["questionnaire_abandoned"] = False
    block["questionnaire_completed_at"] = iso_time(time.time())
    state.session.pending_block_index = None
    env_value = state.session.metadata.get("environment")
    env_key = environment_key(env_value) or "unknown"
    questions_path = ensure_session_file()
    questions = load_json_file(questions_path)
    questions = update_questions_from_session(questions, state.session)
    save_json_file(questions_path, questions)
    user_dir = questions_path.parent
    stage, pending_scenario = determine_resume_stage(user_dir, questions, env_key, state.session.metadata.get("condition_order"))
    blocks, scenario_to_index = build_blocks_for_environment(
        user_dir,
        questions,
        env_key,
        env_value,
        state.session.metadata.get("condition_order"),
    )
    state.session.blocks = blocks
    state.session.pending_block_index = scenario_to_index.get(pending_scenario) if pending_scenario else None
    state.session.stage = stage
    if block.get("result_file"):
        try:
            path = Path(block["result_file"])
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
            else:
                saved = {}
            saved["block_questionnaire"] = block["questionnaire"]
            saved["block_questionnaire_skipped"] = block["questionnaire_skipped"]
            saved["block_questionnaire_completed_at"] = block["questionnaire_completed_at"]
            save_json_file(path, saved)
        except Exception:
            pass
    return block


async def broadcast(payload: dict[str, Any]) -> None:
    if not ws_clients:
        return
    dead: list[WebSocket] = []
    for ws in ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.discard(ws)


def require_connection() -> Any:
    if serial_handle is None or not state.connected:
        raise HTTPException(status_code=400, detail="Not connected to robot.")
    return serial_handle


def write_line_no_wait(ser: Any, line: str) -> None:
    if not line.endswith("\n"):
        line += "\n"
    ser.write(line.encode("ascii", errors="ignore"))


def motion_sync_to_state() -> None:
    motion.pan_f = float(state.pan)
    motion.tilt_f = float(state.tilt)
    motion.last_sent_pan = int(state.pan)
    motion.last_sent_tilt = int(state.tilt)
    motion.last_tick = 0.0
    motion.last_broadcast = 0.0


def set_home_position(ser: Any) -> None:
    write_line_no_wait(ser, f"P{HOME_PAN} T{HOME_TILT}")
    state.pan = HOME_PAN
    state.tilt = HOME_TILT
    motion_sync_to_state()


def guidance_directions() -> tuple[bool, bool, bool, bool]:
    """
    Returns booleans (left, right, up, down) indicating which axes still need
    movement toward the current test target. Only active during an active test.
    """
    if not state.test.active:
        return False, False, False, False
    if state.test.target_index >= len(state.test_plan):
        return False, False, False, False
    target = state.test_plan[state.test.target_index]
    pan_target = target.get("pan", 0)
    tilt_target = target.get("tilt", 0)
    tol = state.tolerance
    # Pan direction is inverted relative to servo orientation (see motion loop),
    # so compute guidance using target - current to match the left/right LEDs.
    pan_delta = pan_target - state.pan
    tilt_delta = state.tilt - tilt_target
    need_right = pan_delta > tol
    need_left = pan_delta < -tol
    need_up = tilt_delta < -tol  # target tilt higher => move up
    need_down = tilt_delta > tol
    return need_left, need_right, need_up, need_down


def compute_led_tuple(now: float) -> tuple[int, int, int, int, int, int]:
    left_click = now < leds.left_click_until
    if state.preset == "discreet" and (motion.pan_dir != 0 or motion.tilt_dir != 0):
        left_click = True
    right_click = now < leds.right_click_until
    guide_left, guide_right, guide_up, guide_down = guidance_directions()
    moving = (motion.pan_dir != 0) or (motion.tilt_dir != 0)

    within_pan = False
    within_tilt = False
    if state.test.active and state.test.target_index < len(state.test_plan):
        target = state.test_plan[state.test.target_index]
        tol = state.tolerance
        within_pan = abs(state.pan - target.get("pan", 0)) <= tol
        within_tilt = abs(state.tilt - target.get("tilt", 0)) <= tol

    def dir_value(active: bool, guide: bool) -> int:
        if active:
            return 3 if guide else 1  # yellow if moving correctly, else white
        if not moving and guide:
            return 2  # green guidance when idle
        return 0

    up = dir_value(motion.tilt_dir > 0, guide_up)
    down = dir_value(motion.tilt_dir < 0, guide_down)
    left = dir_value(motion.pan_dir > 0, guide_left)
    right = dir_value(motion.pan_dir < 0, guide_right)
    if state.test.active:
        # When a movement axis is already within the target tolerance, show "reached"
        # feedback while moving by lighting both LEDs on that axis blue.
        if within_pan and motion.pan_dir != 0:
            left = 4
            right = 4
        if within_tilt and motion.tilt_dir != 0:
            up = 4
            down = 4
    return (
        1 if left_click else 0,  # top-left: left click (white)
        up,                      # top-middle: up (white or green)
        1 if right_click else 0, # top-right: right click (white)
        left,                    # bottom-left: left (white or green)
        down,                    # bottom-middle: down (white or green)
        right,                   # bottom-right: right (white or green)
    )


async def ensure_motion_task() -> None:
    global motion_task
    if motion_task is not None and not motion_task.done():
        return
    motion_task = asyncio.create_task(motion_loop())


async def motion_loop() -> None:
    # Fixed cadence, no blocking reads. Browser sends start/stop; backend streams setpoints steadily.
    global serial_handle
    while True:
        await asyncio.sleep(0.02)  # 50 Hz
        state_payload: dict[str, Any] | None = None
        test_payload: dict[str, Any] | None = None
        async with state_lock:
            if serial_handle is None or not state.connected:
                break
            now = time.monotonic()
            if motion.last_tick == 0.0:
                motion.last_tick = now
                continue
            dt = min(0.1, max(0.0, now - motion.last_tick))
            motion.last_tick = now

            if now - motion.last_test_check >= 0.1:
                motion.last_test_check = now
                advanced = maybe_advance_test()
                if advanced:
                    test_payload = build_test_payload()

            if motion.pan_dir == 0 and motion.tilt_dir == 0:
                # No setpoints to stream; test advancement (if any) will be broadcast below.
                pass
            else:
                speed = step_to_speed_dps(state.step)

                # Pan direction is inverted relative to servo orientation, so subtract.
                motion.pan_f = max(0.0, min(180.0, motion.pan_f - motion.pan_dir * speed * dt))
                motion.tilt_f = max(0.0, min(180.0, motion.tilt_f + motion.tilt_dir * speed * dt))

                pan_int = int(round(motion.pan_f))
                tilt_int = int(round(motion.tilt_f))
                if pan_int != motion.last_sent_pan or tilt_int != motion.last_sent_tilt:
                    motion.last_sent_pan = pan_int
                    motion.last_sent_tilt = tilt_int
                    state.pan = pan_int
                    state.tilt = tilt_int
                    write_line_no_wait(serial_handle, f"P{pan_int} T{tilt_int}")
                    advanced = maybe_advance_test()
                    if advanced:
                        test_payload = build_test_payload()

            led_tuple = compute_led_tuple(now)
            if led_tuple != leds.last_sent:
                write_line_no_wait(serial_handle, f"LED {led_tuple[0]} {led_tuple[1]} {led_tuple[2]} {led_tuple[3]} {led_tuple[4]} {led_tuple[5]}")
                leds.last_sent = led_tuple

            if now - motion.last_broadcast >= 0.1:
                motion.last_broadcast = now
                state_payload = build_state_payload()

        if state_payload is not None:
            await broadcast(state_payload)
        if test_payload is not None:
            await broadcast(test_payload)


async def handle_command(cmd: str) -> dict[str, Any]:
    ser = require_connection()
    async with state_lock:
        resp = ""
        base = cmd.strip().upper()
        if base in ("HOME", "CENTER"):
            # Send the user command (if any) then move to our defined home pose.
            write_line_no_wait(ser, cmd)
            set_home_position(ser)
        else:
            # Non-blocking write for responsiveness.
            write_line_no_wait(ser, cmd)
        if state.test.active:
            state.test.events.append(
                {
                    "t": test_event_time(),
                    "type": "command",
                    "command": cmd.strip(),
                    "target_index": state.test.target_index,
                }
            )
        advanced = maybe_advance_test()
        state_payload = build_state_payload()
        test_payload = build_test_payload() if advanced else None
    await broadcast(state_payload)
    if test_payload is not None:
        await broadcast(test_payload)
    return {"response": resp, "pan": state.pan, "tilt": state.tilt}


async def handle_set(pan: int, tilt: int) -> dict[str, Any]:
    ser = require_connection()
    pan = clamp_angle(pan)
    tilt = clamp_angle(tilt)
    async with state_lock:
        write_line_no_wait(ser, f"P{pan} T{tilt}")
        resp = ""
        state.pan, state.tilt = pan, tilt
        motion_sync_to_state()
        if state.test.active:
            state.test.events.append(
                {
                    "t": test_event_time(),
                    "type": "set",
                    "pan": pan,
                    "tilt": tilt,
                    "target_index": state.test.target_index,
                }
            )
        advanced = maybe_advance_test()
        state_payload = build_state_payload()
        test_payload = build_test_payload() if advanced else None
    await broadcast(state_payload)
    if test_payload is not None:
        await broadcast(test_payload)
    return {"response": resp, "pan": state.pan, "tilt": state.tilt}


def maybe_advance_test() -> bool:
    if not state.test.active:
        return False
    if state.test.target_index >= len(state.test_plan):
        state.test.active = False
        motion.pan_dir = 0
        motion.tilt_dir = 0
        return False
    target = state.test_plan[state.test.target_index]
    within_pan = abs(state.pan - target.get("pan", 0)) <= state.tolerance
    within_tilt = abs(state.tilt - target.get("tilt", 0)) <= state.tolerance
    if not (within_pan and within_tilt):
        return False
    now = time.monotonic()
    elapsed = 0.0
    if state.test.target_started_at is not None:
        elapsed = max(0.0, now - state.test.target_started_at)
    state.test.results.append(
        {
            "index": state.test.target_index,
            "pan": target.get("pan", 0),
            "tilt": target.get("tilt", 0),
            "seconds": round(elapsed, 3),
            "gestures": state.test.gestures_current,
        }
    )
    state.test.events.append(
        {
            "t": test_event_time(now),
            "type": "target_success",
            "index": state.test.target_index,
            "pan": target.get("pan", 0),
            "tilt": target.get("tilt", 0),
            "seconds": round(elapsed, 3),
            "gestures": state.test.gestures_current,
        }
    )
    state.test.target_index += 1
    state.test.gestures_current = 0
    state.test.target_started_at = now
    if state.test.target_index >= len(state.test_plan):
        state.test.events.append(
            {
                "t": test_event_time(now),
                "type": "end_test",
                "reason": "completed",
            }
        )
        state.test.active = False
        motion.pan_dir = 0
        motion.tilt_dir = 0
        if not state.test.saved:
            finalize_test_block("completed")
    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_clients.add(websocket)
    await websocket.send_json(build_state_payload())
    await websocket.send_json(build_test_payload())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_clients.discard(websocket)


@app.websocket("/control")
async def control_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            mtype = message.get("type")
            if mtype == "jog":
                pan_dir = int(message.get("pan_dir", 0))
                tilt_dir = int(message.get("tilt_dir", 0))
                pan_dir = -1 if pan_dir < 0 else (1 if pan_dir > 0 else 0)
                tilt_dir = -1 if tilt_dir < 0 else (1 if tilt_dir > 0 else 0)
                async with state_lock:
                    motion.pan_dir = pan_dir
                    motion.tilt_dir = tilt_dir
                    if motion.last_tick == 0.0:
                        motion.last_tick = time.monotonic()
                await ensure_motion_task()
            elif mtype == "set":
                pan = clamp_angle(int(message.get("pan", state.pan)))
                tilt = clamp_angle(int(message.get("tilt", state.tilt)))
                await handle_set(pan, tilt)
            elif mtype == "command":
                cmd = str(message.get("command", "")).strip()
                if cmd:
                    await handle_command(cmd)
            # ignore unknown types
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "presets": [preset_payload(name) for name in PRESETS],
        "test_allowed": TEST_ALLOWED,
        "test_default": TEST_DEFAULT,
        "free_default": FREE_DEFAULT,
    }


@app.get("/api/ports")
async def get_ports() -> dict[str, Any]:
    ports = []
    for p in list_ports.comports():
        ports.append(
            {
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid,
            }
        )
    return {"ports": ports}


@app.post("/api/preset")
async def set_preset(req: PresetRequest) -> dict[str, Any]:
    name = req.name
    if name not in PRESETS:
        raise HTTPException(status_code=400, detail="Unknown preset.")
    if state.test.active:
        raise HTTPException(status_code=400, detail="Cannot change preset while test is active.")
    if req.context == "test":
        user_id_text = normalize_user_id_text(req.user_id)
        parsed = parse_numeric_user_id(user_id_text)
        if user_id_text is None:
            raise HTTPException(status_code=400, detail="User ID is required for test presets.")
        allowed = allowed_presets_for_user_text(user_id_text)
        if name not in allowed:
            raise HTTPException(status_code=400, detail="Preset not allowed for this user ID.")
        if state.session.active:
            expected = session_expected_preset(state.session.stage)
            if expected is not None and name != expected:
                raise HTTPException(status_code=400, detail="Preset locked for this session stage.")
    async with state_lock:
        apply_preset(name)
        if req.context == "test":
            state.test.user_id_text = user_id_text
            state.test.user_id = parsed if is_valid_user_id(parsed) and parsed > 0 else None
        state_payload = build_state_payload()
    await broadcast(state_payload)
    return state_payload


@app.post("/api/step")
async def set_step(req: StepRequest) -> dict[str, Any]:
    if state.test.active:
        raise HTTPException(status_code=400, detail="Cannot change step size during test.")
    step = max(1, min(20, int(req.step)))
    async with state_lock:
        state.step = step
        state_payload = build_state_payload()
    await broadcast(state_payload)
    return state_payload


@app.post("/api/connect")
async def connect(req: ConnectRequest) -> dict[str, Any]:
    global serial_handle
    async with connect_lock:
        port = req.port or state.port
        if state.device_mode != "real":
            prev = None
            async with state_lock:
                prev = serial_handle
                serial_handle = SimulatedSerial()
                state.connected = True
                state.port = port
                state.last_connect_error = None
                motion.pan_dir = 0
                motion.tilt_dir = 0
                motion_sync_to_state()
                state_payload = build_state_payload()
            if prev is not None:
                try:
                    prev.close()
                except Exception:
                    pass
            await ensure_motion_task()
            await broadcast(state_payload)
            return {"connected": state.connected, "port": state.port, "pan": state.pan, "tilt": state.tilt}
        prev = None
        async with state_lock:
            prev = serial_handle
            serial_handle = None
            state.connected = False
            state.port = port
            state.last_connect_error = None
            motion.pan_dir = 0
            motion.tilt_dir = 0
            state_payload = build_state_payload()
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
        last_exc: Exception | None = None
        for _ in range(6):
            try:
                ser = await asyncio.to_thread(open_port, port)
                break
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(0.35)
        else:
            async with state_lock:
                serial_handle = None
                state.connected = False
                state.last_connect_error = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "Unknown error"
                state_payload = build_state_payload()
            await broadcast(state_payload)
            raise HTTPException(
                status_code=503,
                detail=state.last_connect_error or "Failed to connect to robot.",
            )
        async with state_lock:
            serial_handle = ser
            state.connected = True
            state.last_connect_error = None
            motion_sync_to_state()
            state_payload = build_state_payload()
        await ensure_motion_task()
        await broadcast(state_payload)
        return {"connected": state.connected, "port": state.port, "pan": state.pan, "tilt": state.tilt}


async def auto_connect_loop() -> None:
    backoff = 0.35
    while True:
        await asyncio.sleep(backoff)
        async with state_lock:
            connected = bool(state.connected and serial_handle is not None)
            port = state.port
            device_mode = state.device_mode
        if device_mode != "real":
            backoff = 0.75
            continue
        if connected:
            backoff = 0.75
            continue
        try:
            await connect(ConnectRequest(port=port))
            backoff = 0.75
        except Exception:
            backoff = min(2.0, backoff * 1.4)


@app.post("/api/disconnect")
async def disconnect() -> dict[str, Any]:
    global serial_handle
    async with state_lock:
        if serial_handle is not None:
            try:
                serial_handle.close()
            except Exception:
                pass
        serial_handle = None
        state.connected = False
        motion.pan_dir = 0
        motion.tilt_dir = 0
        state_payload = build_state_payload()
    await broadcast(state_payload)
    return {"connected": state.connected}


@app.post("/api/device-mode")
async def set_device_mode(req: DeviceModeRequest) -> dict[str, Any]:
    mode = (req.mode or "").strip().lower()
    if mode not in ("real", "simulated"):
        raise HTTPException(status_code=400, detail="Unknown device mode.")

    global serial_handle
    async with connect_lock:
        prev = None
        async with state_lock:
            prev = serial_handle
            serial_handle = None
            state.connected = False
            state.last_connect_error = None
            state.device_mode = mode
            motion.pan_dir = 0
            motion.tilt_dir = 0
            if mode != "real":
                serial_handle = SimulatedSerial()
                state.connected = True
                motion_sync_to_state()
            state_payload = build_state_payload()
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
        if state.connected:
            await ensure_motion_task()
        await broadcast(state_payload)
        return state_payload


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return {
        "connected": state.connected,
        "device_mode": state.device_mode,
        "port": state.port,
        "last_connect_error": state.last_connect_error,
        "pan": state.pan,
        "tilt": state.tilt,
        "preset": state.preset,
        "input_mode": state.input_mode,
        "step": state.step,
        "gesture_map": state.gesture_map,
        "test_plan": state.test_plan,
        "tolerance": state.tolerance,
    }


@app.get("/api/test/state")
async def get_test_state() -> dict[str, Any]:
    return build_test_payload()


@app.get("/api/session/state")
async def get_session_state() -> dict[str, Any]:
    if not state.session.active and not state.session.file_path:
        return {"active": False}
    payload = build_session_payload()
    payload["summary"] = build_session_summary()
    return payload


@app.post("/api/session/status")
async def get_session_status(req: SessionStatusRequest) -> dict[str, Any]:
    user_id_text = normalize_user_id_text(req.user_id)
    if user_id_text is None:
        raise HTTPException(status_code=400, detail="User ID is required.")
    existing = find_existing_session(user_id_text)
    if not existing:
        return {
            "exists": False,
            "finished_private": False,
            "finished_public": False,
            "next_environment_key": None,
            "condition_order": None,
        }
    questions, user_dir = existing
    finished_private = environment_completed(user_dir, questions, "private")
    finished_public = environment_completed(user_dir, questions, "public")
    consent = questions.get("consent", {})
    condition_order = None
    if finished_private:
        condition_order = consent.get("private", {}).get("condition_order")
    if not condition_order and finished_public:
        condition_order = consent.get("public", {}).get("condition_order")
    if not condition_order:
        condition_order = consent.get("private", {}).get("condition_order") or consent.get("public", {}).get("condition_order")
    next_env_key = None
    if finished_private and not finished_public:
        next_env_key = "public"
    elif finished_public and not finished_private:
        next_env_key = "private"
    return {
        "exists": True,
        "finished_private": finished_private,
        "finished_public": finished_public,
        "next_environment_key": next_env_key,
        "condition_order": condition_order,
    }


@app.post("/api/session/start")
async def start_session(req: SessionStartRequest) -> dict[str, Any]:
    user_id_text = normalize_user_id_text(req.user_id)
    if user_id_text is None:
        raise HTTPException(status_code=400, detail="User ID is required.")
    allowed = allowed_presets_for_user_text(user_id_text)
    if not allowed:
        raise HTTPException(status_code=400, detail="Invalid user ID.")
    parsed = parse_numeric_user_id(user_id_text)
    async with state_lock:
        if state.session.active:
            state.session.end_reason = "abandoned_new_session"
            state.session.ended_at_wall = time.time()
            state.session.stage = "abandoned"
            state.session.active = False
            save_session_snapshot()

        questions: dict[str, Any] = {}
        user_dir: Path | None = None
        existing = find_existing_session(user_id_text)
        if existing:
            questions, user_dir = existing
        else:
            user_dir = ensure_user_dir(user_id_text)

        env_value = req.environment
        condition_order = req.condition_order

        finished_private = False
        finished_public = False
        if user_dir and questions:
            finished_private = environment_completed(user_dir, questions, "private")
            finished_public = environment_completed(user_dir, questions, "public")
        has_finished_env = finished_private or finished_public

        session = SessionState()
        session.active = True
        session.stage = "prestudy"
        session.started_at = time.monotonic()
        session.started_at_wall = time.time()
        session.user_id_text = user_id_text
        session.user_id = parsed if is_valid_user_id(parsed) and parsed > 0 else None
        session.metadata = {
            "environment": env_value,
            "condition_order": condition_order,
        }
        session.file_path = str(user_dir / QUESTIONS_FILENAME) if user_dir else None
        session.session_id = user_dir.name if user_dir else None
        if has_finished_env:
            session.pre_study = questions.get("prestudy", {})
            session.baseline = questions.get("baseline", {})
            if session.pre_study or session.baseline:
                first, _ = session_order_first_second(condition_order)
                session.stage = f"{first}_intro"
        session.blocks = []
        session.pending_block_index = None
        state.session = session

        # Always track test user for saving results.
        state.test.user_id_text = user_id_text
        state.test.user_id = parsed if is_valid_user_id(parsed) and parsed > 0 else None
        session_payload = save_session_snapshot()
    return session_payload


@app.post("/api/session/prestudy")
async def save_pre_study(req: PreStudyRequest) -> dict[str, Any]:
    async with state_lock:
        if not state.session.active:
            raise HTTPException(status_code=400, detail="Session is not active.")
        # If prestudy already exists (resumed user), keep it and skip overwrite.
        if state.session.pre_study:
            return build_session_payload()
        state.session.pre_study = {
            "age": req.age,
            "gender": normalize_user_id_text(req.gender),
            "handedness": req.handedness,
            "experience": req.experience or [],
        }
        state.session.baseline = req.baseline or {}
        env_value = state.session.metadata.get("environment")
        env_key = environment_key(env_value) or "unknown"
        questions_path = ensure_session_file()
        questions = load_json_file(questions_path)
        questions = update_questions_from_session(questions, state.session)
        save_json_file(questions_path, questions)
        user_dir = questions_path.parent
        stage, pending_scenario = determine_resume_stage(user_dir, questions, env_key, state.session.metadata.get("condition_order"))
        blocks, scenario_to_index = build_blocks_for_environment(
            user_dir,
            questions,
            env_key,
            env_value,
            state.session.metadata.get("condition_order"),
        )
        state.session.blocks = blocks
        state.session.pending_block_index = scenario_to_index.get(pending_scenario) if pending_scenario else None
        state.session.stage = stage
        session_payload = build_session_payload()
    return session_payload


@app.post("/api/session/block")
async def save_block_questionnaire(req: BlockQuestionnaireRequest) -> dict[str, Any]:
    async with state_lock:
        if not state.session.active:
            raise HTTPException(status_code=400, detail="Session is not active.")
        block = update_block_questionnaire(req.block_index, req.answers, req.skipped)
        session_payload = build_session_payload()
    return {"block": block, "session": session_payload}


@app.post("/api/session/final")
async def save_final_questionnaire(req: FinalQuestionnaireRequest) -> dict[str, Any]:
    async with state_lock:
        if not state.session.active:
            raise HTTPException(status_code=400, detail="Session is not active.")
        state.session.final = req.answers or {}
        state.session.final["skipped"] = bool(req.skipped)
        state.session.stage = "done"
        state.session.end_reason = "completed"
        state.session.ended_at_wall = time.time()
        state.session.active = False
        session_payload = save_session_snapshot()
        # Clear in-memory state so next participant starts fresh (results remain on disk).
        reset_test_state()
        reset_session_state()
    return session_payload


@app.post("/api/session/cleanup")
async def cleanup_session(req: SessionCleanupRequest) -> dict[str, Any]:
    user_id_text = normalize_user_id_text(req.user_id)
    if user_id_text is None:
        raise HTTPException(status_code=400, detail="User ID is required.")
    removed = cleanup_user_dirs(user_id_text)
    return {"ok": True, "removed": removed}


@app.post("/api/session/end")
async def end_session(req: SessionEndRequest) -> dict[str, Any]:
    async with state_lock:
        reason = req.reason or "abandoned"
        if state.test.active:
            state.test.events.append(
                {
                    "t": test_event_time(),
                    "type": "end_test",
                    "reason": reason,
                }
            )
        if state.test.started_at is not None and not state.test.saved:
            finalize_test_block(reason)
        reset_test_state()
        if state.session.active:
            if req.stage:
                state.session.metadata["ended_stage"] = req.stage
            if state.session.pending_block_index is not None:
                idx = state.session.pending_block_index
                if 0 <= idx < len(state.session.blocks):
                    block = state.session.blocks[idx]
                    if not block.get("questionnaire_completed_at"):
                        block["questionnaire_abandoned"] = True
                        block["questionnaire_skipped"] = True
                        block["questionnaire_completed_at"] = iso_time(time.time())
            state.session.end_reason = reason
            state.session.ended_at_wall = time.time()
            state.session.stage = "abandoned"
            state.session.active = False
            session_payload = save_session_snapshot()
        else:
            session_payload = build_session_payload()
        # Clear in-memory session so next participant starts fresh; snapshot already saved if any.
        reset_session_state()
    return session_payload


@app.post("/api/command")
async def send_command(req: CommandRequest) -> dict[str, Any]:
    return await handle_command(req.command)


@app.post("/api/set")
async def set_position(req: SetRequest) -> dict[str, Any]:
    return await handle_set(req.pan, req.tilt)


@app.post("/api/gesture")
async def record_gesture(req: GestureRequest) -> dict[str, Any]:
    now = time.monotonic()
    async with state_lock:
        if req.name == "LeftClick":
            leds.left_click_until = max(leds.left_click_until, now + CLICK_LED_SECONDS)
        elif req.name == "RightClick":
            leds.right_click_until = max(leds.right_click_until, now + CLICK_LED_SECONDS)
        if state.test.active:
            state.test.events.append(
                {
                    "t": test_event_time(),
                    "type": "gesture",
                    "name": req.name,
                    "kind": req.kind,
                    "target_index": state.test.target_index,
                }
            )
            state.test.gestures_current += 1
            state.test.gestures_total += 1
        test_payload = build_test_payload()
    await ensure_motion_task()
    await broadcast(test_payload)
    return {
        "gestures_current": state.test.gestures_current,
        "gestures_total": state.test.gestures_total,
    }


@app.get("/api/test-plan")
async def get_test_plan() -> dict[str, Any]:
    return {"targets": state.test_plan, "tolerance": state.tolerance}


@app.post("/api/test-plan")
async def set_test_plan(req: TestPlanRequest) -> dict[str, Any]:
    if not req.targets:
        raise HTTPException(status_code=400, detail="Test plan must include targets.")
    async with state_lock:
        state.test_plan = req.targets
        state.tolerance = TEST_TOLERANCE
        test_payload = build_test_payload()
    await broadcast(test_payload)
    return {"ok": True, "targets": state.test_plan, "tolerance": state.tolerance}


@app.post("/api/test/start")
async def start_test(req: TestStartRequest) -> dict[str, Any]:
    async with state_lock:
        ser = require_connection()
        if not state.test_plan:
            raise HTTPException(status_code=400, detail="Test plan is empty.")
        user_id_text = normalize_user_id_text(req.user_id) or state.test.user_id_text
        if state.session.active:
            if state.session.user_id_text:
                user_id_text = state.session.user_id_text
        parsed = parse_numeric_user_id(user_id_text)
        if user_id_text is None:
            raise HTTPException(status_code=400, detail="User ID is required to start the test.")
        allowed = allowed_presets_for_user_text(user_id_text)
        if not allowed:
            raise HTTPException(status_code=400, detail="Invalid user ID.")
        if state.session.active:
            expected = session_expected_preset(state.session.stage)
            if expected is None or state.session.stage not in (
                "discreet_intro",
                "discreet_test",
                "expressive_intro",
                "expressive_test",
            ):
                raise HTTPException(status_code=400, detail="Session is not ready to start a test.")
            if expected not in allowed:
                raise HTTPException(status_code=400, detail="Preset not allowed for this user ID.")
            apply_preset(expected)
            state.session.stage = f"{expected}_test"
        elif state.preset not in allowed:
            preferred = TEST_DEFAULT if TEST_DEFAULT in allowed else allowed[0]
            apply_preset(preferred)
        state.test.user_id_text = user_id_text
        state.test.user_id = parsed if is_valid_user_id(parsed) and parsed > 0 else None
        # Move to the defined home pose before starting timing.
        motion.pan_dir = 0
        motion.tilt_dir = 0
        set_home_position(ser)
        now = time.monotonic()
        state.test.active = True
        state.test.target_index = 0
        state.test.started_at = now
        state.test.started_at_wall = time.time()
        state.test.target_started_at = now
        state.test.events = []
        state.test.gestures_current = 0
        state.test.gestures_total = 0
        state.test.results = []
        state.test.saved = False
        state.test.last_result_path = None
        state.test.events.append(
            {
                "t": 0.0,
                "type": "start_test",
                "user_id_text": state.test.user_id_text,
                "user_id": state.test.user_id,
                "preset": state.preset,
                "input_mode": state.input_mode,
                "step": state.step,
                "speed_dps": round(step_to_speed_dps(state.step), 3),
                "tolerance": state.tolerance,
            }
        )
        state_payload = build_state_payload()
        test_payload = build_test_payload()
    await broadcast(state_payload)
    await broadcast(test_payload)
    return test_payload


@app.post("/api/test/stop")
async def stop_test(req: TestStopRequest | None = None) -> dict[str, Any]:
    reason = (req.reason or "stopped") if req else "stopped"
    async with state_lock:
        if state.test.active:
            state.test.events.append(
                {
                    "t": test_event_time(),
                    "type": "end_test",
                    "reason": reason,
                }
            )
        if state.test.started_at is not None and not state.test.saved:
            finalize_test_block(reason)
        state.test.active = False
        motion.pan_dir = 0
        motion.tilt_dir = 0
        test_payload = build_test_payload()
    await broadcast(test_payload)
    return test_payload


app.mount("/", StaticFiles(directory="web-ui", html=True), name="static")
