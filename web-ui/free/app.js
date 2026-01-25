import { apiFetch } from "/shared/api.js";
import { watchState, connectControlSocket } from "/shared/ws.js";
import { initI18n } from "/shared/i18n.js";
import { translations } from "/shared/translations.js";

const elements = {
  back: document.getElementById("back-home"),
  status: document.getElementById("connection-status"),
  overlay: document.getElementById("overlay"),
  presetLabel: document.getElementById("preset-label"),
  stepLabel: document.getElementById("step-label"),
  pan: document.getElementById("pan"),
  tilt: document.getElementById("tilt"),
  panSlider: document.getElementById("pan-slider"),
  tiltSlider: document.getElementById("tilt-slider"),
  step: document.getElementById("step"),
  log: document.getElementById("log-entries"),
  clearLog: document.getElementById("clear-log"),
  gestureDebug: document.getElementById("gesture-debug"),
  mouseLock: document.getElementById("mouse-lock"),
  presetButtons: Array.from(document.querySelectorAll(".preset-button")),
  langToggle: document.getElementById("lang-toggle"),
  espModeToggle: document.getElementById("esp-mode-toggle"),
};

const i18n = initI18n(translations);
const { t, setLang, getLang, applyTranslations, onLangChange } = i18n;

const state = {
  connected: false,
  deviceMode: "real",
  preset: "discreet",
  step: 4,
  pan: 90,
  tilt: 60,
};

let controlSocket = null;
const activeDirections = new Set();
const DISCREET_CLICK_WINDOW_MS = 2000;
const DISCREET_HOLD_DELAY_MS = 180;
let discreetSequence = [];
let discreetWindowUntil = 0;
let discreetHoldTimer = null;
let discreetHoldDirection = null;
let discreetLeftDown = false;
let discreetRightDown = false;

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  const div = document.createElement("div");
  div.className = "log-entry";
  div.textContent = `[${stamp}] ${message}`;
  elements.log.prepend(div);
  while (elements.log.children.length > 60) {
    elements.log.lastChild.remove();
  }
}

function formatKeyLabel(key) {
  const map = {
    ArrowUp: t("key.arrow_up"),
    ArrowDown: t("key.arrow_down"),
    ArrowLeft: t("key.arrow_left"),
    ArrowRight: t("key.arrow_right"),
  };
  return map[key] || key;
}

function logPress(label) {
  if (!elements.log) return;
  log(t("log.pressed", { label }));
}

function sendGesture(name, kind = "mouse") {
  apiFetch("/api/gesture", {
    method: "POST",
    body: JSON.stringify({ name, kind }),
  }).catch(() => {});
}

function updatePresetButtons() {
  if (!elements.presetButtons?.length) return;
  elements.presetButtons.forEach((button) => {
    const target = button.dataset.preset;
    button.classList.toggle("selected", target === state.preset);
  });
}

function pointerLockSupported() {
  return Boolean(elements.mouseLock && elements.mouseLock.requestPointerLock);
}

function updateMouseLockUi() {
  if (!elements.mouseLock) return;
  if (!pointerLockSupported()) {
    elements.mouseLock.textContent = t("mouse.unsupported");
    elements.mouseLock.disabled = true;
    return;
  }
  const locked = document.pointerLockElement === elements.mouseLock;
  elements.mouseLock.classList.toggle("locked", locked);
  elements.mouseLock.textContent = locked ? t("mouse.locked") : t("mouse.lock");
}

function setOverlay(show) {
  elements.overlay.classList.toggle("hidden", !show);
}

function setConnectionUi(connected, mode = state.deviceMode) {
  const normalized = mode === "simulated" ? "simulated" : "real";
  elements.status.textContent =
    normalized === "simulated" ? t("status.simulated") : connected ? t("status.connected") : t("status.connecting");
  const dot = document.querySelector(".status-pill .dot");
  if (normalized === "simulated") {
    dot.style.background = "#3b82f6";
    dot.style.boxShadow = "0 0 10px rgba(59, 130, 246, 0.4)";
    return;
  }
  if (connected) {
    dot.style.background = "#22c55e";
    dot.style.boxShadow = "0 0 10px rgba(34, 197, 94, 0.4)";
  } else {
    dot.style.background = "#ef4444";
    dot.style.boxShadow = "0 0 10px rgba(239, 68, 68, 0.4)";
  }
}

function setEspToggleUi(mode = state.deviceMode) {
  if (!elements.espModeToggle) return;
  const normalized = mode === "simulated" ? "simulated" : "real";
  elements.espModeToggle.textContent = normalized === "simulated" ? t("esp.mode.simulated") : t("esp.mode.real");
  elements.espModeToggle.setAttribute("aria-pressed", normalized === "simulated" ? "true" : "false");
}

function applyStatePayload(payload) {
  state.connected = Boolean(payload.connected);
  state.deviceMode = payload.device_mode || state.deviceMode;
  const nextPreset = payload.preset || state.preset;
  if (state.preset === "discreet" && nextPreset !== state.preset) {
    resetDiscreetControl();
  }
  state.preset = nextPreset;
  state.step = typeof payload.step === "number" ? payload.step : state.step;
  state.pan = typeof payload.pan === "number" ? payload.pan : state.pan;
  state.tilt = typeof payload.tilt === "number" ? payload.tilt : state.tilt;

  setConnectionUi(state.connected, state.deviceMode);
  setEspToggleUi(state.deviceMode);
  elements.presetLabel.textContent = prettyPreset(state.preset);
  elements.stepLabel.textContent = String(state.step);
  elements.step.value = String(state.step);
  elements.pan.textContent = `${state.pan} deg`;
  elements.tilt.textContent = `${state.tilt} deg`;
  elements.panSlider.value = String(state.pan);
  elements.tiltSlider.value = String(state.tilt);
  elements.gestureDebug.classList.remove("hidden");
  updatePresetButtons();
}

function prettyPreset(name) {
  if (name === "discreet") return t("preset.discreet");
  if (name === "expressive") return t("preset.expressive");
  return name || "--";
}

function currentJogVector() {
  let panDir = 0;
  let tiltDir = 0;
  if (activeDirections.has("left")) panDir += 1;
  if (activeDirections.has("right")) panDir -= 1;
  if (activeDirections.has("up")) tiltDir += 1;
  if (activeDirections.has("down")) tiltDir -= 1;
  if (panDir < -1) panDir = -1;
  if (panDir > 1) panDir = 1;
  if (tiltDir < -1) tiltDir = -1;
  if (tiltDir > 1) tiltDir = 1;
  return { panDir, tiltDir };
}

function sendJog() {
  if (!controlSocket || controlSocket.readyState !== WebSocket.OPEN) return;
  const { panDir, tiltDir } = currentJogVector();
  controlSocket.send(JSON.stringify({ type: "jog", pan_dir: panDir, tilt_dir: tiltDir }));
}

let lastNotConnectedAlertAt = 0;
function alertNotConnected() {
  if (state.deviceMode !== "real") return false;
  if (state.connected) return false;
  const now = Date.now();
  if (now - lastNotConnectedAlertAt < 1500) return true;
  lastNotConnectedAlertAt = now;
  const message = t("error.robot_not_connected");
  if (elements.log) log(message);
  window.alert(message);
  return true;
}

function startDirection(direction) {
  if (!state.connected) {
    alertNotConnected();
    return;
  }
  if (activeDirections.has(direction)) return;
  activeDirections.add(direction);
  sendJog();
}

function stopDirection(direction) {
  if (!activeDirections.has(direction)) return;
  activeDirections.delete(direction);
  sendJog();
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function isInteractiveTarget(target) {
  if (!target || typeof target.closest !== "function") return false;
  if (document.pointerLockElement === elements.mouseLock) return false;
  return Boolean(target.closest("button, a, input, textarea, select, label"));
}

function keyToDirection(event) {
  if (state.preset !== "expressive") return null;
  const map = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" };
  return map[event.key] || null;
}

function setSoloDirection(direction) {
  activeDirections.clear();
  if (direction) activeDirections.add(direction);
  sendJog();
}

function resetDiscreetControl() {
  discreetLeftDown = false;
  discreetRightDown = false;
  discreetHoldDirection = null;
  discreetSequence = [];
  discreetWindowUntil = 0;
  if (discreetHoldTimer) {
    window.clearTimeout(discreetHoldTimer);
    discreetHoldTimer = null;
  }
  setSoloDirection(null);
}

function registerDiscreetClick(kind) {
  const now = Date.now();
  if (now > discreetWindowUntil) discreetSequence = [];
  discreetSequence.push(kind);
  if (discreetSequence.length > 2) {
    discreetSequence = discreetSequence.slice(-2);
  }
  discreetWindowUntil = now + DISCREET_CLICK_WINDOW_MS;
}

function resolveDiscreetHoldDirection(sequence) {
  const key = sequence.join("");
  if (key === "LL") return "up";
  if (key === "RL") return "down";
  if (key === "LLL") return "left";
  if (key === "RLL") return "right";
  return null;
}

function beginDiscreetHold(direction) {
  if (!direction) return;
  discreetHoldDirection = direction;
  discreetSequence = [];
  discreetWindowUntil = 0;
  setSoloDirection(direction);
}

function endDiscreetHold() {
  discreetHoldDirection = null;
  setSoloDirection(null);
}

function handleDiscreetMouseDown(event) {
  if (event.button !== 0 && event.button !== 2) return;
  if (isInteractiveTarget(event.target)) return;
  if (event.button === 0) {
    event.preventDefault();
    logPress(t("mouse.left_click"));
    sendGesture("LeftClick");
    discreetLeftDown = true;
    if (discreetHoldTimer) window.clearTimeout(discreetHoldTimer);
    discreetHoldTimer = window.setTimeout(() => {
      discreetHoldTimer = null;
      if (!discreetLeftDown) return;
      const baseSequence = Date.now() > discreetWindowUntil ? [] : discreetSequence;
      const direction = resolveDiscreetHoldDirection([...baseSequence, "L"]);
      if (direction) beginDiscreetHold(direction);
    }, DISCREET_HOLD_DELAY_MS);
    return;
  }

  event.preventDefault();
  logPress(t("mouse.right_click"));
  sendGesture("RightClick");
  discreetRightDown = true;
}

function handleDiscreetMouseUp(event) {
  if (event.button !== 0 && event.button !== 2) return;
  if (event.button === 0) {
    event.preventDefault();
    discreetLeftDown = false;
    if (discreetHoldTimer) {
      window.clearTimeout(discreetHoldTimer);
      discreetHoldTimer = null;
      registerDiscreetClick("L");
      return;
    }
    if (discreetHoldDirection) {
      endDiscreetHold();
      return;
    }
    registerDiscreetClick("L");
    return;
  }

  discreetRightDown = false;
  registerDiscreetClick("R");
}

function handleDiscreetKeyDown(event) {
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
  if (event.repeat) return;
  event.preventDefault();
  logPress(formatKeyLabel(event.key));
}

function handleDiscreetKeyUp(event) {
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
}

async function ensureConnected() {
  const state = await apiFetch("/api/state");
  if (state.connected) return true;
  setOverlay(true);
  try {
    await apiFetch("/api/connect", { method: "POST", body: JSON.stringify({}) });
  } catch {
    // ignore; we'll keep polling
  }
  const deadline = Date.now() + 12000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 350));
    const next = await apiFetch("/api/state");
    if (next.connected) return true;
  }
  return false;
}

async function guardConnection() {
  const ok = await ensureConnected();
  if (!ok) {
    window.location.href = "/";
    return false;
  }
  setOverlay(false);
  return true;
}

if (elements.back) {
  elements.back.addEventListener("click", () => {
    window.location.href = "/";
  });
}

if (elements.clearLog) {
  elements.clearLog.addEventListener("click", () => {
    if (elements.log) elements.log.innerHTML = "";
  });
}

if (elements.presetButtons?.length) {
  elements.presetButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const target = button.dataset.preset;
      if (!target || target === state.preset) return;
      try {
        const payload = await apiFetch("/api/preset", {
          method: "POST",
          body: JSON.stringify({ name: target, context: "free" }),
        });
        applyStatePayload(payload);
      } catch {
        log(t("free.switch_failed", { preset: prettyPreset(target) }));
      }
    });
  });
}

if (elements.mouseLock) {
  updateMouseLockUi();
  elements.mouseLock.addEventListener("click", () => {
    if (!pointerLockSupported()) return;
    elements.mouseLock.requestPointerLock();
  });
  document.addEventListener("pointerlockchange", updateMouseLockUi);
  document.addEventListener("pointerlockerror", updateMouseLockUi);
}

if (elements.langToggle) {
  elements.langToggle.addEventListener("click", () => {
    setLang(getLang() === "en" ? "he" : "en");
  });
}

if (elements.espModeToggle) {
  elements.espModeToggle.addEventListener("click", async () => {
    const next = state.deviceMode === "simulated" ? "real" : "simulated";
    try {
      const payload = await apiFetch("/api/device-mode", {
        method: "POST",
        body: JSON.stringify({ mode: next }),
      });
      applyStatePayload(payload);
    } catch {
      // ignore
    }
  });
}

applyTranslations();
if (elements.langToggle) {
  elements.langToggle.textContent = getLang() === "en" ? t("lang.toggle.to_he") : t("lang.toggle.to_en");
}
setEspToggleUi(state.deviceMode);
onLangChange(() => {
  applyTranslations();
  if (elements.langToggle) {
    elements.langToggle.textContent = getLang() === "en" ? t("lang.toggle.to_he") : t("lang.toggle.to_en");
  }
  setEspToggleUi(state.deviceMode);
  updateMouseLockUi();
  applyStatePayload(state);
});

document.addEventListener("keydown", (event) => {
  if (isTypingTarget(event.target)) return;
  if (!state.connected) {
    alertNotConnected();
    return;
  }
  if (state.preset === "discreet") {
    handleDiscreetKeyDown(event);
    return;
  }
  if (event.repeat) return;
  const dir = keyToDirection(event);
  if (!dir) return;
  event.preventDefault();
  logPress(formatKeyLabel(event.key));
  startDirection(dir);
});

document.addEventListener("keyup", (event) => {
  if (isTypingTarget(event.target)) return;
  if (!state.connected) return;
  if (state.preset === "discreet") {
    handleDiscreetKeyUp(event);
    return;
  }
  const dir = keyToDirection(event);
  if (!dir) return;
  stopDirection(dir);
});

document.addEventListener(
  "mousedown",
  (event) => {
    if (state.preset !== "discreet") return;
    if (isInteractiveTarget(event.target)) return;
    if (!state.connected) {
      alertNotConnected();
      return;
    }
    handleDiscreetMouseDown(event);
  },
  { passive: false },
);

document.addEventListener(
  "mouseup",
  (event) => {
    if (state.preset !== "discreet") return;
    if (isInteractiveTarget(event.target)) return;
    if (!state.connected) return;
    handleDiscreetMouseUp(event);
  },
  { passive: false },
);

document.addEventListener(
  "contextmenu",
  (event) => {
    if (state.preset !== "discreet") return;
    if (isInteractiveTarget(event.target)) return;
    if (!state.connected) return;
    event.preventDefault();
  },
  { passive: false },
);

watchState((payload) => applyStatePayload(payload));
connectControlSocket((sock) => {
  controlSocket = sock;
});

guardConnection()
  .then(() => apiFetch("/api/state"))
  .then((st) => applyStatePayload(st))
  .catch(() => {});
