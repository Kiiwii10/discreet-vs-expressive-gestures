
import { apiFetch } from "/shared/api.js";
import { connectControlSocket, watchState } from "/shared/ws.js";
import { initI18n } from "/shared/i18n.js";
import { translations } from "/shared/translations.js";
import {
  collectBlockAnswers,
  collectFinalAnswers,
  collectPrestudyAnswers,
  getCheckedValue,
  renderBlockForm,
  renderFinalForm,
  renderPrestudyForm,
  resetFormInputs,
  restoreForm,
  snapshotForm,
} from "./questionnaire.js";

const elements = {
  status: document.getElementById("connection-status"),
  espModeToggle: document.getElementById("esp-mode-toggle"),
  overlay: document.getElementById("overlay"),
  presetLabel: document.getElementById("preset-label"),
  sessionSummary: document.getElementById("session-summary"),

  exit: document.getElementById("exit-test"),

  stageConsent: document.getElementById("stage-consent"),
  stagePrestudy: document.getElementById("stage-prestudy"),
  stageDiscreetStart: document.getElementById("stage-discreet-start"),
  stageRun: document.getElementById("stage-run"),
  stageBlock: document.getElementById("stage-block-questions"),
  stageExpressiveStart: document.getElementById("stage-expressive-start"),
  stageFinal: document.getElementById("stage-final"),
  stageDone: document.getElementById("stage-done"),

  userId: document.getElementById("user-id"),
  conditionOrder: document.getElementById("condition-order"),
  consentNext: document.getElementById("consent-next"),
  consentError: document.getElementById("consent-error"),

  prestudyForm: document.getElementById("prestudy-form"),
  prestudyBack: document.getElementById("prestudy-back"),
  prestudyNext: document.getElementById("prestudy-next"),
  prestudyError: document.getElementById("prestudy-error"),

  discreetBack: document.getElementById("discreet-back"),
  startDiscreet: document.getElementById("start-discreet"),

  expressiveBack: document.getElementById("expressive-back"),
  startExpressive: document.getElementById("start-expressive"),

  runHeading: document.getElementById("run-heading"),
  targetProgress: document.getElementById("target-progress"),
  targetPan: document.getElementById("target-pan"),
  targetTilt: document.getElementById("target-tilt"),
  currentPan: document.getElementById("current-pan"),
  currentTilt: document.getElementById("current-tilt"),
  poseCompare: document.getElementById("pose-compare"),
  successBanner: document.getElementById("success-banner"),
  gestureGrid: document.getElementById("gesture-grid"),
  mouseLock: document.getElementById("mouse-lock"),
  stopTest: document.getElementById("stop-test"),
  langToggle: document.getElementById("lang-toggle"),

  blockTitle: document.getElementById("block-title"),
  blockSubtitle: document.getElementById("block-subtitle"),
  blockForm: document.getElementById("block-form"),
  blockSkip: document.getElementById("block-skip"),
  blockNext: document.getElementById("block-next"),
  blockError: document.getElementById("block-error"),

  finalForm: document.getElementById("final-form"),
  finalSkip: document.getElementById("final-skip"),
  finalSubmit: document.getElementById("final-submit"),
  finalError: document.getElementById("final-error"),

  doneHome: document.getElementById("done-home"),
};

const i18n = initI18n(translations);
const { t, setLang, getLang, applyTranslations, onLangChange } = i18n;

function updateLangToggle() {
  if (!elements.langToggle) return;
  elements.langToggle.textContent = getLang() === "en" ? t("lang.toggle.to_he") : t("lang.toggle.to_en");
}

let wizardSteps = [
  "prestudy",
  "discreet",
  "discreet-questions",
  "expressive",
  "expressive-questions",
  "final",
  "done",
];
const wizardStepsContainer = document.querySelector(".wizard-steps");

let latestState = null;
let latestTest = null;
let deviceMode = "real";
let userIdText = null;
let sessionStage = null;
let sessionEnvironment = null;
let sessionOrder = null;
let pendingBlockIndex = null;
let currentBlock = null;
let selectedPreset = null;
let currentStage = "consent";
let lastResultsLen = 0;
const DISCREET_CLICK_WINDOW_MS = 2000;
const DISCREET_HOLD_DELAY_MS = 180;
let discreetSequence = [];
let discreetWindowUntil = 0;
let discreetHoldTimer = null;
let discreetHoldDirection = null;
let discreetLeftDown = false;
let discreetRightDown = false;

let controlSocket = null;
connectControlSocket((socket) => {
  controlSocket = socket;
});

function setOverlay(show) {
  elements.overlay.classList.toggle("hidden", !show);
}

function pointerLockSupported() {
  return Boolean(elements.mouseLock && elements.mouseLock.requestPointerLock);
}

function updateMouseLockUi() {
  if (!elements.mouseLock) return;
  if (!pointerLockSupported()) {
    elements.mouseLock.textContent = t("mouse.unsupported");
    elements.mouseLock.disabled = true;
    document.body.classList.remove("mouse-locked");
    return;
  }
  const locked = document.pointerLockElement === elements.mouseLock;
  elements.mouseLock.classList.toggle("locked", locked);
  elements.mouseLock.textContent = locked ? t("mouse.locked") : t("mouse.lock");
  document.body.classList.toggle("mouse-locked", locked);
}

function setConnectionUi(connected) {
  const normalized = deviceMode === "simulated" ? "simulated" : "real";
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

function setEspToggleUi(mode = deviceMode) {
  if (!elements.espModeToggle) return;
  const normalized = mode === "simulated" ? "simulated" : "real";
  elements.espModeToggle.textContent = normalized === "simulated" ? t("esp.mode.simulated") : t("esp.mode.real");
  elements.espModeToggle.setAttribute("aria-pressed", normalized === "simulated" ? "true" : "false");
}

let lastNotConnectedAlertAt = 0;
function alertNotConnected() {
  if (deviceMode !== "real") return false;
  if (latestState?.connected) return false;
  const now = Date.now();
  if (now - lastNotConnectedAlertAt < 1500) return true;
  lastNotConnectedAlertAt = now;
  window.alert(t("error.robot_not_connected"));
  return true;
}

function stageToStep(name) {
  if (name === "prestudy") return "prestudy";
  if (name === "discreet-start" || (name === "run" && currentBlock === "discreet")) return "discreet";
  if (name === "block" && currentBlock === "discreet") return "discreet-questions";
  if (name === "expressive-start" || (name === "run" && currentBlock === "expressive")) return "expressive";
  if (name === "block" && currentBlock === "expressive") return "expressive-questions";
  if (name === "final") return "final";
  if (name === "done") return "done";
  return "prestudy";
}

function markWizardStep(activeStep) {
  if (!wizardStepsContainer) return;
  wizardStepsContainer.querySelectorAll(".wizard-step").forEach((step) => {
    const idx = wizardSteps.indexOf(step.dataset.step);
    step.classList.toggle("active", step.dataset.step === activeStep);
    step.classList.toggle("done", idx !== -1 && idx < wizardSteps.indexOf(activeStep));
  });
}

function showStage(name) {
  elements.stageConsent.classList.toggle("active", name === "consent");
  elements.stagePrestudy.classList.toggle("active", name === "prestudy");
  elements.stageDiscreetStart.classList.toggle("active", name === "discreet-start");
  elements.stageRun.classList.toggle("active", name === "run");
  elements.stageBlock.classList.toggle("active", name === "block");
  elements.stageExpressiveStart.classList.toggle("active", name === "expressive-start");
  elements.stageFinal.classList.toggle("active", name === "final");
  elements.stageDone.classList.toggle("active", name === "done");
  if (wizardStepsContainer) {
    wizardStepsContainer.classList.toggle("hidden", name === "consent");
  }
  currentStage = name;
  if (name !== "run") {
    resetDiscreetControl();
    if (document.pointerLockElement === elements.mouseLock && document.exitPointerLock) {
      document.exitPointerLock();
    }
  }
  document.body.classList.toggle("test-run", name === "run");
  if (name !== "consent") {
    markWizardStep(stageToStep(name));
  } else {
    markWizardStep("consent");
  }
}

function normalizeId(text) {
  const t = String(text ?? "").trim();
  return t.length ? t : null;
}

function prettyPreset(name) {
  if (name === "discreet") return t("preset.discreet");
  if (name === "expressive") return t("preset.expressive");
  return name || "--";
}

function setStageError(el, message) {
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

function updateSessionSummaryText() {
  const parts = [];
  if (userIdText) parts.push(userIdText);
  if (sessionEnvironment) parts.push(prettyEnvironment(sessionEnvironment));
  if (sessionOrder) parts.push(prettyOrder(sessionOrder));
  elements.sessionSummary.textContent = parts.length ? parts.join(" \u2022 ") : "--";
}

function isPrivateEnvironment() {
  return String(sessionEnvironment || "").toLowerCase().includes("private");
}

function prettyEnvironment(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized.includes("private")) return t("test.environment.private");
  if (normalized.includes("public")) return t("test.environment.public");
  return value || t("label.environment");
}

function prettyOrder(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized.includes("expressive")) return t("test.condition_order.expressive_first");
  if (normalized.includes("discreet")) return t("test.condition_order.discreet_first");
  return t("test.condition_order.placeholder");
}

function firstSecondFromOrder(orderValue) {
  const text = String(orderValue || "").toLowerCase();
  if (text.includes("expressive")) return ["expressive", "discreet"];
  return ["discreet", "expressive"];
}

function updateWizardOrder(orderValue) {
  const [first, second] = firstSecondFromOrder(orderValue);
  wizardSteps = [
    "prestudy",
    first,
    `${first}-questions`,
    second,
    `${second}-questions`,
    "final",
    "done",
  ];
  rebuildWizardSteps();
  markWizardStep(stageToStep(currentStage));
}

function rebuildWizardSteps() {
  if (!wizardStepsContainer) return;
  wizardStepsContainer.innerHTML = "";
  const labels = {
    prestudy: t("test.step.background"),
    discreet: t("test.step.discreet"),
    "discreet-questions": t("test.step.discreet_questions"),
    expressive: t("test.step.expressive"),
    "expressive-questions": t("test.step.expressive_questions"),
    final: t("test.step.final"),
    done: t("test.step.done"),
  };
  wizardSteps.forEach((step) => {
    const div = document.createElement("div");
    div.className = "wizard-step";
    div.dataset.step = step;
    div.textContent = labels[step] || step;
    wizardStepsContainer.appendChild(div);
  });
}

function setCurrentBlock(block) {
  currentBlock = block;
  if (block) {
    selectedPreset = block;
    elements.presetLabel.textContent = prettyPreset(block);
  } else {
    selectedPreset = null;
    elements.presetLabel.textContent = "--";
    if (elements.gestureGrid) elements.gestureGrid.innerHTML = "";
  }
  updateRunHeading();
  updateBlockHeader();
  renderGestureGrid();
}

function updateRunHeading() {
  if (!elements.runHeading) return;
  if (!currentBlock) {
    elements.runHeading.textContent = t("test.run.heading");
    return;
  }
  elements.runHeading.textContent = t("test.run.heading_with", { preset: prettyPreset(currentBlock) });
}

function updateBlockHeader() {
  if (!elements.blockSubtitle) return;
  elements.blockSubtitle.textContent = t("test.block.subtitle");
  if (elements.blockTitle) {
    elements.blockTitle.textContent = t("test.block.title_with", { preset: prettyPreset(currentBlock) });
  }
}
function applySessionStage(stage) {
  const mapped = mapSessionStage(stage);
  if (!mapped) return;
  if (stage === "prestudy") setCurrentBlock(null);
  if (stage.startsWith("discreet")) setCurrentBlock("discreet");
  if (stage.startsWith("expressive")) setCurrentBlock("expressive");
   if (sessionOrder) updateWizardOrder(sessionOrder);
  if (mapped === "block") {
    renderBlockForm({ elements, t, isPrivateEnvironment });
    updateBlockHeader();
    resetFormInputs(elements.blockForm);
  }
  if (mapped === "final") {
    renderFinalForm({ elements, t });
    resetFormInputs(elements.finalForm);
  }
  showStage(mapped);
}

function mapSessionStage(stage) {
  switch (stage) {
    case "prestudy":
      return "prestudy";
    case "discreet_intro":
      return "discreet-start";
    case "discreet_test":
      return "run";
    case "discreet_questions":
      return "block";
    case "expressive_intro":
      return "expressive-start";
    case "expressive_test":
      return "run";
    case "expressive_questions":
      return "block";
    case "final":
      return "final";
    case "done":
      return "done";
    default:
      return "consent";
  }
}

function syncSession(payload) {
  if (!payload) return;
  if (payload.active === false) return;
  const summary = payload.summary || payload;
  sessionStage = summary.stage || sessionStage;
  sessionEnvironment = payload.metadata?.environment || summary.environment || sessionEnvironment;
  sessionOrder = payload.metadata?.condition_order || summary.condition_order || sessionOrder;
  if (sessionOrder) updateWizardOrder(sessionOrder);
  pendingBlockIndex = summary.pending_block_index ?? pendingBlockIndex;
  userIdText = payload.user_id_text || summary.user_id || userIdText;
  updateSessionSummaryText();
  if (sessionStage) applySessionStage(sessionStage);
}
function setImageWithFallback(img, urls) {
  const remaining = [...urls];
  const applyNext = () => {
    const next = remaining.shift();
    if (!next) return;
    img.src = next;
  };
  img.onerror = () => applyNext();
  applyNext();
}

function renderGestureGrid() {
  if (!selectedPreset) return;
  let directions = [];
  if (selectedPreset === "expressive") {
    directions = [
      { key: "up", label: t("direction.up"), hint: t("hint.arrow_up") },
      { key: "right", label: t("direction.right"), hint: t("hint.arrow_right") },
      { key: "down", label: t("direction.down"), hint: t("hint.arrow_down") },
      { key: "left", label: t("direction.left"), hint: t("hint.arrow_left") },
    ];
  } else {
    directions = [
      { key: "up", label: t("direction.up"), hint: t("hint.combo.up"), sequence: ["left_click", "pinch"] },
      { key: "down", label: t("direction.down"), hint: t("hint.combo.down"), sequence: ["right_click", "pinch"] },
      { key: "left", label: t("direction.left"), hint: t("hint.combo.left"), sequence: ["left_click", "left_click", "pinch"] },
      { key: "right", label: t("direction.right"), hint: t("hint.combo.right"), sequence: ["right_click", "left_click", "pinch"] },
    ];
  }

  elements.gestureGrid.innerHTML = "";
  directions.forEach((dir) => {
    const card = document.createElement("div");
    card.className = "gesture-card";
    if (dir.sequence) card.classList.add("sequence");

    const head = document.createElement("div");
    head.className = "gesture-head";
    head.textContent = dir.hint ? `${dir.label} (${dir.hint})` : dir.label;

    card.appendChild(head);
    if (dir.sequence) {
      const sequence = document.createElement("div");
      sequence.className = "gesture-sequence";
      dir.sequence.forEach((stepKey) => {
        const img = document.createElement("img");
        img.alt = `${dir.label} gesture step ${stepKey}`;
        setImageWithFallback(img, [
          `/images/gesture/${selectedPreset}/${stepKey}.png`,
          `/images/gesture/${selectedPreset}/${stepKey}.jpg`,
          `/images/gesture/${selectedPreset}/${stepKey}.svg`,
        ]);
        sequence.appendChild(img);
      });
      card.appendChild(sequence);
    } else {
      const img = document.createElement("img");
      img.alt = `${dir.label} gesture reference`;
      setImageWithFallback(img, [
        `/images/gesture/${selectedPreset}/${dir.key}.png`,
        `/images/gesture/${selectedPreset}/${dir.key}.jpg`,
        `/images/gesture/${selectedPreset}/${dir.key}.svg`,
      ]);
      card.appendChild(img);
    }
    elements.gestureGrid.appendChild(card);
  });
}

function updateTargetUi(testPayload) {
  if (!testPayload) return;
  const targets = testPayload.targets ?? [];
  const idx = Number(testPayload.target_index ?? 0);
  const total = targets.length;

  const target = targets[Math.min(idx, Math.max(0, total - 1))] ?? { pan: 0, tilt: 0 };
  elements.targetProgress.textContent = total ? `${Math.min(idx + 1, total)} / ${total}` : "";
  elements.targetPan.textContent = String(target.pan ?? "--");
  elements.targetTilt.textContent = String(target.tilt ?? "--");
}

function flashSuccess() {
  if (!elements.poseCompare || !elements.successBanner) return;
  elements.successBanner.innerHTML = t("test.success");
  elements.successBanner.classList.remove("hidden");
  elements.poseCompare.classList.add("success-flash");
  window.setTimeout(() => {
    elements.poseCompare.classList.remove("success-flash");
    elements.successBanner.classList.add("hidden");
  }, 900);
}

function maybeFlashOnAdvance(testPayload) {
  const len = Array.isArray(testPayload?.results) ? testPayload.results.length : 0;
  if (len > lastResultsLen) {
    lastResultsLen = len;
    if (testPayload?.active) flashSuccess();
  }
}

function allowControlInput() {
  if (currentStage !== "run" || !latestTest?.active) return false;
  if (pointerLockSupported() && document.pointerLockElement !== elements.mouseLock) return false;
  return true;
}

function isInteractiveTarget(target) {
  if (!target || typeof target.closest !== "function") return false;
  if (document.pointerLockElement === elements.mouseLock) return false;
  return Boolean(target.closest("button, a, input, textarea, select, label"));
}

function keyDirection(event) {
  if (selectedPreset === "expressive") {
    const map = {
      ArrowUp: { dir: "up", gesture: "ArrowUp" },
      ArrowDown: { dir: "down", gesture: "ArrowDown" },
      ArrowLeft: { dir: "left", gesture: "ArrowLeft" },
      ArrowRight: { dir: "right", gesture: "ArrowRight" },
    };
    return map[event.key] || null;
  }
  const map = {
    ArrowUp: { dir: "up", gesture: "ArrowUp" },
    ArrowDown: { dir: "down", gesture: "ArrowDown" },
    ArrowLeft: { dir: "left", gesture: "ArrowLeft" },
    ArrowRight: { dir: "right", gesture: "ArrowRight" },
  };
  return map[event.key] || null;
}

function sendJogDirection(direction) {
  if (controlSocket?.readyState !== WebSocket.OPEN) return;
  if (direction && !latestState?.connected) {
    alertNotConnected();
    return;
  }
  const map = {
    left: { pan_dir: 1, tilt_dir: 0 },
    right: { pan_dir: -1, tilt_dir: 0 },
    up: { pan_dir: 0, tilt_dir: 1 },
    down: { pan_dir: 0, tilt_dir: -1 },
  };
  const payload = map[direction] || { pan_dir: 0, tilt_dir: 0 };
  controlSocket.send(JSON.stringify({ type: "jog", ...payload }));
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
  sendJogDirection(null);
}

function clearFormInputs(container) {
  if (!container) return;
  container.querySelectorAll("input, textarea, select").forEach((input) => {
    if (input.type === "checkbox" || input.type === "radio") {
      input.checked = false;
    } else {
      input.value = "";
    }
  });
}

function resetClientState() {
  latestState = null;
  latestTest = null;
  userIdText = null;
  sessionStage = null;
  sessionEnvironment = null;
  sessionOrder = null;
  pendingBlockIndex = null;
  currentBlock = null;
  selectedPreset = null;
  currentStage = "consent";
  lastResultsLen = 0;
  elements.sessionSummary.textContent = "--";
  elements.targetProgress.textContent = "";
  elements.targetPan.textContent = "--";
  elements.targetTilt.textContent = "--";
  elements.currentPan.textContent = "--";
  elements.currentTilt.textContent = "--";
  if (elements.targetImage) elements.targetImage.src = "";
  if (elements.gestureGrid) elements.gestureGrid.innerHTML = "";
  clearFormInputs(elements.prestudyForm);
  clearFormInputs(elements.blockForm);
  clearFormInputs(elements.finalForm);
  showStage("consent");
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
  sendJogDirection(direction);
}

function endDiscreetHold() {
  discreetHoldDirection = null;
  sendJogDirection(null);
}

function recordGesture(name, kind = "key") {
  apiFetch("/api/gesture", {
    method: "POST",
    body: JSON.stringify({ name, kind }),
  }).catch(() => {});
}

function handleDiscreetKeyDown(event) {
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
  if (event.repeat) return;
  event.preventDefault();
}

function handleDiscreetKeyUp(event) {
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
}

function handleDiscreetMouseDown(event) {
  if (event.button !== 0 && event.button !== 2) return;
  if (isInteractiveTarget(event.target)) return;
  if (event.button === 0) {
    event.preventDefault();
    discreetLeftDown = true;
    recordGesture("LeftClick", "mouse");
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
  discreetRightDown = true;
  recordGesture("RightClick", "mouse");
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

function sendJogFromKeys(pressed) {
  if (selectedPreset !== "expressive") return;
  const left = pressed.has("ArrowLeft");
  const right = pressed.has("ArrowRight");
  const up = pressed.has("ArrowUp");
  const down = pressed.has("ArrowDown");
  let panDir = left && !right ? -1 : right && !left ? 1 : 0;
  const tiltDir = up && !down ? 1 : down && !up ? -1 : 0;

  // Invert pan for expressive to match physical orientation.
  panDir = -panDir;

  if (controlSocket?.readyState === WebSocket.OPEN) {
    try {
      controlSocket.send(JSON.stringify({ type: "jog", pan_dir: panDir, tilt_dir: tiltDir }));
    } catch {
      // ignore
    }
  }
}

function enableKeyControlDuringTest() {
  const pressed = new Set();
  window.addEventListener(
    "keydown",
    (event) => {
      if (!allowControlInput()) return;
      if (selectedPreset === "discreet") {
        handleDiscreetKeyDown(event);
        return;
      }
      const dir = keyDirection(event);
      if (!dir) return;
      if (selectedPreset !== "expressive") return;
      event.preventDefault();
      if (event.repeat) return;
      if (pressed.has(event.code || event.key)) return;
      pressed.add(event.code || event.key);
      apiFetch("/api/gesture", {
        method: "POST",
        body: JSON.stringify({ name: dir.gesture, kind: "key" }),
      }).catch(() => {});
      sendJogFromKeys(pressed);
    },
    { passive: false },
  );
  window.addEventListener(
    "keyup",
    (event) => {
      if (!allowControlInput()) return;
      if (selectedPreset === "discreet") {
        handleDiscreetKeyUp(event);
        return;
      }
      const dir = keyDirection(event);
      if (!dir) return;
      if (selectedPreset !== "expressive") return;
      event.preventDefault();
      pressed.delete(event.code || event.key);
      sendJogFromKeys(pressed);
    },
    { passive: false },
  );

  window.addEventListener(
    "mousedown",
    (event) => {
      if (!allowControlInput()) return;
      if (selectedPreset !== "discreet") return;
      handleDiscreetMouseDown(event);
    },
    { passive: false },
  );

  window.addEventListener(
    "mouseup",
    (event) => {
      if (!allowControlInput()) return;
      if (selectedPreset !== "discreet") return;
      handleDiscreetMouseUp(event);
    },
    { passive: false },
  );

  window.addEventListener(
    "contextmenu",
    (event) => {
      if (!allowControlInput()) return;
      if (selectedPreset !== "discreet") return;
      if (isInteractiveTarget(event.target)) return;
      event.preventDefault();
    },
    { passive: false },
  );

  if (elements.mouseLock) {
    updateMouseLockUi();
    elements.mouseLock.addEventListener("click", () => {
      if (!pointerLockSupported()) return;
      elements.mouseLock.requestPointerLock();
    });
    document.addEventListener("pointerlockchange", updateMouseLockUi);
    document.addEventListener("pointerlockerror", updateMouseLockUi);
  }
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

function safeNavigateHome() {
  window.location.href = "/";
}

async function requestExit() {
  const ok = window.confirm(t("confirm.exit"));
  if (!ok) return;
  setOverlay(true);
  try {
    await apiFetch("/api/session/end", {
      method: "POST",
      body: JSON.stringify({ reason: "user_exit", stage: currentStage }),
    });
  } catch {
    // ignore
  }
  setOverlay(false);
  resetClientState();
  safeNavigateHome();
}
async function submitSessionStart() {
  const id = normalizeId(elements.userId.value);
  if (!id) {
    setStageError(elements.consentError, t("error.consent.required"));
    return;
  }
  userIdText = id;
  setStageError(elements.consentError, "");
  setOverlay(true);
  try {
    let environment = getCheckedValue("environment");
    let conditionOrder = elements.conditionOrder.value || null;
    const status = await apiFetch("/api/session/status", {
      method: "POST",
      body: JSON.stringify({ user_id: id }),
    });
    if (status?.exists) {
      if (status.finished_private || status.finished_public) {
        const finishedLabel = status.finished_private ? "private" : "public";
        const ok = window.confirm(`user finished ${finishedLabel} part, skipping consent`);
        if (!ok) return;
        if (status.next_environment_key === "private") {
          environment = "Private (Lab)";
        } else if (status.next_environment_key === "public") {
          environment = "Public (Hallway/Lobby)";
        } else {
          setStageError(elements.consentError, "User already finished both parts.");
          return;
        }
        conditionOrder = status.condition_order || conditionOrder;
      } else {
        const ok = window.confirm("user in DB but didnt finish either part. Clean user dir?");
        if (!ok) return;
        await apiFetch("/api/session/cleanup", {
          method: "POST",
          body: JSON.stringify({ user_id: id }),
        });
      }
    }
    if (!environment || !conditionOrder) {
      setStageError(elements.consentError, t("error.consent.required"));
      return;
    }
    const payload = await apiFetch("/api/session/start", {
      method: "POST",
      body: JSON.stringify({ user_id: id, environment, condition_order: conditionOrder }),
    });
    sessionOrder = conditionOrder;
    updateWizardOrder(sessionOrder);
    syncSession(payload);
    if (sessionStage === "prestudy") {
      renderPrestudyForm({ elements, t });
      showStage("prestudy");
    }
  } catch {
    setStageError(elements.consentError, t("error.save.session"));
  } finally {
    setOverlay(false);
  }
}

async function submitPrestudy() {
  const answers = collectPrestudyAnswers({ elements, t, setStageError });
  if (!answers) return;
  setStageError(elements.prestudyError, "");
  setOverlay(true);
  try {
    const payload = await apiFetch("/api/session/prestudy", {
      method: "POST",
      body: JSON.stringify(answers),
    });
    syncSession(payload);
  } catch {
    setStageError(elements.prestudyError, t("error.save"));
  } finally {
    setOverlay(false);
  }
}

async function startBlockTest(block) {
  if (!userIdText) {
    showStage("consent");
    return;
  }
  setOverlay(true);
  try {
    setCurrentBlock(block);
    const payload = await apiFetch("/api/test/start", {
      method: "POST",
      body: JSON.stringify({ user_id: userIdText }),
    });
    latestTest = payload;
    lastResultsLen = Array.isArray(payload.results) ? payload.results.length : 0;
    updateTargetUi(payload);
    showStage("run");
  } catch {
    if (!alertNotConnected()) {
      showStage("consent");
    }
  } finally {
    setOverlay(false);
  }
}

async function stopTestToQuestions() {
  if (!latestTest?.active) return;
  const ok = window.confirm(t("confirm.stop_test"));
  if (!ok) return;
  setOverlay(true);
  try {
    const payload = await apiFetch("/api/test/stop", {
      method: "POST",
      body: JSON.stringify({ reason: "stopped" }),
    });
    latestTest = payload;
    if (payload.session) {
      syncSession(payload.session);
    }
  } catch {
    // ignore
  } finally {
    setOverlay(false);
  }
}

async function submitBlock(skip) {
  if (!skip) {
    const answers = collectBlockAnswers({
      elements,
      t,
      setStageError,
      currentBlock,
      sessionEnvironment,
      isPrivateEnvironment,
    });
    if (!answers) return;
    setStageError(elements.blockError, "");
    setOverlay(true);
    try {
      const payload = await apiFetch("/api/session/block", {
        method: "POST",
        body: JSON.stringify({ block_index: pendingBlockIndex, answers }),
      });
      syncSession(payload.session);
    } catch {
      setStageError(elements.blockError, t("error.save.block"));
    } finally {
      setOverlay(false);
    }
    return;
  }

  const ok = window.confirm(t("confirm.skip_block"));
  if (!ok) return;
  setOverlay(true);
  try {
    const payload = await apiFetch("/api/session/block", {
      method: "POST",
      body: JSON.stringify({ block_index: pendingBlockIndex, skipped: true }),
    });
    syncSession(payload.session);
  } catch {
    setStageError(elements.blockError, t("error.save.block"));
  } finally {
    setOverlay(false);
  }
}

async function submitFinal(skip) {
  if (!skip) {
    const answers = collectFinalAnswers({ elements, t, setStageError });
    if (!answers) return;
    setStageError(elements.finalError, "");
    setOverlay(true);
    try {
      const payload = await apiFetch("/api/session/final", {
        method: "POST",
        body: JSON.stringify({ answers }),
      });
      syncSession(payload);
      showStage("done");
    } catch {
      setStageError(elements.finalError, t("error.save.final"));
    } finally {
      setOverlay(false);
    }
    return;
  }

  const ok = window.confirm(t("confirm.skip_final"));
  if (!ok) return;
  setOverlay(true);
  try {
    const payload = await apiFetch("/api/session/final", {
      method: "POST",
      body: JSON.stringify({ skipped: true }),
    });
    syncSession(payload);
    showStage("done");
  } catch {
    setStageError(elements.finalError, t("error.save.final"));
  } finally {
    setOverlay(false);
  }
}

function setImageValuesFromState(payload) {
  if (typeof payload.pan === "number") elements.currentPan.textContent = String(payload.pan);
  if (typeof payload.tilt === "number") elements.currentTilt.textContent = String(payload.tilt);
}

function updateSessionFromTest(payload) {
  if (payload?.session?.stage) {
    pendingBlockIndex = payload.session.pending_block_index ?? pendingBlockIndex;
    sessionStage = payload.session.stage;
    sessionEnvironment = payload.session.environment || sessionEnvironment;
    sessionOrder = payload.session.condition_order || sessionOrder;
    userIdText = payload.session.user_id || userIdText;
    updateSessionSummaryText();
    applySessionStage(sessionStage);
  } else if (payload?.active) {
    showStage("run");
  }
}

elements.exit.addEventListener("click", () => requestExit());

elements.consentNext.addEventListener("click", () => submitSessionStart());

elements.prestudyBack.addEventListener("click", () => showStage("consent"));
elements.prestudyNext.addEventListener("click", () => submitPrestudy());

elements.discreetBack.addEventListener("click", () => showStage("prestudy"));
elements.startDiscreet.addEventListener("click", () => startBlockTest("discreet"));

elements.expressiveBack.addEventListener("click", () => showStage("block"));
elements.startExpressive.addEventListener("click", () => startBlockTest("expressive"));

elements.stopTest.addEventListener("click", () => stopTestToQuestions());

elements.blockSkip.addEventListener("click", () => submitBlock(true));
elements.blockNext.addEventListener("click", () => submitBlock(false));

elements.finalSkip.addEventListener("click", () => submitFinal(true));
elements.finalSubmit.addEventListener("click", () => submitFinal(false));

elements.doneHome.addEventListener("click", () => safeNavigateHome());

if (elements.langToggle) {
  elements.langToggle.addEventListener("click", () => {
    setLang(getLang() === "en" ? "he" : "en");
  });
}

if (elements.espModeToggle) {
  elements.espModeToggle.addEventListener("click", async () => {
    const next = deviceMode === "simulated" ? "real" : "simulated";
    try {
      const payload = await apiFetch("/api/device-mode", {
        method: "POST",
        body: JSON.stringify({ mode: next }),
      });
      latestState = payload;
      deviceMode = payload.device_mode || "real";
      setEspToggleUi(deviceMode);
      setConnectionUi(Boolean(payload.connected));
    } catch {
      // ignore
    }
  });
}

applyTranslations();
updateLangToggle();
setEspToggleUi(deviceMode);
onLangChange(() => {
  applyTranslations();
  updateLangToggle();
  setEspToggleUi(deviceMode);
  updateMouseLockUi();
  setConnectionUi(Boolean(latestState?.connected));
  elements.presetLabel.textContent = prettyPreset(selectedPreset);
  updateSessionSummaryText();
  updateRunHeading();
  updateBlockHeader();
  renderGestureGrid();

  if (currentStage === "prestudy") {
    const snap = snapshotForm(elements.prestudyForm);
    renderPrestudyForm({ elements, t });
    restoreForm(elements.prestudyForm, snap);
  } else if (currentStage === "block") {
    const snap = snapshotForm(elements.blockForm);
    renderBlockForm({ elements, t, isPrivateEnvironment });
    updateBlockHeader();
    restoreForm(elements.blockForm, snap);
  } else if (currentStage === "final") {
    const snap = snapshotForm(elements.finalForm);
    renderFinalForm({ elements, t });
    restoreForm(elements.finalForm, snap);
  }
});

enableKeyControlDuringTest();

watchState(
  (payload) => {
    latestState = payload;
    deviceMode = payload.device_mode || "real";
    setEspToggleUi(deviceMode);
    setConnectionUi(Boolean(payload.connected));
    if (!currentBlock && payload.preset) {
      selectedPreset = payload.preset;
      elements.presetLabel.textContent = prettyPreset(payload.preset);
      renderGestureGrid();
    }
    setImageValuesFromState(payload);
  },
  (payload) => {
    latestTest = payload;
    maybeFlashOnAdvance(payload);
    updateTargetUi(payload);
    elements.stopTest.disabled = !payload.active;
    updateSessionFromTest(payload);
  },
);

(async () => {
  rebuildWizardSteps();
  renderPrestudyForm({ elements, t });
  renderBlockForm({ elements, t, isPrivateEnvironment });
  renderFinalForm({ elements, t });

  setOverlay(true);
  const ok = await ensureConnected();
  setOverlay(false);
  if (!ok) return safeNavigateHome();

  try {
    const [state, test, session] = await Promise.all([
      apiFetch("/api/state"),
      apiFetch("/api/test/state"),
      apiFetch("/api/session/state").catch(() => null),
    ]);
    latestState = state;
    latestTest = test;
    deviceMode = state.device_mode || "real";
    setEspToggleUi(deviceMode);
    setConnectionUi(Boolean(state.connected));
    selectedPreset = state.preset;
    elements.presetLabel.textContent = prettyPreset(selectedPreset);
    setImageValuesFromState(state);
    renderGestureGrid();

    if (session && session.active !== false) {
      syncSession(session);
    } else if (test?.active) {
      userIdText = test.user_id ?? null;
      lastResultsLen = Array.isArray(test.results) ? test.results.length : 0;
      updateTargetUi(test);
      showStage("run");
    } else {
      resetClientState();
    }
  } catch {
    safeNavigateHome();
  }
})();
