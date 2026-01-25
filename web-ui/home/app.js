import { apiFetch } from "/shared/api.js";
import { watchState } from "/shared/ws.js";
import { initI18n } from "/shared/i18n.js";
import { translations } from "/shared/translations.js";

const elements = {
  goFree: document.getElementById("go-free"),
  goTest: document.getElementById("go-test"),
  overlay: document.getElementById("overlay"),
  status: document.getElementById("connection-status"),
  espModeToggle: document.getElementById("esp-mode-toggle"),
  hint: document.getElementById("home-hint"),
  langToggle: document.getElementById("lang-toggle"),
};

const i18n = initI18n(translations);
const { t, setLang, getLang, applyTranslations, onLangChange } = i18n;

let latestState = null;
let deviceMode = "real";

function updateLangToggle() {
  if (!elements.langToggle) return;
  elements.langToggle.textContent = getLang() === "en" ? t("lang.toggle.to_he") : t("lang.toggle.to_en");
}

function setEspToggleUi(mode) {
  if (!elements.espModeToggle) return;
  const normalized = mode === "simulated" ? "simulated" : "real";
  elements.espModeToggle.textContent = normalized === "simulated" ? t("esp.mode.simulated") : t("esp.mode.real");
  elements.espModeToggle.setAttribute("aria-pressed", normalized === "simulated" ? "true" : "false");
}

function setOverlay(show) {
  elements.overlay.classList.toggle("hidden", !show);
}

function setConnectionUi(connected, lastError, mode = deviceMode) {
  const normalized = mode === "simulated" ? "simulated" : "real";
  elements.status.textContent =
    normalized === "simulated" ? t("status.simulated") : connected ? t("status.connected") : t("status.connecting");
  const dot = document.querySelector(".status-pill .dot");
  if (normalized === "simulated") {
    dot.style.background = "#3b82f6";
    dot.style.boxShadow = "0 0 10px rgba(59, 130, 246, 0.4)";
    elements.hint.textContent = "";
    return;
  }
  if (connected) {
    dot.style.background = "#22c55e";
    dot.style.boxShadow = "0 0 10px rgba(34, 197, 94, 0.4)";
    elements.hint.textContent = "";
  } else {
    dot.style.background = "#ef4444";
    dot.style.boxShadow = "0 0 10px rgba(239, 68, 68, 0.4)";
    elements.hint.textContent = lastError ? t("home.connection_failed", { error: lastError }) : t("home.waiting");
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

async function go(path) {
  const ok = await ensureConnected();
  setOverlay(false);
  if (!ok) return;
  window.location.href = path;
}

elements.goFree.addEventListener("click", () => go("/free/"));
elements.goTest.addEventListener("click", () => go("/test/"));

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
      setConnectionUi(Boolean(payload.connected), payload.last_connect_error, deviceMode);
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
  setConnectionUi(Boolean(latestState?.connected), latestState?.last_connect_error || null, deviceMode);
});

watchState((payload) => {
  latestState = payload;
  deviceMode = payload.device_mode || "real";
  setEspToggleUi(deviceMode);
  setConnectionUi(Boolean(payload.connected), payload.last_connect_error, deviceMode);
});
apiFetch("/api/state")
  .then((s) => {
    latestState = s;
    deviceMode = s.device_mode || "real";
    setEspToggleUi(deviceMode);
    setConnectionUi(Boolean(s.connected), s.last_connect_error, deviceMode);
  })
  .catch(() => setConnectionUi(false, null));
