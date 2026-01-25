let currentLang = localStorage.getItem("lang") || "en";
const listeners = new Set();

function formatString(value, vars) {
  if (!vars) return value;
  return value.replace(/\{(\w+)\}/g, (_, key) => (vars[key] != null ? String(vars[key]) : ""));
}

export function initI18n(translations) {
  function setLang(lang) {
    if (!translations[lang]) return;
    currentLang = lang;
    localStorage.setItem("lang", lang);
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "he" ? "rtl" : "ltr";
    listeners.forEach((fn) => fn(lang));
  }

  function getLang() {
    return currentLang;
  }

  function t(key, vars) {
    const bundle = translations[currentLang] || translations.en || {};
    const fallback = translations.en || {};
    const value = bundle[key] ?? fallback[key] ?? key;
    return formatString(value, vars);
  }

  function applyTranslations(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.dataset.i18n;
      if (!key) return;
      el.textContent = t(key);
    });
    root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const key = el.dataset.i18nPlaceholder;
      if (!key) return;
      el.setAttribute("placeholder", t(key));
    });
  }

  function onLangChange(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  // Initialize document state.
  document.documentElement.lang = currentLang;
  document.documentElement.dir = currentLang === "he" ? "rtl" : "ltr";

  return { t, setLang, getLang, applyTranslations, onLangChange };
}
