// Shared page chrome: copy loading, language switch, top-bar hairline on scroll, live region.
import { applyCopy, getLang, loadCopy, setLang } from "./copy.js";

export async function initChrome(onLangChange) {
  await loadCopy(getLang());
  applyCopy();
  syncLangButtons();
  for (const b of document.querySelectorAll("#lang-switch [data-lang]")) {
    b.addEventListener("click", async () => {
      setLang(b.dataset.lang);
      await loadCopy(b.dataset.lang);
      applyCopy();
      syncLangButtons();
      if (onLangChange) onLangChange(b.dataset.lang);
    });
  }
  const bar = document.getElementById("bar");
  if (bar) {
    const onScroll = () => bar.classList.toggle("is-scrolled", window.scrollY > 4);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
}

function syncLangButtons() {
  const lang = getLang();
  for (const b of document.querySelectorAll("#lang-switch [data-lang]")) {
    b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  }
}

export function announce(text) {
  const el = document.getElementById("live");
  if (el) el.textContent = text;
}

export function show(el, on) { if (el) el.hidden = !on; }
