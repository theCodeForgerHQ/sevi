// Landing page (B spec §10.2, §10.3 BOOT/BACK_SOON/LANDING).
import * as api from "./api.js";
import { setLang, getLang } from "./copy.js";
import { initChrome, show } from "./ui.js";

const $ = (id) => document.getElementById(id);
const START_KEY = "sevi.start";
let status = null;

function form() { return $("landing"); }
function earphones() {
  const v = form().elements.earphones.value;
  return v === "" ? null : v === "yes";
}

function refresh() {
  const f = form();
  const ep = earphones();
  show($("earphones-note"), ep === false);
  $("start-keypad").disabled = ep === null || !api.features().keypad;
  const voiceOk = !!status && status.voice_offered && api.features().voice && !(api.isIOS() && ep === false);
  show($("try-voice"), voiceOk);
  $("try-voice").disabled = ep === null;
  show($("voice-busy"), !!status && !status.voice_offered && api.features().voice);
  show($("pilot-line"), !!status && status.phone_pilot);
  f.elements.lang.value = getLang();
  if (status) {
    for (const l of status.languages) {
      const input = f.querySelector(`input[name=lang][value=${l.lang}]`);
      if (input) input.disabled = !l.available;
    }
  }
}

function go(mode) {
  const ep = earphones();
  if (ep === null) return;
  sessionStorage.setItem(START_KEY, JSON.stringify({ mode, earphones: ep, lang: getLang() }));
  location.href = "test.html";
}

async function boot() {
  await initChrome(refresh);
  const f = form();
  for (const r of f.elements.lang) r.addEventListener("change", async () => {
    setLang(r.value);
    document.querySelector(`#lang-switch [data-lang=${r.value}]`).click();
  });
  for (const r of f.elements.earphones) r.addEventListener("change", refresh);
  f.addEventListener("submit", (e) => { e.preventDefault(); go("keypad"); });
  $("try-voice").addEventListener("click", () => go("voice"));

  let up = false;
  try { await api.loadConfig(); up = await api.ready(); } catch { up = false; }
  show($("boot"), false);
  if (!up) { show($("back-soon"), true); return; }
  show($("unsupported"), !api.features().keypad);
  try { status = await api.status(); } catch { status = null; }
  if (status && status.admission_level === "L4") { show($("back-soon"), true); return; }
  show(f, true);
  refresh();
  setInterval(async () => { try { status = await api.status(); refresh(); } catch { /* keep last */ } }, 30000);
}

boot();
