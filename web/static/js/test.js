// test.html: the whole session (B spec §10.3-§10.7) and the in-place result view.
import * as api from "./api.js";
import { Mic, Player, loadCaptureWorklet, unlock } from "./audio.js";
import { lang, t } from "./copy.js";
import { RESULT_KEY, renderResult } from "./result-view.js";
import { CANT_HEAR_STATES, LIVE_STATES, S, SessionClient, next } from "./session.js";
import { announce, initChrome, show } from "./ui.js";

const $ = (id) => document.getElementById(id);
const WAIT_RETRY_MS = 60000;
const QUEUE_POLL_MS = 5000;

const start = JSON.parse(sessionStorage.getItem("sevi.start") || "null");
let state = S.BOOT;
let ctx = null;
let client = null;
let mic = null;
let phase = null; // last UiPhase
let feedback = null;
let result = null;
let explanation = null;
let denied = false;
let mode = start ? start.mode : "keypad";

function go(ev) {
  const prev = state;
  state = next(state, ev);
  if (state !== prev) render();
}

// ---------- rendering ----------

function buildKeypad() {
  const pad = $("keypad");
  pad.replaceChildren();
  for (let d = 1; d <= 9; d++) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "key";
    b.textContent = String(d);
    b.dataset.digit = String(d);
    b.addEventListener("click", () => press(String(d)));
    pad.append(b);
  }
  const del = document.createElement("button");
  del.type = "button";
  del.className = "key key-action";
  del.id = "key-delete";
  del.textContent = "⌫";
  del.addEventListener("click", erase);
  const ok = document.createElement("button");
  ok.type = "button";
  ok.className = "key key-action key-confirm";
  ok.id = "key-confirm";
  ok.textContent = "✓";
  ok.addEventListener("click", confirmAnswer);
  const gap = document.createElement("span");
  gap.className = "key-gap";
  pad.append(del, gap, ok);
  labelKeypad();
}

function labelKeypad() {
  for (const b of $("keypad").querySelectorAll("[data-digit]")) b.setAttribute("aria-label", t(`ui.digit.${b.dataset.digit}`));
  $("key-delete").setAttribute("aria-label", t("ui.test.key_delete"));
  $("key-confirm").setAttribute("aria-label", t("ui.test.key_confirm"));
}

function phaseKey() {
  if (!phase) return "ui.test.phase_instructions";
  if (phase.state === "INSTRUCTIONS") return "ui.test.phase_instructions";
  return { VOLUME: "ui.test.phase_volume", PRACTICE: "ui.test.phase_practice", LOUD: "ui.test.phase_loud", SCORED: "ui.test.phase_test" }[phase.kind] || "ui.test.phase_test";
}

function renderDots() {
  const held = client ? client.digits : [];
  [...document.querySelectorAll(".dots i")].forEach((d, i) => {
    d.classList.toggle("on", i < held.length);
    d.textContent = held[i] || "";
  });
}

function render() {
  document.body.dataset.state = state;
  for (const v of document.querySelectorAll("[data-view]")) v.hidden = !v.dataset.view.split(" ").includes(state);
  show($("cant-hear-bar"), CANT_HEAR_STATES.has(state));

  const scored = phase && phase.kind === "SCORED" && phase.scored_index;
  show($("progress"), LIVE_STATES.has(state) && !!scored);
  $("round").textContent = scored ? t("ui.test.round", { k: phase.scored_index, n: 24 }) : "";
  $("track").style.setProperty("--p", scored ? String(phase.scored_index / 24) : "0");

  $("busy-text").textContent = state === S.CONNECTING || state === S.ADMITTING ? t("ui.test.connecting") : "";
  $("phase-title").textContent = t(phaseKey());
  const answering = state === S.ANSWER;
  const voiceAnswer = answering && mode === "voice";
  $("listen-text").textContent = t(voiceAnswer ? "ui.test.listening" : answering ? "ui.test.keypad_hint" : "ui.test.listen");
  $("listen-line").dataset.on = String(!answering || voiceAnswer);
  show($("keypad-hint"), false);
  show($("keypad-area"), state === S.ANSWER || state === S.LISTEN || state === S.INSTRUCTIONS);
  show($("repeat"), state === S.INSTRUCTIONS || (phase && phase.kind === "PRACTICE"));
  $("hear-again").disabled = !client || client.hearAgainUsed || state === S.INSTRUCTIONS;

  const fb = $("feedback");
  show(fb, state === S.FEEDBACK && feedback !== null);
  if (feedback !== null) {
    fb.dataset.correct = String(feedback);
    fb.replaceChildren(icon(feedback), document.createTextNode(t(feedback ? "ui.test.feedback_correct" : "ui.test.feedback_wrong")));
  }

  if (state === S.RESULT || state === S.DONE) {
    renderResult($("result-view"), { result, explanation, explanationPending: state === S.RESULT && explanation === null });
  }
  $("end-title").textContent = t({ INTERRUPTED: "ui.test.interrupted", EXPIRED: "ui.test.session_expired" }[state] || "ui.error.generic");
  renderDots();
  if (state === S.LISTEN || state === S.ANSWER || state === S.INSTRUCTIONS) announce($("phase-title").textContent);
}

function icon(ok) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", ok ? "M5 12.5l4.5 4.5L19 7.5" : "M7 7l10 10M17 7L7 17");
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", "currentColor");
  p.setAttribute("stroke-width", "2.4");
  p.setAttribute("stroke-linecap", "round");
  svg.append(p);
  return svg;
}

function banner(key) {
  const host = $("banners");
  if (host.querySelector(`[data-key="${key}"]`)) return;
  const b = document.createElement("div");
  b.className = "banner";
  b.dataset.key = key;
  const dot = document.createElement("span");
  dot.className = "notice-dot";
  dot.setAttribute("aria-hidden", "true");
  const p = document.createElement("p");
  p.textContent = t(key);
  b.append(dot, p);
  host.append(b);
}

// ---------- actions ----------

function press(d) {
  if (!client) return;
  const btn = $("keypad").querySelector(`[data-digit="${d}"]`);
  if (btn && !btn.disabled) { btn.classList.add("is-pressed"); setTimeout(() => btn.classList.remove("is-pressed"), 120); }
  client.press(d);
  renderDots();
  render();
}

function erase() {
  if (!client) return;
  client.del();
  renderDots();
  render();
}

function confirmAnswer() {
  if (!client || !client.confirm()) return;
  renderDots();
  render();
}

function onUi(k) {
  if (k.type === "phase") { phase = k; feedback = null; }
  if (k.type === "consent") $("consent-text").textContent = k.text;
  if (k.type === "feedback") feedback = k.correct;
  if (k.type === "result") {
    result = k;
    sessionStorage.setItem(RESULT_KEY, JSON.stringify({ result, explanation: null, lang: lang() }));
    announce(k.page_text);
  }
  if (k.type === "explanation") {
    explanation = k.text;
    sessionStorage.setItem(RESULT_KEY, JSON.stringify({ result, explanation, lang: lang() }));
  }
  if (k.type === "mic" && k.start) startMic();
  if (k.type === "mode" && k.mode === "keypad") { mode = "keypad"; banner(denied ? "ui.test.mic_denied_keypad" : "ui.test.voice_switched_keypad"); }
  if (k.type === "notice") banner("ui.test.speaker_echo_warning");
  render();
}

async function startMic() {
  if (!ctx || mode !== "voice") return;
  mic = new Mic(ctx, {
    earphones: start.earphones,
    hooks: {
      onFrame: (i, ts, bytes) => client.media(i, ts, bytes),
      onSettings: (s) => client.micSettings(s),
      onDenied: () => { denied = true; mode = "keypad"; client.micDenied(); banner("ui.test.mic_denied_keypad"); render(); },
      onLost: () => client.micLost(),
    },
  });
  client.o.mic = mic;
  const ok = await mic.start();
  for (const el of document.querySelectorAll(".mic-state")) el.dataset.on = String(ok);
}

function connect(body) {
  client = new SessionClient({
    wsUrl: body.ws_url, sessionId: body.session_id, token: body.ws_token,
    WebSocket: window.WebSocket, player: new Player(ctx),
    onState: (s) => { state = s; render(); },
    onUi, onBanner: banner,
    onAutoSubmit: () => { renderDots(); render(); },
  });
  client.open();
}

async function admit() {
  go({ type: "unlocked" });
  const req = { lang: start.lang, mode, earphones: start.earphones, client: api.clientHello(ctx) };
  let res;
  try { res = await api.createSession(req); } catch { res = { status: 0, body: null }; }
  go({ type: "admit", status: res.status, body: res.body });
  if (state === S.CONNECTING) connect(res.body);
  else if (state === S.WAITING_ROOM) {
    const ahead = res.body && res.body.details && res.body.details.ahead;
    $("wait-ahead").textContent = ahead ? t("ui.wait.ahead", { n: ahead }) : "";
    setTimeout(admit, WAIT_RETRY_MS);
  } else if (state === S.QUEUED) {
    const b = res.body;
    const min = b.eta_s ? Math.max(1, Math.round(b.eta_s / 60)) : null;
    $("queue-detail").textContent = t("ui.queue.detail", { pos: b.queue_position, min: min ?? "?" });
    pollQueue(b);
  }
}

async function pollQueue(b) {
  if (state !== S.QUEUED) return;
  const r = await api.sessionStatus(b.session_id, b.ws_token).catch(() => ({ status: 0, body: null }));
  if (r.status === 200 && r.body.status === "ADMITTED") { go({ type: "queue_admitted" }); connect(b); return; }
  if (r.status === 200 && r.body.queue_position) {
    const min = r.body.eta_s ? Math.max(1, Math.round(r.body.eta_s / 60)) : "?";
    $("queue-detail").textContent = t("ui.queue.detail", { pos: r.body.queue_position, min });
  }
  setTimeout(() => pollQueue(b), QUEUE_POLL_MS);
}

function keypadInstead() {
  mode = "keypad";
  state = S.LANDING;
  go({ type: "start" });
  admit();
}

function watchInterruptions() {
  const check = () => {
    if (!LIVE_STATES.has(state) || !ctx) return;
    ctx.resume();
    setTimeout(() => { if (ctx.state !== "running" && client) client.interrupt(); }, 1500);
  };
  ctx.onstatechange = () => { if (ctx.state !== "running") check(); };
  document.addEventListener("visibilitychange", () => { if (document.hidden && (api.isIOS() || /Android/.test(navigator.userAgent))) check(); });
  window.addEventListener("offline", () => { if (client) client.interrupt(); });
  window.addEventListener("keydown", (e) => {
    if (e.target && e.target.closest && e.target.closest("dialog")) return;
    if (/^[1-9]$/.test(e.key) && state === S.ANSWER) press(e.key);
    else if (e.key === "Backspace" && state === S.ANSWER) { e.preventDefault(); erase(); }
    else if (e.key === "Enter" && state === S.ANSWER) { e.preventDefault(); confirmAnswer(); }
    else if (e.key === "*" && client) { client.hearAgain(); render(); }
    else if (e.key === "0" && client && !$("repeat").hidden) client.repeatInstructions();
  });
}

async function boot() {
  await initChrome(() => { labelKeypad(); render(); });
  // The test's language is fixed when the session is created on the landing page; switching it
  // here would only relabel buttons while prompts stay in the session language.
  show($("lang-switch"), false);
  if (!start) { location.replace("index.html"); return; }
  buildKeypad();
  $("agree").addEventListener("click", () => client && client.consent(true));
  $("decline").addEventListener("click", () => client && client.consent(false));
  $("hear-again").addEventListener("click", () => { if (client) { client.hearAgain(); render(); } });
  $("repeat").addEventListener("click", () => client && client.repeatInstructions());
  $("take-keypad").addEventListener("click", keypadInstead);
  $("fallback-keypad").addEventListener("click", keypadInstead);
  const dlg = $("cant-hear-dialog");
  $("cant-hear").addEventListener("click", () => dlg.showModal());
  $("cant-hear-no").addEventListener("click", () => dlg.close());
  $("cant-hear-yes").addEventListener("click", () => { dlg.close(); if (client) client.cantHear(); });

  try { await api.loadConfig(); } catch { go({ type: "boot_down" }); render(); return; }
  go({ type: "boot_ok" });
  go({ type: "start" });
  ctx = unlock({ voice: mode === "voice" });
  watchInterruptions();
  $("tap-audio-btn").addEventListener("click", () => { ctx.resume(); show($("tap-audio"), false); });
  setTimeout(() => show($("tap-audio"), ctx.state !== "running"), 1000);
  if (mode === "voice") {
    try { await loadCaptureWorklet(ctx); } catch { mode = "keypad"; }
  }
  admit();
}

boot();
