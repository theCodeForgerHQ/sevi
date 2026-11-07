// Boot and HTTP helpers (B spec §10.2): config.json, /readyz, /api/v1/status, sessions, report.

let config = null;

export async function loadConfig() {
  const r = await fetch(new URL(`../config.json?ts=${Date.now()}`, import.meta.url), { cache: "no-store" });
  config = await r.json();
  return config;
}

export function apiBase() { return config ? config.api_base.replace(/\/$/, "") : ""; }

async function timed(url, init = {}, ms = 5000) {
  const ac = new AbortController();
  const h = setTimeout(() => ac.abort(), ms);
  try { return await fetch(url, { ...init, signal: ac.signal }); } finally { clearTimeout(h); }
}

async function readyOnce() {
  try {
    const r = await timed(`${apiBase()}/readyz`);
    if (!r.ok) return false;
    const b = await r.json();
    return b.status === "ok" || b.status === "degraded";
  } catch { return false; }
}

/** true when the backend is up; on failure re-fetch config (ingress may have moved) and retry once after 3 s. */
export async function ready() {
  if (await readyOnce()) return true;
  await new Promise((r) => setTimeout(r, 3000));
  try { await loadConfig(); } catch { return false; }
  return readyOnce();
}

export async function status() {
  const r = await timed(`${apiBase()}/api/v1/status`);
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

/** POST /api/v1/sessions. Never retried here (R-56). Returns {status, body}. */
export async function createSession(req) {
  const r = await timed(`${apiBase()}/api/v1/sessions`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req),
  }, 10000);
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  return { status: r.status, body };
}

export async function sessionStatus(sessionId, token) {
  const r = await timed(`${apiBase()}/api/v1/sessions/${sessionId}`, { headers: { Authorization: `Bearer ${token}` } });
  return { status: r.status, body: r.ok ? await r.json() : null };
}

export async function report(token) {
  const r = await timed(`${apiBase()}/api/v1/report`, { headers: { Authorization: `Bearer ${token}` } });
  return { status: r.status, body: r.ok ? await r.json() : null };
}

/** ClientHello (contracts §9.2). */
export function clientHello(ctx, pageVersion = "dev") {
  const ua = navigator.userAgent.slice(0, 300);
  const browser = /Firefox\//.test(ua) ? "firefox" : /Edg\//.test(ua) ? "edge" : /Chrome\//.test(ua) ? "chrome" : /Safari\//.test(ua) ? "safari" : "other";
  const os = /iPhone|iPad/.test(ua) ? "ios" : /Android/.test(ua) ? "android" : /Mac OS X/.test(ua) ? "macos" : /Windows/.test(ua) ? "windows" : "other";
  return { browser_family: browser, os_family: os, ua, audio_worklet: "AudioWorkletNode" in window, sample_rate_hz: Math.round(ctx.sampleRate), page_version: pageVersion };
}

export function isIOS() { return /iPhone|iPad|iPod/.test(navigator.userAgent); }

export function features() {
  const keypad = "AudioContext" in window && "WebSocket" in window && "TextDecoder" in window && !!(window.crypto && crypto.getRandomValues);
  const voice = keypad && "AudioWorkletNode" in window && !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  return { keypad, voice };
}
