// Page init script for scripts/browser_e2e.py (see its docstring). `window.__SEVI_CFG` is
// prepended by the harness: {mode, digits (base64 WAV per digit), digitSet, gapMs, gainDb,
// wrongAfter, verbose}.
(() => {
  const CFG = window.__SEVI_CFG;
  const S = (window.__sevi = {
    t0: Date.now(), plan: null, sessionId: null, presentations: 0, cur: null,
    events: [], answers: [], fallback: false, result: null, explanation: null,
    mediaOut: 0, mediaIn: 0, feedback: [], closed: null, errors: [],
  });
  const now = () => Date.now() - S.t0;
  const log = (kind, data) => { S.events.push({ t: now(), kind, data }); if (CFG.verbose) console.log("[sevi-e2e] " + kind + " " + JSON.stringify(data)); };

  // ---- session id -> plan (Python computes the triplet order) ----
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const res = await nativeFetch(input, init);
    const url = typeof input === "string" ? input : input.url;
    if (/\/api\/v1\/sessions$/.test(url) && init && init.method === "POST") {
      try {
        const body = await res.clone().json();
        if (body && body.session_id) {
          S.sessionId = body.session_id;
          S.plan = await window.__seviPlan(body.session_id);
          log("plan", { session_id: body.session_id, n: S.plan.length });
        }
      } catch (e) { S.errors.push("plan: " + e); }
    }
    return res;
  };

  // ---- injected microphone ----
  let actx = null, dest = null;
  const clips = {};
  function audio() {
    if (!actx) {
      actx = new AudioContext({ sampleRate: 48000 });
      dest = actx.createMediaStreamDestination();
    }
    return actx;
  }
  async function clip(d) {
    if (!clips[d]) {
      const bin = Uint8Array.from(atob(CFG.digits[d]), (c) => c.charCodeAt(0));
      clips[d] = await audio().decodeAudioData(bin.buffer);
    }
    return clips[d];
  }
  if (CFG.mode === "voice" && navigator.mediaDevices) {
    navigator.mediaDevices.getUserMedia = async (c) => {
      if (!c || !c.audio) throw new DOMException("video not offered", "NotAllowedError");
      await audio().resume();
      log("gum", {});
      return dest.stream.clone();
    };
  }
  async function speak(digits) {
    const ctx = audio();
    await ctx.resume();
    const gain = ctx.createGain();
    gain.gain.value = Math.pow(10, CFG.gainDb / 20);
    gain.connect(dest);
    let t = ctx.currentTime + 0.05;
    for (const d of digits) {
      const buf = await clip(d);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      src.start(t);
      t += buf.duration + CFG.gapMs / 1000;
    }
  }

  // ---- answering ----
  const wrongOf = (trip) => trip.map((d) => { const ds = CFG.digitSet; return ds[(ds.indexOf(d) + 1) % ds.length]; });
  function answerFor(cur) {
    const ok = cur.triplet;
    const wrong = CFG.wrongAfter !== null && cur.kind === "SCORED" && cur.scored_index > CFG.wrongAfter;
    return { digits: wrong ? wrongOf(ok) : ok, intendedCorrect: !wrong };
  }
  function click(sel) { const el = document.querySelector(sel); if (el && !el.disabled) { el.click(); return true; } return false; }
  async function pressKeys(digits) {
    for (const d of digits) {
      for (let i = 0; i < 40 && !click(`#keypad [data-digit="${d}"]`); i++) await new Promise((r) => setTimeout(r, 25));
      await new Promise((r) => setTimeout(r, 60));
    }
  }
  async function onWindowOpen() {
    const cur = S.cur;
    if (!cur) { S.errors.push("window opened before any presentation"); return; }
    if (!cur.triplet) { S.errors.push("no plan for presentation " + cur.n); return; }
    cur.windows += 1;
    const a = answerFor(cur);
    const by = CFG.mode === "voice" && !S.fallback ? "voice" : "keypad";
    S.answers.push({ n: cur.n, kind: cur.kind, scored_index: cur.scored_index, window: cur.windows, by, digits: a.digits, intended_correct: a.intendedCorrect, t: now() });
    log("answer", { n: cur.n, by, digits: a.digits.join("") });
    if (by === "voice") await speak(a.digits);
    else { await new Promise((r) => setTimeout(r, 40)); await pressKeys(a.digits); }
  }

  function onKj(k) {
    log("ui", k);
    switch (k.type) {
      case "consent": setTimeout(() => click("#agree"), 200); break;
      case "phase":
        if (k.state === "TEST") {
          S.presentations += 1;
          const n = S.presentations;
          const triplet = S.plan && S.plan[n - 1] ? S.plan[n - 1] : null;
          S.cur = { n, kind: k.kind, scored_index: k.scored_index, triplet, windows: 0, t: now() };
        }
        break;
      case "keypad": if (k.enabled) onWindowOpen().catch((e) => S.errors.push("answer: " + e)); break;
      case "feedback": S.feedback.push({ n: S.cur && S.cur.n, correct: k.correct }); break;
      case "mode": if (k.mode === "keypad") S.fallback = true; break;
      case "result": S.result = k; break;
      case "explanation": S.explanation = k.text; break;
      case "error": S.errors.push("server error: " + JSON.stringify(k)); break;
      default: break;
    }
  }

  // ---- WebSocket tap ----
  const NativeWS = window.WebSocket;
  function TapWS(url, protocols) {
    const ws = protocols === undefined ? new NativeWS(url) : new NativeWS(url, protocols);
    ws.addEventListener("message", (e) => {
      let m; try { m = JSON.parse(e.data); } catch { return; }
      if (m.event === "media") S.mediaIn += 1;
      else if (m.event === "kj" && m.kj) onKj(m.kj);
    });
    ws.addEventListener("close", (e) => { S.closed = e.code; log("close", { code: e.code }); });
    const send = ws.send.bind(ws);
    ws.send = (d) => {
      try { const m = JSON.parse(d); if (m.event === "media") S.mediaOut += 1; else if (m.event !== "mark") log("send", m); } catch { /* binary */ }
      return send(d);
    };
    return ws;
  }
  TapWS.prototype = NativeWS.prototype;
  for (const k of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) TapWS[k] = NativeWS[k];
  window.WebSocket = TapWS;
})();
