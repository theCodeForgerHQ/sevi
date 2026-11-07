// r.html: family report. Token comes from the URL fragment and is removed from history at once.
import * as api from "./api.js";
import { t } from "./copy.js";
import { initChrome, show } from "./ui.js";

const $ = (id) => document.getElementById(id);
const token = location.hash.slice(1);
if (token) history.replaceState(null, "", location.pathname + location.search);
let data = null;

function ist(iso) {
  return new Intl.DateTimeFormat(document.documentElement.lang === "ta" ? "ta-IN" : "en-IN", {
    dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Kolkata",
  }).format(new Date(iso));
}

function render() {
  if (!data) return;
  $("report-meta").textContent = t("ui.report.expires", { when: ist(data.expires_at) });
  $("report-text").textContent = data.page_text;
  show($("report-srt"), !!data.srt_display);
  $("report-srt").textContent = data.srt_display ? `SRT ${data.srt_display}` : "";
  show($("report-evon"), !!data.evon_text);
  $("report-evon-text").textContent = data.evon_text || "";
  $("report-disclaimer").textContent = data.disclaimer;
  const rows = data.trials.map((tr) => {
    const row = document.createElement("tr");
    for (const v of [tr.scored_index, tr.snr_db.toFixed(1), t(tr.correct ? "ui.report.yes" : "ui.report.no")]) {
      const td = document.createElement("td");
      td.textContent = String(v);
      row.append(td);
    }
    return row;
  });
  $("trials").replaceChildren(...rows);
}

async function boot() {
  await initChrome(render);
  $("print").addEventListener("click", () => window.print());
  let res = { status: 401, body: null };
  if (token) {
    try { await api.loadConfig(); res = await api.report(token); } catch { res = { status: 0, body: null }; }
  }
  show($("report-loading"), false);
  if (res.status === 200) { data = res.body; show($("report"), true); render(); return; }
  if (res.status === 410 || res.status === 404 || res.status === 401) {
    show($("report-expired"), true);
    return;
  }
  $("report-expired").textContent = t("ui.error.generic");
  show($("report-expired"), true);
}

boot();
