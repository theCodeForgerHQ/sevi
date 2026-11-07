// Renders a UiResult (+ optional explanation) into a container; shared by test.html and result.html.
import { t } from "./copy.js";

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

/**
 * @param {HTMLElement} root
 * @param {{result: object, explanation: string|null, explanationPending: boolean}} s
 */
export function renderResult(root, s) {
  const r = s.result;
  root.replaceChildren();
  root.append(el("h1", "result-title", t("ui.result.title")));
  const text = r.page_text === "" ? t("ui.test.session_expired") : r.page_text;
  root.append(el("p", "result-text", text));
  if (r.srt_display) root.append(el("span", "srt", `SRT ${r.srt_display}`));

  // Evon explanation slot: skeleton while pending, text when it arrives, hidden if none.
  if (s.explanation || s.explanationPending) {
    const sec = el("section", "section");
    sec.setAttribute("aria-live", "polite");
    sec.append(el("h2", null, t("ui.result.explanation_label")));
    if (s.explanation) sec.append(el("p", null, s.explanation));
    else { sec.setAttribute("aria-busy", "true"); sec.append(el("div", "skeleton"), el("div", "skeleton")); }
    root.append(sec);
  }

  const more = el("section", "section");
  if (r.report_url) {
    const a = el("a", null, t("ui.result.report_link"));
    a.href = r.report_url;
    a.rel = "noopener";
    more.append(a);
  }
  more.append(el("p", "meta", t("ui.result.expiry_note")));
  root.append(more);

  const actions = el("div", "actions");
  const share = el("button", "btn btn-primary", t("ui.result.share"));
  share.type = "button";
  share.addEventListener("click", async () => {
    const url = new URL("index.html", location.href).href;
    const data = { text: t("share.card"), url };
    try {
      if (navigator.share) await navigator.share(data);
      else { await navigator.clipboard.writeText(`${data.text} ${url}`); share.textContent = t("ui.result.copied"); }
    } catch { /* user cancelled */ }
  });
  const again = el("a", "btn btn-secondary", t("ui.test.start_again"));
  again.href = "index.html";
  actions.append(share, again);
  root.append(actions);
}

export const RESULT_KEY = "sevi.result";
