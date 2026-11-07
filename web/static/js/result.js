// result.html: re-render the last result from sessionStorage, else `ui.result.not_on_device`.
import { t } from "./copy.js";
import { RESULT_KEY, renderResult } from "./result-view.js";
import { initChrome } from "./ui.js";

function render() {
  const root = document.getElementById("result-view");
  const raw = sessionStorage.getItem(RESULT_KEY);
  if (!raw) {
    const p = document.createElement("p");
    p.className = "result-text";
    p.textContent = t("ui.result.not_on_device");
    const a = document.createElement("a");
    a.className = "btn btn-primary";
    a.href = "index.html";
    a.textContent = t("ui.test.start_again");
    const actions = document.createElement("div");
    actions.className = "actions";
    actions.append(a);
    root.replaceChildren(p, actions);
    return;
  }
  const s = JSON.parse(raw);
  renderResult(root, { result: s.result, explanation: s.explanation, explanationPending: false });
}

initChrome(render).then(render);
