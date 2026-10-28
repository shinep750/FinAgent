"use strict";

/* ---------------- helpers ---------------- */
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
const nl2br = (s) => esc(s).replace(/\n/g, "<br>");

function fmtNum(v) {
  const n = Number(v);
  if (v === null || v === undefined || Number.isNaN(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (a >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return (Math.round(n * 100) / 100).toString();
}
const pct = (v) => Math.round(Number(v || 0) * 100) + "%";
const dateStr = (d) => {
  if (!d) return "";
  const dt = new Date(d);
  return Number.isNaN(dt.getTime()) ? "" : dt.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};
const human = (s) => String(s || "").replace(/_/g, " ");
const bandOf = (v) => (v >= 0.75 ? "high" : v >= 0.5 ? "moderate" : v >= 0.3 ? "low" : "very_low");

const STAGES = [
  ["plan", "Plan"],
  ["retrieve", "Gather"],
  ["analyse", "Analyse"],
  ["verify", "Verify"],
  ["synthesise", "Synthesise"],
];

const SEND_ICON =
  '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 19V5M6 11l6-6 6 6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
const STOP_ICON =
  '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/></svg>';

/* ---------------- theme ---------------- */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  $("themeLabel").textContent = theme === "dark" ? "Light mode" : "Dark mode";
  $("themeIcon").style.transform = theme === "dark" ? "rotate(180deg)" : "none";
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", theme === "dark" ? "#05070d" : "#eef2fb");
  localStorage.setItem("fa-theme-2", theme);
}
function initTheme() {
  const saved = localStorage.getItem("fa-theme-2");
  applyTheme(saved || "dark");
}

/* ---------------- sidebar ---------------- */
function initSidebar() {
  const app = $("app");
  if (localStorage.getItem("fa-collapsed") === "1" && window.innerWidth > 900) {
    app.classList.add("collapsed");
  }
  $("menuBtn").addEventListener("click", () => {
    if (window.innerWidth <= 900) {
      app.classList.toggle("mobile-open");
    } else {
      app.classList.toggle("collapsed");
      localStorage.setItem("fa-collapsed", app.classList.contains("collapsed") ? "1" : "0");
    }
  });
  $("scrim").addEventListener("click", () => app.classList.remove("mobile-open"));
  $("themeBtn").addEventListener("click", () => {
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
}

/* ---------------- history ---------------- */
function getHistory() {
  try {
    return JSON.parse(localStorage.getItem("fa-history") || "[]");
  } catch {
    return [];
  }
}
function saveHistory(items) {
  localStorage.setItem("fa-history", JSON.stringify(items.slice(0, 12)));
  renderHistory();
}
function pushHistory(entry) {
  const items = getHistory().filter((h) => !(h.q === entry.q && h.ticker === entry.ticker));
  items.unshift(entry);
  saveHistory(items);
}
function renderHistory() {
  const list = $("historyList");
  const items = getHistory();
  list.innerHTML = "";
  $("historyEmpty").hidden = items.length > 0;
  items.forEach((h) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "history-item";
    btn.type = "button";
    btn.title = h.q;
    btn.textContent = (h.ticker ? h.ticker + " · " : "") + h.q;
    btn.addEventListener("click", () => {
      if (busy) return;
      clearChatContext();
      $("question").value = h.q;
      $("ticker").value = h.ticker || "";
      setPill("verifyPill", h.noVerify !== true);
      setPill("seqPill", h.sequential !== false);
      $("app").classList.remove("mobile-open");
      submit();
    });
    li.appendChild(btn);
    list.appendChild(li);
  });
}

/* ---------------- composer ---------------- */
function setPill(id, on) {
  const el = $(id);
  el.classList.toggle("on", !!on);
  el.setAttribute("aria-pressed", on ? "true" : "false");
}
const pillOn = (id) => $(id).getAttribute("aria-pressed") === "true";

function autoGrow() {
  const t = $("question");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 200) + "px";
}

/* ---------------- state views ---------------- */
function showOnly(which) {
  $("emptyState").hidden = which !== "empty";
  $("runState").hidden = which !== "run";
  $("reportState").hidden = which !== "report";
}

function setServer(ok) {
  const dot = $("serverDot");
  dot.classList.toggle("ok", !!ok);
  dot.classList.toggle("bad", !ok);
  $("serverText").textContent = ok ? "Connected" : "Offline";
}

/* ---------------- run + polling ---------------- */
let runTimer = null;
let runStart = 0;
let activeJob = null;
let busy = false;
let stopping = false;
let pendingCancel = false;
let pollGen = 0;
// Successful turns in the current chat. The next question sends this to the
// server, which reuses the subject and only runs the lenses the follow-up
// needs. "New research", history clicks and suggestions start a new chat.
let chatTurns = [];

function clearChatContext() {
  chatTurns = [];
  const step = $("runStep");
  if (step) {
    step.hidden = true;
    step.textContent = "";
  }
}

function setBusy(on) {
  busy = on;
  const btn = $("sendBtn");
  if (on) {
    stopping = false;
    btn.innerHTML = STOP_ICON;
    btn.type = "button";
    btn.classList.add("is-stop");
    btn.classList.remove("stopping");
    btn.disabled = false;
    btn.setAttribute("aria-label", "Stop research");
    btn.title = "Stop research";
  } else {
    stopping = false;
    pendingCancel = false;
    btn.innerHTML = SEND_ICON;
    btn.type = "submit";
    btn.classList.remove("is-stop", "stopping");
    btn.disabled = false;
    btn.setAttribute("aria-label", "Analyse");
    btn.title = "Analyse";
  }
}

function startTimer() {
  runStart = Date.now();
  clearInterval(runTimer);
  const tick = () => {
    const s = Math.round((Date.now() - runStart) / 1000);
    $("runTimer").textContent = s < 60 ? s + "s" : Math.floor(s / 60) + "m " + (s % 60) + "s";
  };
  tick();
  runTimer = setInterval(tick, 1000);
}

function renderStepper(stage, done) {
  let idx = STAGES.findIndex((s) => s[0] === stage);
  if (stage === "queued") idx = -1;
  if (done || stage === "report") idx = STAGES.length;

  const stepper = $("stepper");
  stepper.innerHTML = STAGES.map(([key, label], i) => {
    const cls = done || i < idx ? "done" : i === idx ? "active" : "";
    const mark = done || i < idx ? "✓" : i + 1;
    return (
      `<div class="step ${cls}">` +
      `<span class="rail"></span>` +
      `<span class="bullet">${mark}</span>` +
      `<span class="name">${label}</span>` +
      `</div>`
    );
  }).join("");

  const frac = done || stage === "report" ? 1 : Math.max(0, (idx + 0.5) / STAGES.length);
  $("progressBar").style.width = Math.max(4, Math.round(frac * 100)) + "%";

  const runStep = $("runStep");
  if (!done && idx >= 0 && idx < STAGES.length) {
    runStep.hidden = false;
    runStep.textContent = `Stage ${idx + 1} of ${STAGES.length}`;
  } else {
    runStep.hidden = true;
  }
}

function stageText(stage) {
  if (stage === "queued") return "Queued — waiting for a free slot…";
  if (stage === "plan")
    return "Planning the research and resolving tickers — the first model call is the slowest.";
  if (stage === "retrieve") return "Retrieving SEC filings, news and market data…";
  if (stage === "analyse") return "Specialist agents are reading the evidence…";
  if (stage === "verify") return "Verifying every claim against its cited sources…";
  if (stage === "synthesise") return "Synthesising the final report…";
  if (stage === "report") return "Finishing up…";
  return "Working…";
}

async function submit() {
  if (busy) return;
  const question = $("question").value.trim();
  if (!question) {
    $("question").focus();
    return;
  }
  const ticker = $("ticker").value.trim().toUpperCase() || null;
  const payload = {
    question,
    ticker,
    no_verify: !pillOn("verifyPill"),
    sequential: pillOn("seqPill"),
  };
  if (chatTurns.length) payload.history = chatTurns.slice(-3);

  const gen = ++pollGen;
  setBusy(true);
  $("question").value = "";
  autoGrow();
  showOnly("run");
  $("runQuery").textContent = question;
  const priorTickers = chatTurns.length
    ? (chatTurns[chatTurns.length - 1].tickers || [])
    : [];
  $("runStep").hidden = !priorTickers.length;
  $("runStep").textContent = priorTickers.length ? "Follow-up · " + priorTickers.join(", ") : "";
  renderStepper("queued", false);
  $("runStage").textContent = stageText("queued");
  startTimer();

  let job;
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    job = await res.json();
  } catch (err) {
    $("question").value = question;
    autoGrow();
    setBusy(false);
    setServer(false);
    return showError({ detail: "Could not reach the server: " + err.message });
  }

  activeJob = job.job_id;
  if (gen !== pollGen) {
    fetch("/jobs/" + activeJob + "/cancel", { method: "POST" }).catch(() => {});
    activeJob = null;
    return;
  }
  pushHistory({ q: question, ticker, noVerify: payload.no_verify, sequential: payload.sequential, at: Date.now() });
  if (pendingCancel) {
    pendingCancel = false;
    cancelRun();
  }
  poll(activeJob, gen);
}

async function poll(id, gen) {
  while (true) {
    await sleep(1500);
    if (gen !== pollGen) return;
    let job;
    try {
      const res = await fetch("/jobs/" + id);
      if (res.status === 404) {
        setServer(false);
        return showError({ detail: "The server no longer knows this run (it may have restarted)." });
      }
      if (!res.ok) throw new Error("HTTP " + res.status);
      job = await res.json();
      setServer(true);
    } catch {
      setServer(false);
      $("runStage").textContent = "Reconnecting…";
      continue;
    }

    if (job.state === "done") return finish(job);
    if (job.state === "error") return showError(job);
    if (job.state === "cancelled") return showStopped(job);

    renderStepper(job.stage || "queued", false);
    $("runStage").textContent = stopping
      ? "Stopping after the current stage finishes…"
      : stageText(job.stage);
  }
}

async function cancelRun() {
  if (!busy || stopping) return;
  if (!activeJob) {
    pendingCancel = true;
    $("runStage").textContent = "Stopping as soon as the run starts…";
    return;
  }
  stopping = true;
  const btn = $("sendBtn");
  btn.disabled = true;
  btn.classList.add("stopping");
  $("runStage").textContent = "Stopping after the current stage finishes…";
  try {
    const res = await fetch("/jobs/" + activeJob + "/cancel", { method: "POST" });
    if (!res.ok) throw new Error("HTTP " + res.status);
  } catch {
    stopping = false;
    btn.disabled = false;
    btn.classList.remove("stopping");
    $("runStage").textContent = "Could not reach the server to stop — the run is still going.";
  }
}

function showStopped(job) {
  pollGen++;
  stopTimer();
  setBusy(false);
  activeJob = null;
  const question = job.question || $("runQuery").textContent;
  $("reportState").innerHTML =
    `<div class="report-top"><div class="report-kicker">Stopped</div>` +
    `<p class="report-q">${esc(question)}</p>` +
    `<div class="notice">Research stopped. No report was produced.</div></div>`;
  showOnly("report");
  $("viewport").scrollTop = 0;
}

function stopTimer() {
  clearInterval(runTimer);
  runTimer = null;
}

function finish(job) {
  stopTimer();
  setBusy(false);
  const report = job.report;
  if (!report) {
    const errs = (job.errors || []).map((e) => `${e.agent_name}: ${e.error}`);
    return showError({ detail: "The run finished without a report.", errors: job.errors, list: errs });
  }
  const tickers = job.tickers || (report.query && report.query.tickers) || [];
  chatTurns.push({
    question: job.question || $("runQuery").textContent,
    tickers,
    perspectives: job.perspectives || [],
    thesis: report.thesis || "",
  });
  chatTurns = chatTurns.slice(-3);
  renderStepper("report", true);
  $("reportState").innerHTML = reportHTML(report, job.text);
  showOnly("report");
  $("viewport").scrollTop = 0;
  wireReportActions(report);
}

function showError(job) {
  stopTimer();
  setBusy(false);
  const lines = job.list || [];
  const outOfScope = job.status === "out_of_scope";
  const detail = esc(job.detail || job.status || "The run failed.");
  $("reportState").innerHTML =
    `<div class="report-top"><div class="report-kicker">${outOfScope ? "Out of scope" : "Run failed"}</div>` +
    `<p class="report-q">${outOfScope ? "This question is outside our research area" : "Something went wrong"}</p>` +
    `<div class="notice${outOfScope ? "" : " bad"}">${detail}</div>` +
    (lines.length ? `<ul class="list limits" style="margin-top:14px">${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : "") +
    `</div>`;
  showOnly("report");
}

/* ---------------- report rendering ---------------- */
function badge(text, cls) {
  return `<span class="badge ${esc(cls || "")}">${esc(human(text))}</span>`;
}

function ringHTML(overall) {
  const r = 40;
  const circ = 2 * Math.PI * r;
  const pctv = Math.max(0, Math.min(1, Number(overall || 0)));
  const dash = circ * pctv;
  return (
    `<div class="ring">` +
    `<svg width="92" height="92" viewBox="0 0 92 92">` +
    `<circle cx="46" cy="46" r="${r}" fill="none" stroke="var(--surface-3)" stroke-width="8"/>` +
    `<circle cx="46" cy="46" r="${r}" fill="none" stroke="url(#grad)" stroke-width="8" ` +
    `stroke-linecap="round" stroke-dasharray="${dash.toFixed(1)} ${(circ - dash).toFixed(1)}"/>` +
    `<defs><linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">` +
    `<stop offset="0" stop-color="var(--accent)"/><stop offset="1" stop-color="var(--accent-2)"/>` +
    `</linearGradient></defs></svg>` +
    `<div class="ring-label">${Math.round(pctv * 100)}%<small>confidence</small></div>` +
    `</div>`
  );
}

function confBar(label, v) {
  const value = Math.max(0, Math.min(1, Number(v || 0)));
  return (
    `<div class="conf-line"><span>${esc(label)}</span>` +
    `<div class="bar"><i style="width:${Math.round(value * 100)}%"></i></div>` +
    `<span class="val">${Math.round(value * 100)}%</span></div>`
  );
}

function claimTextMap(report) {
  const map = {};
  (report.findings || []).forEach((f) =>
    (f.claims || []).forEach((c) => {
      map[c.claim_id] = c.text;
    })
  );
  return map;
}

function sectionsHTML(report) {
  if (!report.sections || !report.sections.length) return "";
  return report.sections
    .map((s) => {
      const cites = (s.citations || [])
        .map(
          (c) =>
            `<div class="cite"><span class="cite-mark">❝</span><div>` +
            `<div class="src">${esc(c.title)} · ${esc(c.source)}</div>` +
            (c.quote ? `<blockquote>${esc(c.quote)}</blockquote>` : "") +
            `</div></div>`
        )
        .join("");
      return (
        `<div class="card section">` +
        `<div class="section-head">` +
        `<h3>${esc(s.heading)}</h3>` +
        badge(s.perspective, s.perspective) +
        `<span class="spacer"></span>` +
        (s.stance ? badge(s.stance, s.stance) : "") +
        (s.confidence ? `<span class="meta-chip">${pct(s.confidence)} confidence</span>` : "") +
        `</div>` +
        `<div class="section-body">${nl2br(s.body)}</div>` +
        (cites ? `<div class="citations">${cites}</div>` : "") +
        `</div>`
      );
    })
    .join("");
}

function metricsHTML(report) {
  const entries = Object.entries(report.key_metrics || {});
  if (!entries.length) return "";
  const stats = entries
    .map(([k, v]) => `<div class="stat"><div class="k">${esc(human(k))}</div><div class="v">${esc(fmtNum(v))}</div></div>`)
    .join("");
  return `<div class="card"><p class="card-title">Key metrics</p><div class="metrics">${stats}</div></div>`;
}

function risksHTML(report) {
  if (!report.risks || !report.risks.length) return "";
  return (
    `<div class="card"><p class="card-title">Risks</p>` +
    `<ul class="list risks">${report.risks.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`
  );
}

function claimsHTML(report) {
  const claims = report.verified_claims || [];
  if (!claims.length) return "";
  const texts = claimTextMap(report);
  const rows = claims
    .map((v) => {
      const text = texts[v.claim_id] || v.claim_id;
      const support = Math.round(Number(v.support_score || 0) * 100);
      return (
        `<div class="claim"><div class="claim-main">` +
        `<div class="claim-text">${esc(text)}</div>` +
        (v.rationale ? `<div class="claim-rationale">${esc(v.rationale)}</div>` : "") +
        `<div class="score">${badge(v.verdict, v.verdict)}` +
        `<span class="bar"><i style="width:${support}%"></i></span>` +
        `<span>${support}% support</span></div>` +
        `</div></div>`
      );
    })
    .join("");
  return `<div class="card"><p class="card-title">Verified claims · ${claims.length}</p>${rows}</div>`;
}

function conflictsHTML(report) {
  const items = report.conflicts || [];
  if (!items.length) return "";
  const rows = items
    .map(
      (c) =>
        `<div class="ev-item"><span class="ev-type">${esc(human(c.conflict_type))}</span>` +
        `<div class="ev-body"><div class="ev-title">${esc(c.description)}</div>` +
        `<div class="ev-meta">${badge(c.severity, "sev-" + c.severity)} ${c.resolved ? "resolved" : "unresolved"}` +
        (c.resolution ? ` · ${esc(c.resolution)}` : "") +
        `</div></div></div>`
    )
    .join("");
  return `<div class="card"><p class="card-title">Conflicts · ${items.length}</p><div class="evidence">${rows}</div></div>`;
}

function findingsHTML(report) {
  const items = report.findings || [];
  if (!items.length) return "";
  const rows = items
    .map(
      (f) =>
        `<div class="finding"><div class="finding-head">` +
        badge(f.perspective, f.perspective) +
        `<span class="finding-agent">${esc(human(f.agent_name))}</span>` +
        badge(f.stance, f.stance) +
        `<span class="meta-chip">${pct(f.confidence)}</span>` +
        `</div><p>${esc(f.summary)}</p></div>`
    )
    .join("");
  return `<div class="card"><p class="card-title">Agent findings</p><div class="findings">${rows}</div></div>`;
}

function evidenceHTML(report) {
  const items = report.evidence || [];
  if (!items.length) return "";
  const row = (it) => {
    const d = it.document || {};
    const meta = [d.source, d.doc_type ? human(d.doc_type) : "", dateStr(d.published_at), `reliability ${pct(d.reliability)}`]
      .filter(Boolean)
      .join(" · ");
    return (
      `<div class="ev-item"><span class="ev-type">${esc(human(d.doc_type || "doc"))}</span>` +
      `<div class="ev-body"><div class="ev-title">${
        d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.title || "Untitled")}</a>` : esc(d.title || "Untitled")
      }</div><div class="ev-meta">${esc(meta)}</div></div></div>`
    );
  };
  const first = items.slice(0, 8).map(row).join("");
  const rest = items.slice(8);
  const more = rest.length
    ? `<details class="more"><summary>Show ${rest.length} more sources</summary>${rest.map(row).join("")}</details>`
    : "";
  return `<div class="card"><p class="card-title">Evidence · ${items.length} sources</p><div class="evidence">${first}</div>${more}</div>`;
}

function limitationsHTML(report) {
  if (!report.limitations || !report.limitations.length) return "";
  return (
    `<div class="card"><p class="card-title">Limitations</p>` +
    `<ul class="list limits">${report.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`
  );
}

function traceHTML(report) {
  const rows = report.execution_trace || [];
  if (!rows.length) return "";
  const body = rows
    .map((t) => {
      const extra = Object.entries(t)
        .filter(([k]) => k !== "node" && k !== "ms")
        .map(([k, v]) => `${human(k)}: ${Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : v}`)
        .join(" · ");
      return `<tr><td>${esc(human(t.node))}</td><td>${t.ms != null ? Math.round(t.ms) + " ms" : "—"}</td><td>${esc(extra)}</td></tr>`;
    })
    .join("");
  return (
    `<details class="trace"><summary>Execution trace · ${rows.length} steps</summary>` +
    `<table class="trace-table"><thead><tr><th>Stage</th><th>Time</th><th>Detail</th></tr></thead>` +
    `<tbody>${body}</tbody></table></details>`
  );
}

function sourcesHTML(report) {
  const src = report.data_sources || {};
  const chips = Object.entries(src)
    .map(([k, v]) => `<span class="meta-chip">${esc(human(k))}: ${esc(v)}</span>`)
    .join("");
  return chips ? `<div class="meta-row" style="margin-top:18px">${chips}</div>` : "";
}

function reportHTML(r, fallbackText) {
  const q = r.query || {};
  const tickers = (q.tickers || []).map((t) => `<span class="meta-chip ticker">${esc(t)}</span>`).join("");
  const conf = r.confidence || {};
  const band = bandOf(Number(conf.overall || 0));
  const sections = sectionsHTML(r);

  const top =
    `<div class="report-top">` +
    `<div class="report-kicker">Research report · ${esc(dateStr(r.generated_at))}</div>` +
    `<p class="report-q">${esc(q.raw_text || "")}</p>` +
    `<div class="meta-row">` +
    tickers +
    badge(r.overall_stance, r.overall_stance) +
    `<span class="meta-chip">${Math.round(Number(conf.overall || 0) * 100)}% confidence · ${esc(human(band))}</span>` +
    `<span class="meta-chip">${esc(r.report_id || "")}</span>` +
    `<div class="report-actions"><button class="mini-btn" id="downloadBtn" type="button">Download JSON</button></div>` +
    `</div>` +
    (r.thesis ? `<div class="thesis"><span class="thesis-label">Thesis</span>${nl2br(r.thesis)}</div>` : "") +
    sourcesHTML(r) +
    `</div>`;

  const confCard =
    `<div class="card"><p class="card-title">Confidence breakdown</p>` +
    `<div class="confidence">${ringHTML(conf.overall)}` +
    `<div class="conf-bars">` +
    confBar("Evidence quality", conf.evidence_quality) +
    confBar("Agreement", conf.agreement) +
    confBar("Coverage", conf.coverage) +
    confBar("Corroboration", conf.corroboration) +
    `</div></div>` +
    ((conf.penalties || []).length
      ? `<ul class="list limits" style="margin-top:14px">${conf.penalties.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>`
      : "") +
    `</div>`;

  const emptyNote = !sections
    ? `<div class="card"><div class="notice">No narrative sections were produced.${
        fallbackText ? " Raw output is available below." : ""
      }</div></div>`
    : "";

  const body =
    (sections ? `<div class="section">${sections}</div>` : emptyNote) +
    `<div class="grid-2" style="margin-top:16px">${confCard}${metricsHTML(r) || `<div class="card"><p class="card-title">Key metrics</p><p class="section-body" style="color:var(--faint)">No metrics were extracted for this run.</p></div>`}</div>` +
    risksHTML(r) +
    claimsHTML(r) +
    conflictsHTML(r) +
    findingsHTML(r) +
    evidenceHTML(r) +
    limitationsHTML(r) +
    traceHTML(r) +
    `<p class="report-footer">Generated by FinAgents · grounded in SEC EDGAR, Google News and Yahoo Finance · not investment advice.</p>`;

  return top + body;
}

function wireReportActions(report) {
  const btn = $("downloadBtn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (report.report_id || "finagents-report") + ".json";
    a.click();
    URL.revokeObjectURL(url);
  });
}

/* ---------------- init ---------------- */
function resetToEmpty() {
  if (busy && activeJob && !stopping) {
    fetch("/jobs/" + activeJob + "/cancel", { method: "POST" }).catch(() => {});
  }
  pollGen++;
  stopTimer();
  activeJob = null;
  stopping = false;
  pendingCancel = false;
  clearChatContext();
  setBusy(false);
  $("question").value = "";
  $("ticker").value = "";
  $("reportState").innerHTML = "";
  showOnly("empty");
  autoGrow();
  $("question").focus();
}

async function checkServer() {
  try {
    const res = await fetch("/health", { cache: "no-store" });
    setServer(res.ok);
  } catch {
    setServer(false);
  }
}

function init() {
  initTheme();
  initSidebar();
  renderHistory();
  autoGrow();
  checkServer();
  setInterval(checkServer, 15000);

  const form = $("composer");
  const q = $("question");
  q.addEventListener("input", autoGrow);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    }
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submit();
  });

  $("sendBtn").addEventListener("click", (e) => {
    if (!busy) return;
    e.preventDefault();
    cancelRun();
  });

  $("verifyPill").addEventListener("click", () => setPill("verifyPill", !pillOn("verifyPill")));
  $("seqPill").addEventListener("click", () => setPill("seqPill", !pillOn("seqPill")));
  $("newBtn").addEventListener("click", () => {
    $("app").classList.remove("mobile-open");
    resetToEmpty();
  });

  $("suggestions").addEventListener("click", (e) => {
    if (busy) return;
    const chip = e.target.closest(".chip");
    if (!chip) return;
    clearChatContext();
    $("question").value = chip.textContent.trim();
    autoGrow();
    submit();
  });

  $("ticker").addEventListener("input", (e) => {
    e.target.value = e.target.value.toUpperCase();
  });
}

document.addEventListener("DOMContentLoaded", init);

