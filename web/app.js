/* PCB Defect Detector — control panel.
   No framework, no build step. Three files served by the same FastAPI process
   that runs the model. */

"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- fetch --- */

/* A sleeping Space answers requests before the model is ready, so a 503 has to
   stay distinguishable from a real failure — flag it rather than letting an
   error body flow into the success path. */
async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.detail || `${res.status} ${res.statusText}`);
    err.notReady = res.status === 503;
    throw err;
  }
  return body;
}

let toastTimer = null;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
}

/* ----------------------------------------------------------------- boot --- */

let ready = false;

async function boot() {
  const bootEl = $("boot");
  try {
    const info = await api("/model/info");
    ready = true;
    bootEl.hidden = true;
    $("iVersion").textContent = info.model_version;
    $("iImgsz").textContent = `${info.inference_imgsz}px`;
    $("iClasses").textContent = info.classes.length;
    $("iSamples").textContent = info.n_samples;
    $("rVersion").dataset.version = info.model_version;
    $("gateSub").textContent = `promote on ${info.gate_metric}`;
  } catch (e) {
    $("bootMsg").textContent =
      "Container is waking up — the model loads on first use. Retrying…";
    setTimeout(boot, 3000);
  }
}

/* ----------------------------------------------------------------- tabs --- */

const VIEWS = ["manual", "inspect", "registry", "drift", "limits", "api"];
let current = "manual";

$("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-view]");
  if (btn) show(btn.dataset.view);
});

/* A tablist is expected to move with the arrow keys, with Home/End jumping to
   the ends. Tab itself should leave the strip rather than walk through six
   buttons, which is what the roving tabindex in show() is for. */
$("tabs").addEventListener("keydown", (e) => {
  const keys = { ArrowRight: 1, ArrowLeft: -1, Home: 0, End: 0 };
  if (!(e.key in keys)) return;
  e.preventDefault();
  const i = VIEWS.indexOf(current);
  const next =
    e.key === "Home" ? 0 :
    e.key === "End" ? VIEWS.length - 1 :
    (i + keys[e.key] + VIEWS.length) % VIEWS.length;
  show(VIEWS[next]);
  $(`tab-${VIEWS[next]}`).focus();
});

function show(name) {
  if (!VIEWS.includes(name)) return;
  current = name;
  for (const v of VIEWS) $(`v-${v}`).hidden = v !== name;
  for (const b of $("tabs").querySelectorAll("button")) {
    const on = b.dataset.view === name;
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;   // roving tabindex: one stop for the whole strip
  }
  if (name === "registry") loadRegistry();
  if (name === "drift") loadMonitor();
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.scrollTo({ top: 0, behavior: still ? "auto" : "smooth" });
}

/* ------------------------------------------------------------- tooltips --- */

/* One fixed node, positioned by arithmetic. A tooltip rendered inside a card
   gets clipped by the card, and one centred on a 17px button runs off the left
   edge of a narrow screen. */
const tipEl = $("tip");
let tipFor = null;

function showTip(btn) {
  tipEl.textContent = btn.dataset.tip;
  tipEl.hidden = false;
  tipFor = btn;
  placeTip();
}

function placeTip() {
  if (!tipFor) return;
  const r = tipFor.getBoundingClientRect();
  const t = tipEl.getBoundingClientRect();
  const pad = 10;
  let left = r.left + r.width / 2 - t.width / 2;
  left = Math.max(pad, Math.min(left, window.innerWidth - t.width - pad));
  let top = r.top - t.height - 9;
  if (top < pad) top = r.bottom + 9;
  tipEl.style.left = `${Math.round(left)}px`;
  tipEl.style.top = `${Math.round(top)}px`;
}

function hideTip() { tipEl.hidden = true; tipFor = null; }

document.addEventListener("mouseover", (e) => {
  const b = e.target.closest(".q");
  if (b) showTip(b);
  else if (tipFor && !e.target.closest("#tip")) hideTip();
});
document.addEventListener("focusin", (e) => {
  const b = e.target.closest(".q");
  if (b) showTip(b);
});
document.addEventListener("focusout", hideTip);
/* Touch: a tap fires neither hover nor :focus-visible, so the tap has to open
   the tip itself. It must not *toggle*, though — a tap also emits a synthetic
   mouseover first on most devices, so toggling would open the tip and then
   immediately close it again. Show on the button, hide on anything else. */
document.addEventListener("click", (e) => {
  const b = e.target.closest(".q");
  if (b) { e.preventDefault(); showTip(b); }
  else if (tipFor) hideTip();
});
addEventListener("scroll", () => { if (tipFor) requestAnimationFrame(placeTip); }, { passive: true });
addEventListener("resize", () => { if (tipFor) placeTip(); });
addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });

/* -------------------------------------------------------------- inspect --- */

let samples = [];
let activeSample = null;
let busy = false;

async function loadSamples() {
  try {
    samples = await api("samples/manifest.json");
  } catch {
    $("samples").innerHTML = '<span class="note">Sample boards are unavailable.</span>';
    return;
  }
  $("samples").innerHTML = samples
    .map((s) => `<button type="button" data-id="${s.id}" aria-pressed="false">${s.name}</button>`)
    .join("");
}

$("samples").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-id]");
  if (b && !busy) runSample(b.dataset.id);
});

$("conf").addEventListener("input", (e) => { $("confOut").textContent = Number(e.target.value).toFixed(2); });
$("conf").addEventListener("change", () => {
  if (activeSample) runSample(activeSample);
});

function setBusy(on) {
  busy = on;
  for (const b of $("samples").querySelectorAll("button")) b.disabled = on;
}

async function runSample(id) {
  const s = samples.find((x) => x.id === id);
  if (!s) return;
  activeSample = id;
  setBusy(true);
  for (const b of $("samples").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.id === id));
  }
  $("stage").innerHTML = `<img src="${s.file}" alt="Circuit board ${s.name}">
    <div class="empty" style="position:absolute;inset:auto 0 0 0">Scoring…</div>`;
  try {
    const conf = $("conf").value;
    const r = await api(`/predict/sample?sample_id=${encodeURIComponent(id)}&conf=${conf}`,
                        { method: "POST" });
    render(s.file, r, s.truth, s.name);
  } catch (e) {
    $("stage").innerHTML = '<div class="empty">Could not score that board.</div>';
    toast(e.notReady ? "The model is still loading — try again in a few seconds." : e.message);
  } finally {
    setBusy(false);
  }
}

$("file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  activeSample = null;
  for (const b of $("samples").querySelectorAll("button")) b.setAttribute("aria-pressed", "false");
  $("uploadNote").textContent = f.name;
  const url = URL.createObjectURL(f);
  $("stage").innerHTML = `<img src="${url}" alt="Uploaded board">
    <div class="empty" style="position:absolute;inset:auto 0 0 0">Scoring…</div>`;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await api(`/predict?conf=${$("conf").value}`, { method: "POST", body: fd });
    render(url, r, null, f.name);
  } catch (err) {
    $("stage").innerHTML = '<div class="empty">Could not score that image.</div>';
    toast(err.notReady ? "The model is still loading — try again in a few seconds." : err.message);
  }
});

/* --------------------------------------------------------- box matching --- */

/* "pcb-defect-detector@production:v3" and "local:best.pt" both mean v3 here.
   The readout has room for a version, not a resolution path. */
function shortVersion(v) {
  const tagged = /:v(\d+)$/.exec(v);
  if (tagged) return `v${tagged[1]}`;
  if (/^local:/.test(v)) return regData ? `v${regData.shipped_weights_version}` : "baked in";
  if (/^base:/.test(v)) return "untrained";
  return v;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (!inter) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

/* Greedy highest-overlap-first matching on location alone. Whether the class
   was also right is reported separately — a box in the right place with the
   wrong name is a different failure from not seeing the defect at all. */
function match(dets, truth, thr = 0.3) {
  const pairs = [];
  dets.forEach((d, di) => truth.forEach((t, ti) => {
    const v = iou(d.box, t.box);
    if (v >= thr) pairs.push({ di, ti, v });
  }));
  pairs.sort((a, b) => b.v - a.v);
  const dUsed = new Set(), tUsed = new Set(), map = new Map();
  for (const p of pairs) {
    if (dUsed.has(p.di) || tUsed.has(p.ti)) continue;
    dUsed.add(p.di); tUsed.add(p.ti);
    map.set(p.di, { ti: p.ti, labelOk: dets[p.di].label === truth[p.ti].label });
  }
  return {
    map,
    missed: truth.filter((_, ti) => !tUsed.has(ti)),
    falseAlarms: dets.filter((_, di) => !dUsed.has(di)).length,
  };
}

/* ------------------------------------------------------------- rendering --- */

function render(imgSrc, r, truth, label) {
  const [w, h] = r.image_size;
  const dets = [...r.detections].sort((a, b) => b.confidence - a.confidence);
  const m = truth ? match(dets, truth) : null;

  const pct = (b) => ({
    left: (b[0] / w) * 100, top: (b[1] / h) * 100,
    width: ((b[2] - b[0]) / w) * 100, height: ((b[3] - b[1]) / h) * 100,
  });
  const boxHtml = (b, cls, text) => {
    const p = pct(b);
    return `<div class="bbox ${cls}" style="left:${p.left}%;top:${p.top}%;width:${p.width}%;height:${p.height}%"><b>${text}</b></div>`;
  };

  let html = `<img src="${imgSrc}" alt="Board ${label}">`;
  dets.forEach((d) => { html += boxHtml(d.box, "", `${d.label} ${d.confidence.toFixed(2)}`); });
  if (m) m.missed.forEach((t) => { html += boxHtml(t.box, "missed", `missed · ${t.label}`); });
  $("stage").innerHTML = html;

  $("rConf").textContent = dets.length ? dets[0].confidence.toFixed(2) : "—";
  $("rVersion").textContent = shortVersion(r.model_version);

  if (m) {
    const found = truth.length - m.missed.length;
    $("rFound").textContent = `${found} / ${truth.length}`;
    $("rMissed").textContent = m.missed.length;
    $("rMissed").className = m.missed.length ? "warnv" : "";
    $("rFalse").textContent = m.falseAlarms;
    const wrongClass = [...m.map.values()].filter((v) => !v.labelOk).length;
    $("matchNote").textContent =
      `Boxes are matched to the labels by overlap (IoU ≥ 0.3). ` +
      `${found} of ${truth.length} labelled defects were located` +
      (wrongClass ? `, though ${wrongClass} of them were given the wrong class name` : "") +
      `. ${m.falseAlarms} detection${m.falseAlarms === 1 ? "" : "s"} matched no label at all.`;
  } else {
    $("rFound").textContent = dets.length;
    $("rMissed").textContent = "n/a";
    $("rMissed").className = "";
    $("rFalse").textContent = "n/a";
    $("matchNote").textContent =
      "This image has no ground-truth labels, so misses and false alarms cannot be computed — " +
      "only what the model claims to see.";
  }

  const body = $("detTable").querySelector("tbody");
  body.innerHTML = dets.length
    ? dets.map((d, i) => {
        const mm = m ? m.map.get(i) : null;
        const verdict = !m ? "—"
          : mm ? (mm.labelOk ? "yes" : "overlaps, wrong class")
          : "no label there";
        return `<tr><td>${d.label}</td><td>${d.confidence.toFixed(3)}</td>
          <td>${d.box.map((n) => Math.round(n)).join(", ")}</td><td>${verdict}</td></tr>`;
      }).join("")
    : '<tr><td colspan="4">The model reported nothing above this confidence threshold.</td></tr>';
}

/* ------------------------------------------------------------- registry --- */

let regData = null;      // registry.json, fetched once at boot
let registryLoaded = false;

/* The serving container has no MLflow in it, so the registry lookup fails by
   design and the loader falls back to the weights baked into the image. Those
   weights *are* the version that held the production alias when the image was
   built — say so, rather than showing a visitor a bare filename. */
function servingExplainer(reg, serving) {
  const prod = reg.versions.find((v) => v.status === "production");
  if (!serving || serving === "—") return "The server has not reported a version yet.";
  if (/^local:/.test(serving)) {
    return `These are the weights baked into this image at build time — v${reg.shipped_weights_version},
      the version holding the <code>production</code> alias. The registry itself is not queried here:
      MLflow is deliberately not installed in the serving container, and the loader falls back to
      local weights exactly as it is designed to.`;
  }
  if (/^base:/.test(serving)) {
    return `<b>Warning:</b> this is the untrained base model, not a PCB detector. The trained
      weights are missing from the image, so anything on the Inspect tab is meaningless.`;
  }
  return `That is v${prod.version} in the table — read from the live server, not from this file.`;
}

async function loadRegistry() {
  if (registryLoaded || !regData) return;
  registryLoaded = true;
  const reg = regData;
  const serving = $("iVersion").textContent;
  const n = (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d));
  const flag = { production: 'f-prod', promoted: 'f-pass', rejected: 'f-rej' };
  const word = { production: 'production', promoted: 'promoted', rejected: 'blocked' };

  $("regTable").querySelector("tbody").innerHTML = reg.versions.map((v) => `
    <tr class="${v.status === 'production' ? 'prod' : v.status === 'rejected' ? 'reject' : ''}">
      <td>v${v.version}</td><td>${v.base}</td><td>${v.imgsz}</td><td>${v.epochs ?? "—"}</td>
      <td>${n(v["mAP50"])}</td><td>${n(v["mAP50-95"])}</td>
      <td>${n(v.precision)}</td><td>${n(v.recall)}</td>
      <td>${v.train_minutes == null ? "—" : v.train_minutes + " min"}</td>
      <td><span class="flag ${flag[v.status]}">${word[v.status]}</span></td>
    </tr>`).join("");

  $("regNotes").innerHTML = `
    <div class="banner">
      <b>Serving right now:</b> <code>${serving || "unknown"}</code>. ${servingExplainer(reg, serving)}
    </div>
    ${reg.versions.map((v) => `<h3>v${v.version} — ${word[v.status]}</h3><p>${v.note}</p>`).join("")}
    <p class="note" style="margin-top:18px">${reg.source}</p>`;
}

/* -------------------------------------------------------------- monitor --- */

async function loadMonitor() {
  try {
    const m = await api("/monitor/summary");
    const set = (id, v) => { $(id).textContent = v; };
    if (!m.n) {
      set("mCount", "0"); set("mDet", "—"); set("mConf", "—"); set("mBright", "—");
      return;
    }
    set("mCount", m.n);
    set("mDet", m.avg_detections.toFixed(2));
    set("mConf", m.avg_confidence.toFixed(3));
    set("mBright", m.avg_brightness.toFixed(1));
  } catch (e) {
    toast(e.notReady ? "The container is still waking up." : e.message);
  }
}

$("refreshMonitor").addEventListener("click", loadMonitor);

/* ----------------------------------------------------------------- init --- */

/* registry.json is a static file, so it loads even while the model is warming —
   which is what lets the Registry tab render during a cold start. */
api("registry.json")
  .then((r) => { regData = r; if (current === "registry") loadRegistry(); })
  .catch(() => {});

boot();
loadSamples();
show("manual");
