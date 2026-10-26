/* InspectAI · Evidence Explorer
   Reads window.INSPECTAI_DEMO (demo/data.js) and replays the real pipeline
   output. No measurements are computed here: the page only formats them. */

(() => {
  "use strict";

  const DATA = window.INSPECTAI_DEMO;
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)");

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const STATUS_LABELS = {
    applied: "LOCKED",
    ambiguous: "AMBIGUOUS",
    rejected: "REJECTED",
    skipped: "SKIPPED",
    not_needed: "NOT NEEDED",
  };

  const LAYER_LABELS = {
    aligned: "ALIGNED TEST",
    heatmap: "LAB ΔE HEATMAP",
    mask: "CHANGE MASK",
    regions: "CHANGE REGIONS",
  };

  const LAYER_NOTES = {
    aligned:
      "The registered test image on the reference canvas. The wipe reveals how far the final transform had to move it.",
    heatmap:
      "Turbo colormap of the Lab ΔE magnitude after alignment, scaled for display. It is a difference map, not a defect score.",
    mask:
      "Pixels above the adaptive Lab threshold after the 3×3 open/close cleanup and the minimum-area / minimum-contrast gates.",
    regions:
      "Connected components that survived the reporting gate, drawn from their measured bounds. Select a box or a row to inspect it.",
  };

  const TABS = ["overview", "register", "analyze", "verify"];
  const state = {
    index: 0,
    tab: "overview",
    step: 0,
    layer: "aligned",
    wipe: 50,
    region: -1,
    timer: null,
  };

  // ------------------------------------------------------------ utilities

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    applyAttrs(node, attrs);
    appendChildren(node, children);
    return node;
  }

  function svgEl(tag, attrs, ...children) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    applyAttrs(node, attrs);
    appendChildren(node, children);
    return node;
  }

  function applyAttrs(node, attrs) {
    if (!attrs) return;
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.setAttribute("class", value);
      else if (key === "text") node.textContent = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else node.setAttribute(key, value === true ? "" : String(value));
    }
  }

  function appendChildren(node, children) {
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function number(value, digits = 2) {
    if (value === null || value === undefined) return "—";
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    return String(Number(numeric.toFixed(digits)));
  }

  function fixed(value, digits = 2) {
    if (value === null || value === undefined) return "—";
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    return numeric.toFixed(digits);
  }

  function percent(value, digits = 2) {
    if (value === null || value === undefined) return "—";
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    return `${numeric.toFixed(digits)}%`;
  }

  function yesNo(value) {
    if (value === null || value === undefined) return "not scored";
    return value ? "yes" : "no";
  }

  function statusLabel(status) {
    return STATUS_LABELS[status] || String(status || "—").toUpperCase();
  }

  function stepLabel(step) {
    if (step.status === "accepted") return "LOCKED";
    if (step.status === "ambiguous") return "AMBIGUOUS";
    if (step.status === "rejected") return "REJECTED";
    if (step.name === "Final") return "NO ALIGNMENT";
    return "SKIPPED";
  }

  function stepTone(step) {
    if (step.status === "accepted") return "accepted";
    if (step.status === "ambiguous") return "ambiguous";
    if (step.status === "rejected") return "rejected";
    return "skipped";
  }

  function formatValue(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") {
      if (Number.isInteger(value)) return String(value);
      return String(Number(value.toFixed(4)));
    }
    if (Array.isArray(value)) {
      if (!value.length) return "—";
      if (value.every((item) => typeof item === "number")) {
        return value.map((item) => number(item, 4)).join(" · ");
      }
      return JSON.stringify(value);
    }
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function kv(label, value, extraClass) {
    return el(
      "div",
      { class: `kv ${extraClass || ""}` },
      el("span", { class: "k", text: label }),
      el("span", { class: "v", text: value }),
    );
  }

  function fillKv(container, entries) {
    clear(container);
    for (const [label, value] of entries) container.append(kv(label, value));
  }

  function metricEntries(metrics, skip = []) {
    const skipSet = new Set([...skip, "opencv_functions", "functions", "reject_reason"]);
    return Object.entries(metrics || {})
      .filter(([key, value]) => !skipSet.has(key) && value !== null && value !== undefined)
      .map(([key, value]) => [
        key.replaceAll("_", " "),
        typeof value === "number" ? number(value, 4) : formatValue(value),
      ]);
  }

  function caseIndex(caseId) {
    return DATA.cases.findIndex((item) => item.id === caseId);
  }

  function currentCase() {
    return DATA.cases[state.index];
  }

  function finalStep(caseData) {
    return caseData.steps[caseData.steps.length - 1] || { metrics: {} };
  }

  function methodNames(caseData) {
    if (!caseData.method || caseData.method === "none") return [];
    return String(caseData.method).split("+").filter(Boolean);
  }

  // --------------------------------------------------------- URL handling

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    const caseId = params.get("case");
    const found = caseId ? caseIndex(caseId) : -1;
    if (found >= 0) state.index = found;
    const tab = params.get("tab");
    if (TABS.includes(tab)) state.tab = tab;
    const step = Number(params.get("step"));
    if (Number.isInteger(step) && step >= 0) state.step = step;
    const layer = params.get("layer");
    if (Object.prototype.hasOwnProperty.call(LAYER_LABELS, layer)) state.layer = layer;
    const wipe = Number(params.get("wipe"));
    if (Number.isFinite(wipe)) state.wipe = Math.min(100, Math.max(0, wipe));
  }

  function writeUrl() {
    const params = new URLSearchParams();
    const caseData = currentCase();
    if (state.index > 0) params.set("case", caseData.id);
    if (state.tab !== "overview") params.set("tab", state.tab);
    if (state.tab === "register" && state.step > 0) params.set("step", String(state.step));
    if (state.tab === "analyze") {
      if (state.layer !== "aligned") params.set("layer", state.layer);
      if (state.wipe !== 50) params.set("wipe", String(state.wipe));
    }
    const query = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
  }

  // ------------------------------------------------------------ case rail

  function renderCaseList() {
    const list = $("#case-list");
    clear(list);
    DATA.cases.forEach((caseData, index) => {
      const button = el(
        "button",
        {
          class: "case-btn",
          type: "button",
          "aria-current": index === state.index ? "true" : "false",
          onclick: () => selectCase(index),
        },
        el("span", { class: "case-num", text: String(index + 1).padStart(2, "0") }),
        el(
          "span",
          {},
          el("span", { class: "case-label", text: caseData.label }),
          el("span", {
            class: "case-meta",
            text: `${percent(caseData.changed_percentage)} · ${caseData.region_count} region${caseData.region_count === 1 ? "" : "s"}`,
          }),
          el("span", {
            class: "case-state",
            dataset: { status: caseData.status },
            text: statusLabel(caseData.status),
          }),
        ),
      );
      list.append(button);
    });
    $("#case-count").textContent = `${DATA.cases.length} CASES`;
  }

  function selectCase(index) {
    stopPlayback();
    state.index = index;
    state.step = 0;
    state.region = -1;
    state.wipe = 50;
    renderCaseList();
    renderCase();
    writeUrl();
    if (window.innerWidth <= 980) {
      $("#workspace").scrollIntoView({ behavior: REDUCED.matches ? "auto" : "smooth", block: "start" });
    }
  }

  // ------------------------------------------------------------- case head

  function renderHead() {
    const caseData = currentCase();
    $("#case-index").textContent =
      `CASE ${String(state.index + 1).padStart(2, "0")} / ${String(DATA.cases.length).padStart(2, "0")}  ·  ${caseData.id.toUpperCase()}`;
    $("#case-name").textContent = caseData.label;
    $("#case-expectation").textContent = `Designed expectation: ${caseData.expectation}.`;

    const chips = $("#case-chips");
    clear(chips);
    chips.append(el("span", { class: "chip chip-strong", text: `METHOD ${caseData.method.toUpperCase()}` }));
    for (const method of methodNames(caseData)) {
      chips.append(el("span", { class: "chip", text: method.toUpperCase() }));
    }
    const referenceSize = caseData.dimensions.reference_dimensions;
    const testSize = caseData.dimensions.test_dimensions;
    chips.append(
      el("span", {
        class: "chip",
        text: `REFERENCE ${referenceSize.width}×${referenceSize.height}`,
      }),
    );
    if (testSize.width !== referenceSize.width || testSize.height !== referenceSize.height) {
      chips.append(el("span", { class: "chip", text: `TEST ${testSize.width}×${testSize.height}` }));
    }
    chips.append(el("span", { class: "chip", text: `SCHEMA 2` }));
    chips.append(el("span", { class: "chip", text: `${number(caseData.duration_seconds, 2)} s COMPUTE` }));

    const verdict = $("#case-verdict");
    verdict.dataset.status = caseData.status;
    const badge = $("#verdict-badge");
    badge.dataset.status = caseData.status;
    badge.textContent = caseData.status_label;
    $("#verdict-value").textContent = fixed(caseData.changed_percentage, 2);
    $("#verdict-sub").textContent =
      `${caseData.method.toUpperCase()} · ${caseData.region_count} REGION${caseData.region_count === 1 ? "" : "S"} · ` +
      `OVERLAP ${percent(caseData.alignment.overlap_fraction * 100, 1)}`;
  }

  // -------------------------------------------------------------- overview

  function describeCase(caseData) {
    const orientationSteps = caseData.steps.filter((step) => step.model === "orientation");
    const translation = caseData.steps.find((step) => step.name === "Coarse: phase correlation");
    const parts = [];

    if (orientationSteps.some((step) => step.status === "accepted" && step.name.startsWith("Orientation fix"))) {
      const applied = orientationSteps.find((step) => step.status === "accepted");
      const degrees = applied.metrics.clockwise_degrees ?? "90";
      parts.push(`A ${degrees}° clockwise orientation fix was applied before the shape check.`);
    } else if (orientationSteps.some((step) => step.status === "ambiguous")) {
      parts.push("Quarter-turn orientation scoring was not decisive, so no rotation was silently chosen.");
    } else {
      parts.push("No quarter-turn orientation correction was needed.");
    }

    if (translation) {
      if (translation.status === "accepted") {
        parts.push(
          `Phase correlation proposed a ${number(translation.metrics.shift_x, 1)} px x / ` +
            `${number(translation.metrics.shift_y, 1)} px y shift and the candidate passed the improvement gate.`,
        );
      } else if (translation.status === "ambiguous") {
        parts.push("The phase-correlation peak was not unique, so the ambiguous translation was left unresolved.");
      } else {
        parts.push(`The coarse translation step did not change the alignment (${stepLabel(translation).toLowerCase()}).`);
      }
    }

    if (methodNames(caseData).includes("photometric")) {
      parts.push("A robust per-channel gain/offset/gradient fit normalized smooth exposure differences before differencing.");
    }

    const final = finalStep(caseData);
    const metrics = final.metrics || {};
    const valid = Number(caseData.valid_pixels || caseData.thresholds.valid_pixels || 0);
    parts.push(
      `The final Lab ΔE comparison marks ${fixed(caseData.changed_percentage, 2)}% of the valid overlap as changed` +
        (valid ? ` (${caseData.valid_pixels.toLocaleString()} px)` : "") +
        ` and reports ${caseData.region_count} change region${caseData.region_count === 1 ? "" : "s"}.`,
    );
    if (metrics.ncc !== undefined) {
      parts.push(
        `Final overlap correlation is ${number(metrics.ncc, 4)} with a normalized residual of ` +
          `${number(metrics.residual_normalized, 4)}.`,
      );
    }
    return parts.join(" ");
  }

  function renderOverview() {
    const caseData = currentCase();
    const stats = $("#overview-stats");
    clear(stats);
    const stat = (label, value, tone) =>
      el(
        "div",
        { class: "stat", dataset: tone ? { tone } : {} },
        el("span", { class: "stat-label", text: label }),
        el("span", { class: "stat-value", text: value }),
      );
    const tone =
      caseData.status === "applied"
        ? "cyan"
        : caseData.status === "ambiguous"
          ? "amber"
          : caseData.status === "rejected"
            ? "red"
            : "";
    stats.append(stat("alignment", caseData.status_label, tone));
    stats.append(stat("method", caseData.method.toUpperCase()));
    stats.append(stat("measured change", percent(caseData.changed_percentage)));
    stats.append(
      stat("change regions", `${caseData.region_count} / ${caseData.detected_region_count} detected`),
    );

    $("#overview-story").textContent = describeCase(caseData);

    const finalMetrics = finalStep(caseData).metrics || {};
    const photometric = caseData.photometric || {};
    fillKv($("#overview-metrics"), [
      ["lab Δe threshold", number(caseData.thresholds.lab_delta_final, 2)],
      ["noise mad", number(caseData.thresholds.lab_noise_mad_final, 3)],
      ["valid pixels", (caseData.valid_pixels || 0).toLocaleString()],
      ["overlap correlation", number(finalMetrics.ncc, 4)],
      ["normalized residual", number(finalMetrics.residual_normalized, 4)],
      ["canvas overlap", percent(caseData.alignment.overlap_fraction * 100, 1)],
      ["estimated shift", `${number(caseData.alignment.shift_x, 1)}, ${number(caseData.alignment.shift_y, 1)} px`],
      ["estimated rotation", `${number(caseData.alignment.rotation_degrees, 2)}°`],
      ["estimated scale", number(caseData.alignment.scale, 4)],
      ["phase response", number(caseData.alignment.response, 4)],
      ["edge similarity", caseData.edge.similarity_percentage === null ? "not scored" : percent(caseData.edge.similarity_percentage)],
      ["mean ssim", caseData.ssim.mean_ssim === null ? "not scored" : number(caseData.ssim.mean_ssim, 4)],
      [
        "photometric fit",
        photometric.applied
          ? `applied · ${percent((photometric.improvement_relative || 0) * 100, 1)} better`
          : photometric.status || "not run",
      ],
      ["agent iterations", String(caseData.iteration_count)],
      ["compute time", `${number(caseData.duration_seconds, 2)} s`],
    ]);

    const decisions = $("#overview-decisions");
    clear(decisions);
    for (const decision of caseData.decisions) decisions.append(el("li", { text: decision }));

    const evidence = $("#overview-evidence");
    clear(evidence);
    for (const line of caseData.evidence) evidence.append(el("li", { text: line }));
  }

  // -------------------------------------------------------------- register

  function renderSteps() {
    const caseData = currentCase();
    const list = $("#step-list");
    clear(list);
    caseData.steps.forEach((step, index) => {
      list.append(
        el(
          "button",
          {
            class: "step-btn",
            type: "button",
            role: "option",
            "aria-selected": index === state.step ? "true" : "false",
            onclick: () => selectStep(index),
          },
          el("span", { class: "step-num", text: String(step.index).padStart(2, "0") }),
          el(
            "span",
            {},
            el("span", { class: "step-name", text: step.name }),
            el("span", {
              class: "step-status",
              dataset: { status: stepTone(step) },
              text: stepLabel(step),
            }),
          ),
        ),
      );
    });
    $("#step-count").textContent = `${caseData.steps.length} STEPS`;
    $("#step-range").max = String(Math.max(0, caseData.steps.length - 1));
    $("#step-range").value = String(state.step);
  }

  function selectStep(index, { updateUrl = true } = {}) {
    const caseData = currentCase();
    const step = caseData.steps[index];
    if (!step) return;
    state.step = index;

    $$(".step-btn").forEach((button, buttonIndex) => {
      button.setAttribute("aria-selected", buttonIndex === index ? "true" : "false");
    });
    $("#step-prev").disabled = index === 0;
    $("#step-next").disabled = index === caseData.steps.length - 1;
    $("#step-range").value = String(index);
    $("#step-range-out").textContent = `${String(index).padStart(2, "0")} / ${String(caseData.steps.length - 1).padStart(2, "0")}`;

    $("#step-kicker").textContent =
      `STEP ${String(step.index).padStart(2, "0")} / ${String(caseData.steps.length - 1).padStart(2, "0")}`;
    $("#step-name").textContent = step.name;
    const badge = $("#step-badge");
    badge.textContent = stepLabel(step);
    badge.dataset.status = stepTone(step);
    $("#step-reason").textContent = step.reason;

    const image = $("#step-image");
    const empty = $("#step-image-empty");
    if (step.preview) {
      image.src = step.preview;
      image.hidden = false;
      empty.hidden = true;
      image.alt = `Recorded preview of step ${step.index}: ${step.name}`;
    } else {
      image.removeAttribute("src");
      image.hidden = true;
      empty.hidden = false;
    }

    const facts = [
      ["model", step.model],
      ["status", stepLabel(step)],
      ["keypoints", step.keypoint_count ? String(step.keypoint_count) : "—"],
      ["opencv calls", step.functions.length ? String(step.functions.length) : "—"],
    ];
    fillKv($("#step-facts"), facts);

    const matrix = $("#step-matrix");
    clear(matrix);
    (step.matrix || []).forEach((row, rowIndex) => {
      row.forEach((value) => {
        const label = formatValue(value);
        const classes = [];
        if (label.startsWith("-")) classes.push("negative");
        if (rowIndex === 2) classes.push("identity");
        matrix.append(el("span", { class: classes.join(" "), text: label }));
      });
    });

    const metricsContainer = $("#step-metrics");
    clear(metricsContainer);
    const entries = Array.isArray(step.metrics.reject_reason)
      ? []
      : metricEntries(step.metrics);
    if (!entries.length) {
      metricsContainer.append(el("p", { class: "footnote", text: "No numeric metrics were recorded for this step." }));
    } else {
      for (const [label, value] of entries) metricsContainer.append(kv(label, value));
    }
    if (step.functions.length) {
      const funcs = el("div", { class: "funcs", style: "margin-top:10px" });
      for (const name of step.functions) funcs.append(el("span", { class: "func", text: name }));
      metricsContainer.append(funcs);
    }

    if (updateUrl) writeUrl();
  }

  function renderRegister() {
    const caseData = currentCase();
    state.step = Math.min(state.step, caseData.steps.length - 1);
    renderSteps();
    selectStep(state.step, { updateUrl: false });
    const play = $("#step-play");
    play.disabled = REDUCED.matches;
    play.title = REDUCED.matches ? "Playback is disabled because reduced motion is requested." : "Advance one step every 1.1 s";
    play.textContent = state.timer ? "❚❚ PAUSE" : "▶ PLAY";
  }

  function togglePlayback() {
    if (state.timer) {
      stopPlayback();
      return;
    }
    if (REDUCED.matches) return;
    const caseData = currentCase();
    if (state.step >= caseData.steps.length - 1) selectStep(0);
    $("#step-play").textContent = "❚❚ PAUSE";
    state.timer = window.setInterval(() => {
      if (state.step >= currentCase().steps.length - 1) {
        stopPlayback();
        return;
      }
      selectStep(state.step + 1);
      $$(".step-btn")[state.step]?.scrollIntoView({ block: "nearest" });
    }, 1100);
  }

  function stopPlayback() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null;
    const play = $("#step-play");
    if (play) play.textContent = "▶ PLAY";
  }

  // --------------------------------------------------------------- analyze

  function methodState(value) {
    if (value === null || value === undefined) return "unavailable";
    return value ? "change" : "nochange";
  }

  function renderAnalyze() {
    const caseData = currentCase();
    const stage = $("#stage");
    stage.style.aspectRatio = `${caseData.width} / ${caseData.height}`;

    for (const button of $$("#layer-tabs .chip-btn")) {
      button.classList.toggle("is-active", button.dataset.layer === state.layer);
    }
    const layerImage = $("#stage-layer");
    layerImage.src = caseData.images[state.layer === "regions" ? "aligned" : state.layer];
    layerImage.alt =
      state.layer === "regions" ? "Aligned test image with change-region boxes" : `${LAYER_LABELS[state.layer]} for ${caseData.label}`;
    $("#stage-ref").src = caseData.images.reference;
    $("#stage-ref").alt = `Reference image for ${caseData.label}`;
    $("#layer-name").textContent = LAYER_LABELS[state.layer];
    $("#stage-label-right").textContent = LAYER_LABELS[state.layer];
    $("#stage-label-left").textContent = "REFERENCE";
    $("#layer-note").textContent = LAYER_NOTES[state.layer];
    $("#wipe-label-right").textContent = LAYER_LABELS[state.layer];

    const wipe = $("#wipe");
    wipe.value = String(state.wipe);
    applyWipe();

    renderBoxes();
    renderRegionList();
    renderMethodPills();
  }

  function applyWipe() {
    const wipe = $("#wipe");
    const top = $("#stage-ref");
    const value = Number(wipe.value);
    state.wipe = value;
    top.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    $("#wipe-line").style.left = `${value}%`;
    $("#wipe-out").textContent = `${value}%`;
  }

  function renderBoxes() {
    const caseData = currentCase();
    const svg = $("#stage-boxes");
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${caseData.width} ${caseData.height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const show = state.layer === "regions" && caseData.regions.length > 0;
    $("#stage-empty").hidden = !(state.layer === "regions" && caseData.regions.length === 0);
    svg.style.display = show ? "block" : "none";
    if (!show) return;

    caseData.regions.forEach((region, index) => {
      const rect = svgEl("rect", {
        x: region.x,
        y: region.y,
        width: region.width,
        height: region.height,
        class: index === state.region ? "is-selected" : "",
        tabindex: "0",
        role: "button",
        "aria-label": `Change region ${index + 1}: ${region.width} by ${region.height} pixels at ${region.x}, ${region.y}`,
      });
      rect.style.pointerEvents = "auto";
      rect.style.cursor = "pointer";
      rect.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectRegion(index);
        }
      });
      svg.append(rect);
      if (index === state.region) {
        svg.append(
          svgEl(
            "text",
            { x: region.x + 4, y: Math.max(16, region.y - 6) },
            `R${index + 1}`,
          ),
        );
      }
    });
  }

  function selectRegion(index) {
    state.region = index;
    renderBoxes();
    renderRegionList();
  }

  function renderRegionList() {
    const caseData = currentCase();
    const list = $("#region-list");
    clear(list);
    $("#region-count").textContent = `${caseData.region_count} REPORTED`;
    if (!caseData.regions.length) {
      list.append(
        el("p", {
          class: "region-empty",
          text: `No connected component passed the ${caseData.minimum_region_area} px area and Lab-contrast gates. Change indications are below the reporting gate.`,
        }),
      );
      $("#region-detail").textContent = "";
      return;
    }
    caseData.regions.forEach((region, index) => {
      list.append(
        el(
          "button",
          {
            class: `region-btn ${index === state.region ? "is-selected" : ""}`,
            type: "button",
            onclick: () => selectRegion(index),
          },
          el("span", { class: "r-id", text: `R${index + 1}` }),
          el("span", {
            text: `x ${region.x}, y ${region.y} · ${region.width} × ${region.height}`,
          }),
          el("span", { text: `${region.area_pixels} px` }),
        ),
      );
    });
    const region = caseData.regions[state.region];
    $("#region-detail").textContent = region
      ? `R${state.region + 1} · x ${region.x}, y ${region.y} · ${region.width} × ${region.height} px · area ${region.area_pixels} px. Boxed by cv2.connectedComponentsWithStats after the Lab threshold and 3×3 open/close cleanup.`
      : `Select a region box or row to read its measured bounds. ${caseData.region_count} region${caseData.region_count === 1 ? "" : "s"} survived the gate.`;
  }

  function renderMethodPills() {
    const caseData = currentCase();
    const container = $("#method-pills");
    clear(container);
    const entries = [
      ["Lab ΔE", caseData.methods.lab_difference_change_indicated, percent(caseData.changed_percentage)],
      ["Canny", caseData.methods.canny_change_indicated, caseData.edge.similarity_percentage === null ? "not scored" : `${percent(caseData.edge.similarity_percentage)} similar`],
      ["SSIM", caseData.methods.ssim_change_indicated, caseData.ssim.mean_ssim === null ? "not scored" : `${number(caseData.ssim.mean_ssim, 4)} mean`],
    ];
    for (const [name, value, detail] of entries) {
      const tone = methodState(value);
      const stateText =
        tone === "change" ? "change indicated" : tone === "nochange" ? "no change" : "not scored";
      container.append(
        el(
          "div",
          { class: "method-pill", dataset: { state: tone } },
          el("span", {}, el("b", { text: name }), ` · ${detail}`),
          el("span", { class: "state", text: stateText }),
        ),
      );
    }
    const notes = {
      both_indicate_change: "Lab ΔE and Canny edges both indicate change.",
      pixel_only_indicates_change: "Lab ΔE indicates change; Canny edges do not. Read the difference map and region bounds as evidence, not a verdict.",
      edge_only_indicates_change: "Canny edges indicate change; the Lab difference does not. Check for structural or sub-threshold colour differences.",
      neither_indicates_change: "Neither Lab ΔE nor Canny edges indicate change above their thresholds.",
      insufficient_edge_evidence: "There were too few edge pixels to score agreement. The Lab result stands alone.",
    };
    $("#method-note").textContent =
      (notes[caseData.agreement.summary] || "Agreement could not be summarised.") +
      " SSIM is reported separately; it is never averaged into a single score.";
  }

  // ---------------------------------------------------------------- verify

  function methodCard(title, stateValue, rows) {
    const tone = methodState(stateValue);
    const label =
      tone === "change" ? "change indicated" : tone === "nochange" ? "no change indicated" : "not scored";
    return el(
      "article",
      { class: "method-card", dataset: { state: tone } },
      el("h3", { text: title }),
      el("div", { class: "method-status", text: label }),
      el(
        "div",
        { class: "kv-grid compact" },
        ...rows.map(([key, value]) => kv(key, value)),
      ),
    );
  }

  function renderVerify() {
    const caseData = currentCase();
    const grid = $("#verify-methods");
    clear(grid);
    grid.append(
      methodCard("Lab ΔE difference", caseData.methods.lab_difference_change_indicated, [
        ["threshold", number(caseData.thresholds.lab_delta_final, 2)],
        ["changed", percent(caseData.changed_percentage)],
        ["regions", String(caseData.region_count)],
        ["valid pixels", (caseData.valid_pixels || 0).toLocaleString()],
      ]),
    );
    grid.append(
      methodCard("Canny edge agreement", caseData.methods.canny_change_indicated, [
        ["status", caseData.edge.status],
        ["similarity", caseData.edge.similarity_percentage === null ? "—" : percent(caseData.edge.similarity_percentage)],
        ["tolerance", `${caseData.thresholds.edge_matching_tolerance_pixels} px`],
        ["edge pixels", `${caseData.edge.reference_edge_pixels} / ${caseData.edge.test_edge_pixels}`],
      ]),
    );
    grid.append(
      methodCard("SSIM (local implementation)", caseData.methods.ssim_change_indicated, [
        ["status", caseData.ssim.status],
        ["mean ssim", caseData.ssim.mean_ssim === null ? "—" : number(caseData.ssim.mean_ssim, 4)],
        ["below threshold", caseData.ssim.changed_percentage === null ? "—" : percent(caseData.ssim.changed_percentage)],
        ["threshold", caseData.ssim.threshold === null ? "—" : number(caseData.ssim.threshold, 2)],
      ]),
    );

    const alert = $("#disagreement");
    if (caseData.disagree) {
      alert.hidden = false;
      alert.textContent =
        "The Lab difference, Canny edge and SSIM methods do not all agree. Method disagreement is kept visible rather than averaged away — inspect each map before drawing a conclusion.";
    } else {
      alert.hidden = true;
    }

    const trace = $("#agent-trace");
    clear(trace);
    for (const iteration of caseData.iterations) {
      trace.append(
        el(
          "li",
          {},
          el("span", { class: "trace-index", text: String(iteration.index).padStart(2, "0") }),
          el(
            "div",
            {},
            el("div", { class: "trace-action", text: iteration.action }),
            el("div", { class: "trace-outcome", text: iteration.outcome }),
            el(
              "div",
              { class: "trace-sub" },
              ...iteration.evidence.map((line) =>
                el("span", {}, el("b", { text: "evidence · " }), line),
              ),
              el("span", {}, el("b", { text: "uncertainty · " }), el("em", { text: iteration.uncertainty })),
            ),
          ),
        ),
      );
    }

    const fillList = (selector, items, empty) => {
      const list = $(selector);
      clear(list);
      if (!items.length) {
        list.append(el("li", { class: "is-empty", text: empty }));
        return;
      }
      for (const item of items) list.append(el("li", { text: item }));
    };
    fillList("#verify-decisions", caseData.decisions, "No explicit decisions were recorded for this pair.");
    fillList("#verify-warnings", caseData.warnings, "No additional uncertainty warnings were recorded.");
    fillList("#verify-limitations", caseData.limitations, "No extra limitations were recorded.");

    const functionTrace = $("#function-trace");
    clear(functionTrace);
    for (const entry of caseData.function_trace) {
      const funcs = el("div", { class: "funcs" });
      for (const name of entry.functions) funcs.append(el("span", { class: "func", text: name }));
      functionTrace.append(
        el(
          "div",
          { class: "trace-row" },
          el("span", { class: "stage", text: entry.stage.replaceAll("_", " ") }),
          funcs,
        ),
      );
    }

    const download = $("#report-download");
    download.href = caseData.report_url;
    download.setAttribute("download", `inspectai-${caseData.id}.json`);
    $("#report-open").href = caseData.report_url;
  }

  // ------------------------------------------------------------------ tabs

  function selectTab(tab, { updateUrl = true } = {}) {
    if (!TABS.includes(tab)) return;
    stopPlayback();
    state.tab = tab;
    for (const button of $$(".tab")) {
      const selected = button.dataset.tab === tab;
      button.setAttribute("aria-selected", selected ? "true" : "false");
      button.setAttribute("tabindex", selected ? "0" : "-1");
    }
    for (const panel of $$(".panel")) panel.hidden = panel.id !== `panel-${tab}`;
    if (updateUrl) writeUrl();
  }

  function renderCase() {
    renderHead();
    renderOverview();
    renderVerify();
    renderRegister();
    renderAnalyze();
    selectTab(state.tab, { updateUrl: false });
  }

  // ------------------------------------------------------------------ boot

  function bindEvents() {
    for (const button of $$(".tab")) {
      button.addEventListener("click", () => selectTab(button.dataset.tab));
    }
    $(".tabs").addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const index = TABS.indexOf(state.tab);
      const next = (index + (event.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length;
      selectTab(TABS[next]);
      $(`#tab-${TABS[next]}`).focus();
    });

    $("#step-prev").addEventListener("click", () => {
      stopPlayback();
      selectStep(Math.max(0, state.step - 1));
    });
    $("#step-next").addEventListener("click", () => {
      stopPlayback();
      selectStep(Math.min(currentCase().steps.length - 1, state.step + 1));
    });
    $("#step-play").addEventListener("click", togglePlayback);
    $("#step-range").addEventListener("input", (event) => {
      stopPlayback();
      selectStep(Number(event.target.value));
    });

    for (const button of $$("#layer-tabs .chip-btn")) {
      button.addEventListener("click", () => {
        state.layer = button.dataset.layer;
        state.region = -1;
        renderAnalyze();
        writeUrl();
      });
    }
    $("#wipe").addEventListener("input", () => {
      applyWipe();
      writeUrl();
    });

    $("#stage-boxes").addEventListener("click", (event) => {
      if (event.target.tagName !== "rect") return;
      const index = Array.from(event.currentTarget.querySelectorAll("rect")).indexOf(event.target);
      if (index >= 0) selectRegion(index);
    });

    document.addEventListener("keydown", (event) => {
      const target = event.target;
      const typing = target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName);
      if (typing) return;
      if (state.tab === "register" && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        event.preventDefault();
        stopPlayback();
        const delta = event.key === "ArrowRight" ? 1 : -1;
        selectStep(Math.min(currentCase().steps.length - 1, Math.max(0, state.step + delta)));
      }
    });

    REDUCED.addEventListener?.("change", () => {
      if (state.tab === "register") renderRegister();
    });
  }

  function init() {
    if (!DATA || !Array.isArray(DATA.cases) || !DATA.cases.length) {
      $("#load-error").hidden = false;
      $("#workspace").hidden = true;
      return;
    }
    readUrl();
    state.index = Math.min(state.index, DATA.cases.length - 1);
    const meta = DATA.meta || {};
    $("#build-meta").textContent =
      `SCHEMA 2 · ${DATA.cases.length} CASES · ${meta.opencv ? `OPENCV ${meta.opencv}` : ""}`;
    $("#footer-meta").textContent =
      `generated ${meta.generated_at_utc || "—"} · Python ${meta.python || "—"} · OpenCV ${meta.opencv || "—"} · NumPy ${meta.numpy || "—"}`;

    renderCaseList();
    renderCase();
    bindEvents();
    $("#workspace").hidden = false;
    $("#load-error").hidden = true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
