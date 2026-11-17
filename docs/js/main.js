/* ==========================================================================
   ES 335 Project Report — Client Module
   Pure ES-module JavaScript, hand-crafted SVG charts, zero external CDNs.
   ========================================================================== */

const COLORS = {
  bg: '#F7F8FA',
  surface: '#FFFFFF',
  ink: '#1D2733',
  muted: '#4A5568',
  rule: '#D8DEE9',
  dev: '#4F6788',
  test: '#C98A00',
  testInk: '#7D5300',
  model: '#0B7285',
  fixed: '#7A828E',
  fixedInk: '#4F5661',
  fail: '#B93A2B',
};

const SVG_NS = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, text = '') {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    node.setAttribute(k, String(v));
  }
  if (text !== '') node.textContent = text;
  return node;
}

function createSvg(width, height, ariaLabel) {
  const svg = el('svg', {
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-label': ariaLabel,
  });
  return svg;
}

function showSvgHost(hostId, containerEl) {
  const host = document.getElementById(hostId);
  if (!host) return;
  host.hidden = false;
  const parentFig = host.closest('figure');
  if (parentFig) {
    const fallback = parentFig.querySelector('.chart-fallback');
    if (fallback) fallback.hidden = true;
  }
}

function niceTicks(min, max, count = 5) {
  if (max === min) return [min];
  const step = (max - min) / Math.max(1, count - 1);
  const ticks = [];
  for (let i = 0; i < count; i++) {
    ticks.push(min + i * step);
  }
  return ticks;
}

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return (val) => rangeMin + ((val - domainMin) / span) * (rangeMax - rangeMin);
}

function buildLinePath(xs, ys, xMap, yMap) {
  return xs
    .map((x, i) => `${i === 0 ? 'M' : 'L'}${xMap(x).toFixed(1)},${yMap(ys[i]).toFixed(1)}`)
    .join(' ');
}

function buildAreaPath(xs, yLow, yHigh, xMap, yMap) {
  const top = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${xMap(x).toFixed(1)},${yMap(yHigh[i]).toFixed(1)}`).join(' ');
  const bot = xs
    .slice()
    .reverse()
    .map((x, revIdx) => {
      const i = xs.length - 1 - revIdx;
      return `L${xMap(x).toFixed(1)},${yMap(yLow[i]).toFixed(1)}`;
    })
    .join(' ');
  return `${top} ${bot} Z`;
}

function drawAxes(svg, { xMap, yMap, xMin, xMax, yMin, yMax, xTicks, yTicks, xLabel, yLabel, xFmt, yFmt }) {
  const g = el('g');
  const y0 = yMap(yMin);
  const y1 = yMap(yMax);
  const x0 = xMap(xMin);
  const x1 = xMap(xMax);

  // Horizontal minimal gridlines
  yTicks.forEach((t) => {
    const py = yMap(t);
    g.appendChild(el('line', { x1: x0, x2: x1, y1: py, y2: py, stroke: COLORS.rule, 'stroke-width': 0.8 }));
    g.appendChild(
      el('text', {
        x: x0 - 7,
        y: py + 4,
        'text-anchor': 'end',
        fill: COLORS.muted,
        'font-size': 11.8,
      }, yFmt ? yFmt(t) : String(Math.round(t)))
    );
  });

  // Thin X and Y axes (no outer chart border)
  g.appendChild(el('line', { x1: x0, x2: x1, y1: y0, y2: y0, stroke: COLORS.muted, 'stroke-width': 1 }));
  g.appendChild(el('line', { x1: x0, x2: x0, y1: y0, y2: y1, stroke: COLORS.muted, 'stroke-width': 1 }));

  xTicks.forEach((t) => {
    const px = xMap(t);
    g.appendChild(el('line', { x1: px, x2: px, y1: y0, y2: y0 + 4, stroke: COLORS.muted, 'stroke-width': 1 }));
    g.appendChild(
      el('text', {
        x: px,
        y: y0 + 17,
        'text-anchor': 'middle',
        fill: COLORS.muted,
        'font-size': 11.8,
      }, xFmt ? xFmt(t) : String(Math.round(t)))
    );
  });

  if (xLabel) {
    g.appendChild(
      el('text', {
        x: (x0 + x1) / 2,
        y: y0 + 34,
        'text-anchor': 'middle',
        fill: COLORS.ink,
        'font-size': 12.5,
        'font-weight': 600,
      }, xLabel)
    );
  }

  if (yLabel) {
    const maxTickLen = yTicks.reduce((acc, t) => {
      const s = yFmt ? yFmt(t) : String(Math.round(t));
      return Math.max(acc, s.length);
    }, 1);
    const labelOffset = Math.max(40, maxTickLen * 6.8 + 16);
    const lx = Math.max(12, x0 - labelOffset);
    const ly = (y0 + y1) / 2;
    g.appendChild(
      el('text', {
        x: lx,
        y: ly,
        'text-anchor': 'middle',
        fill: COLORS.ink,
        'font-size': 12.5,
        'font-weight': 600,
        transform: `rotate(-90 ${lx} ${ly})`,
      }, yLabel)
    );
  }

  svg.appendChild(g);
}

/* ==========================================================================
   Figure 1: Fleet Lifetime Distribution
   ========================================================================== */
function renderFigure1(data) {
  const host = document.getElementById('fig1-svg-host');
  if (!host) return;

  const devLife = data.meta.dev_lifetimes;
  const testLife = data.meta.test_lifetimes;
  const bins = [];
  for (let start = 120; start < 380; start += 20) {
    bins.push({
      start,
      end: start + 20,
      dev: devLife.filter((v) => v >= start && v < start + 20).length / devLife.length * 100,
      test: testLife.filter((v) => v >= start && v < start + 20).length / testLife.length * 100,
    });
  }

  const W = 760;
  const H = 260;
  const pad = { l: 58, r: 22, t: 22, b: 42 };
  const svg = createSvg(W, H, 'Fleet lifetime percentage distribution across development and held-out engines');

  const xMap = scaleLinear(120, 380, pad.l, W - pad.r);
  const yMap = scaleLinear(0, 35, H - pad.b, pad.t);

  drawAxes(svg, {
    xMap,
    yMap,
    xMin: 120,
    xMax: 380,
    yMin: 0,
    yMax: 35,
    xTicks: [120, 160, 200, 240, 280, 320, 360],
    yTicks: [0, 10, 20, 30],
    xLabel: 'Engine lifetime until failure (cycles)',
    yLabel: 'Fleet share (%)',
    yFmt: (v) => `${v}%`,
  });

  bins.forEach((b) => {
    const bw = xMap(b.end) - xMap(b.start);
    const barW = bw * 0.42;
    const devH = yMap(0) - yMap(b.dev);
    const testH = yMap(0) - yMap(b.test);

    if (b.dev > 0) {
      svg.appendChild(el('rect', {
        x: xMap(b.start) + 2,
        y: yMap(b.dev),
        width: barW,
        height: devH,
        fill: COLORS.dev,
        opacity: 0.85,
      }));
    }
    if (b.test > 0) {
      svg.appendChild(el('rect', {
        x: xMap(b.start) + 2 + barW + 1,
        y: yMap(b.test),
        width: barW,
        height: testH,
        fill: COLORS.test,
        opacity: 0.88,
      }));
    }
  });

  // Fixed K*=179 vertical line
  const xK = xMap(179);
  svg.appendChild(el('line', {
    x1: xK,
    x2: xK,
    y1: pad.t,
    y2: H - pad.b,
    stroke: COLORS.fixed,
    'stroke-width': 2,
    'stroke-dasharray': '5,4',
  }));
  svg.appendChild(el('text', {
    x: xK + 5,
    y: pad.t + 13,
    fill: COLORS.fixedInk,
    'font-size': 11.8,
    'font-weight': 600,
  }, 'Fixed schedule K* = 179'));

  // Direct group labels
  svg.appendChild(el('text', { x: xMap(250), y: pad.t + 14, fill: COLORS.dev, 'font-size': 12, 'font-weight': 600 },
    '■ 80 Development engines (128–362 cyc, median 199)'));
  svg.appendChild(el('text', { x: xMap(250), y: pad.t + 31, fill: COLORS.testInk, 'font-size': 12, 'font-weight': 600 },
    '■ 20 Held-out test engines (147–341 cyc, median 198.5)'));

  host.innerHTML = '';
  host.appendChild(svg);
  showSvgHost('fig1-svg-host', host);
}

/* ==========================================================================
   Figure 2: Failure-Aligned Trajectories & PCA Health Index
   ========================================================================== */
function renderFigure2(data) {
  const host = document.getElementById('fig2-svg-host');
  const aligned = data.aligned_sensor_summary;
  if (!host || !aligned) return;

  const W = 780;
  const H = 265;
  const svg = createSvg(W, H, 'Failure-aligned Sensor 11 and 1D PCA Health Index');

  const xs = aligned.t_to_fail_bins;
  const s11 = aligned.series.sensor_11;
  const pc1 = aligned.series.pc1;

  // Panel A: Sensor 11 aligned to failure
  const padA = { l: 58, r: 422, t: 28, b: 44 };
  const xA = scaleLinear(-200, 0, padA.l, W - padA.r);
  const yA = scaleLinear(47.0, 48.4, H - padA.b, padA.t);

  drawAxes(svg, {
    xMap: xA,
    yMap: yA,
    xMin: -200,
    xMax: 0,
    yMin: 47.0,
    yMax: 48.4,
    xTicks: [-200, -150, -125, -80, -30, 0],
    yTicks: [47.0, 47.4, 47.8, 48.2],
    xLabel: 'Cycles relative to failure (t - T_e)',
    yLabel: 'Sensor 11 (psia)',
    yFmt: (v) => v.toFixed(1),
  });

  const s11Low = s11.mean.map((m, i) => m - s11.std[i]);
  const s11High = s11.mean.map((m, i) => m + s11.std[i]);
  svg.appendChild(el('path', { d: buildAreaPath(xs, s11Low, s11High, xA, yA), fill: COLORS.dev, opacity: 0.16 }));
  svg.appendChild(el('path', { d: buildLinePath(xs, s11.mean, xA, yA), fill: 'none', stroke: COLORS.dev, 'stroke-width': 2.2 }));

  svg.appendChild(el('line', { x1: xA(-125), x2: xA(-125), y1: padA.t, y2: H - padA.b, stroke: COLORS.fixed, 'stroke-dasharray': '4,3', 'stroke-width': 1.2 }));
  svg.appendChild(el('text', { x: xA(-122), y: padA.t + 13, fill: COLORS.fixedInk, 'font-size': 11.5 }, 'Cap onset (-125)'));
  svg.appendChild(el('text', { x: padA.l + 4, y: padA.t - 8, fill: COLORS.ink, 'font-size': 12.2, 'font-weight': 600 },
    'A. Sensor 11 mean ± 1 SD (80 dev engines)'));

  // Panel B: 1D PCA Health Index
  const padB = { l: 452, r: 22, t: 28, b: 44 };
  const xB = scaleLinear(-200, 0, padB.l, W - padB.r);
  const yB = scaleLinear(-6.5, 4.0, H - padB.b, padB.t);

  drawAxes(svg, {
    xMap: xB,
    yMap: yB,
    xMin: -200,
    xMax: 0,
    yMin: -6.5,
    yMax: 4.0,
    xTicks: [-200, -150, -100, -30, 0],
    yTicks: [-6, -3, 0, 3],
    xLabel: 'Cycles relative to failure (t - T_e)',
    yLabel: 'PC1 Health Index',
    yFmt: (v) => v.toFixed(0),
  });

  const pcLow = pc1.mean.map((m, i) => m - pc1.std[i]);
  const pcHigh = pc1.mean.map((m, i) => m + pc1.std[i]);
  svg.appendChild(el('path', { d: buildAreaPath(xs, pcLow, pcHigh, xB, yB), fill: COLORS.model, opacity: 0.16 }));
  svg.appendChild(el('path', { d: buildLinePath(xs, pc1.mean, xB, yB), fill: 'none', stroke: COLORS.model, 'stroke-width': 2.2 }));

  svg.appendChild(el('line', { x1: xB(-30), x2: xB(-30), y1: padB.t, y2: H - padB.b, stroke: COLORS.fail, 'stroke-dasharray': '4,3', 'stroke-width': 1.2 }));
  svg.appendChild(el('text', { x: xB(-34), y: padB.t + 14, 'text-anchor': 'end', fill: COLORS.fail, 'font-size': 11.5, 'font-weight': 600 }, 'Alarm window (w=30)'));
  svg.appendChild(el('text', { x: padB.l + 4, y: padB.t - 8, fill: COLORS.ink, 'font-size': 12.2, 'font-weight': 600 },
    `B. 1D PCA Health Index (r = +${aligned.pca_info.corr_pc1_with_rul_capped})`));

  host.innerHTML = '';
  host.appendChild(svg);
  showSvgHost('fig2-svg-host', host);
}

/* ==========================================================================
   Figure 3: Trailing Window Denoising & OLS Slope
   ========================================================================== */
function renderFigure3(data) {
  const host = document.getElementById('fig3-svg-host');
  const eng = data.sample_dev_engine;
  if (!host || !eng) return;

  const W = 780;
  const H = 250;
  const svg = createSvg(W, H, 'Raw Sensor 11 versus trailing 30-cycle mean and OLS drift slope on dev Engine 1');

  const cyc = eng.cycles;
  const raw = eng.sensor_11_raw;
  const mean30 = eng.sensor_11_mean_k30;
  const slope30 = eng.sensor_11_slope_k30;

  // Panel A: Raw vs Trailing Mean
  const padA = { l: 58, r: 422, t: 28, b: 44 };
  const xA = scaleLinear(1, eng.max_cycle, padA.l, W - padA.r);
  const yA = scaleLinear(47.0, 48.3, H - padA.b, padA.t);

  drawAxes(svg, {
    xMap: xA,
    yMap: yA,
    xMin: 1,
    xMax: eng.max_cycle,
    yMin: 47.0,
    yMax: 48.3,
    xTicks: [1, 50, 100, 150, 192],
    yTicks: [47.0, 47.4, 47.8, 48.2],
    xLabel: 'Operating cycle t',
    yLabel: 'Sensor 11 (psia)',
    yFmt: (v) => v.toFixed(1),
  });

  svg.appendChild(el('path', { d: buildLinePath(cyc, raw, xA, yA), fill: 'none', stroke: COLORS.fixed, 'stroke-width': 1, opacity: 0.75 }));
  svg.appendChild(el('path', { d: buildLinePath(cyc, mean30, xA, yA), fill: 'none', stroke: COLORS.model, 'stroke-width': 2.4 }));
  svg.appendChild(el('text', { x: xA(20), y: yA(47.97), fill: COLORS.fixedInk, 'font-size': 11.5 }, 'Raw cycle reading'));
  svg.appendChild(el('text', { x: xA(20), y: yA(47.81), fill: COLORS.model, 'font-size': 11.8, 'font-weight': 600 }, 'Trailing mean (k=30)'));
  svg.appendChild(el('text', { x: padA.l + 4, y: padA.t - 8, fill: COLORS.ink, 'font-size': 12.2, 'font-weight': 600 },
    'A. Trailing mean suppresses cycle noise'));

  // Panel B: Trailing OLS Slope
  const padB = { l: 452, r: 22, t: 28, b: 44 };
  const xB = scaleLinear(1, eng.max_cycle, padB.l, W - padB.r);
  const yB = scaleLinear(-0.015, 0.030, H - padB.b, padB.t);

  drawAxes(svg, {
    xMap: xB,
    yMap: yB,
    xMin: 1,
    xMax: eng.max_cycle,
    yMin: -0.015,
    yMax: 0.030,
    xTicks: [1, 50, 100, 150, 192],
    yTicks: [-0.01, 0, 0.01, 0.02, 0.03],
    xLabel: 'Operating cycle t',
    yLabel: 'OLS slope (psia/cyc)',
    yFmt: (v) => v.toFixed(2),
  });

  svg.appendChild(el('line', { x1: xB(1), x2: xB(eng.max_cycle), y1: yB(0), y2: yB(0), stroke: COLORS.fixed, 'stroke-dasharray': '3,3', 'stroke-width': 1 }));
  svg.appendChild(el('path', { d: buildLinePath(cyc, slope30, xB, yB), fill: 'none', stroke: COLORS.dev, 'stroke-width': 2.2 }));
  svg.appendChild(el('text', { x: xB(148), y: yB(0.022), 'text-anchor': 'end', fill: COLORS.dev, 'font-size': 11.5, 'font-weight': 600 },
    'Steep upward wear drift'));
  svg.appendChild(el('text', { x: padB.l + 4, y: padB.t - 8, fill: COLORS.ink, 'font-size': 12.2, 'font-weight': 600 },
    'B. Stabilized trailing OLS drift slope'));

  host.innerHTML = '';
  host.appendChild(svg);
  showSvgHost('fig3-svg-host', host);
}

/* ==========================================================================
   Figure 4: Split Architecture Diagram
   ========================================================================== */
function renderFigure4() {
  const host = document.getElementById('fig4-svg-host');
  if (!host) return;

  const W = 780;
  const H = 272;
  const svg = createSvg(W, H, 'Engine-grouped split architecture: 80 development engines vs 20 sealed held-out test engines');

  function box(x, y, w, h, title, lines, stroke, fill) {
    const g = el('g');
    g.appendChild(el('rect', { x, y, width: w, height: h, rx: 3, ry: 3, fill, stroke, 'stroke-width': 1.6 }));
    g.appendChild(el('text', { x: x + 12, y: y + 20, fill: stroke === COLORS.test ? COLORS.testInk : stroke, 'font-size': 12.5, 'font-weight': 700 }, title));
    lines.forEach((line, idx) => {
      g.appendChild(el('text', { x: x + 12, y: y + 40 + idx * 17, fill: COLORS.ink, 'font-size': 11.5 }, line));
    });
    svg.appendChild(g);
  }

  // Raw training file box
  box(10, 18, 200, 118, 'train_FD001.txt', [
    '100 run-to-failure engines',
    '20,631 rows total',
    'Partitioned by unit_id',
    'Fixed random seed = 42',
  ], COLORS.ink, COLORS.surface);

  // Dev box
  box(250, 18, 255, 118, '80 Development Engines', [
    '16,340 rows | 5-fold GroupKFold',
    'Sensor screening & Pipeline scaling',
    'Model comparison (Rungs 0–5)',
    'Locks K*=179, p*=0.99, tau*=9.0',
  ], COLORS.dev, '#EEF2F7');

  // Held-out test box
  box(545, 18, 225, 118, '20 Held-Out Test Engines', [
    '4,291 rows | Sealed until final test',
    'Evaluated ONCE with locked rules',
    'Full-lifetime policy simulation',
    '1,000-engine bootstrap 95% CIs',
  ], COLORS.test, '#FDF7E7');

  // Official truncated test box
  box(250, 168, 520, 90, 'Secondary Check: Official test_FD001.txt (100 Truncated Engines, 13,096 rows)', [
    'Models fitted on all 80 development engines are checked once at each',
    'engine’s final cutoff cycle against official RUL_FD001.txt targets.',
    'Not used for cost-policy simulation (trajectories end before failure).',
  ], COLORS.fixed, COLORS.surface);

  // Arrows
  svg.appendChild(el('line', { x1: 210, x2: 245, y1: 77, y2: 77, stroke: COLORS.dev, 'stroke-width': 2 }));
  svg.appendChild(el('polygon', { points: '245,72 252,77 245,82', fill: COLORS.dev }));

  svg.appendChild(el('line', { x1: 505, x2: 540, y1: 77, y2: 77, stroke: COLORS.test, 'stroke-width': 2, 'stroke-dasharray': '4,3' }));
  svg.appendChild(el('polygon', { points: '540,72 547,77 540,82', fill: COLORS.test }));

  svg.appendChild(el('line', { x1: 377, x2: 377, y1: 136, y2: 163, stroke: COLORS.fixed, 'stroke-width': 1.5, 'stroke-dasharray': '3,3' }));
  svg.appendChild(el('polygon', { points: '372,163 377,169 382,163', fill: COLORS.fixed }));

  host.innerHTML = '';
  host.appendChild(svg);
  showSvgHost('fig4-svg-host', host);
}

/* ==========================================================================
   Main Interactive Explorer (Figure 5 — Visual Focal Point)
   ========================================================================== */
function initMainExplorer(data) {
  const explorer = data.interactive_explorer;
  if (!explorer) return;

  const modelSel = document.getElementById('exp-model-select');
  const ratioSlider = document.getElementById('exp-ratio-slider');
  const ratioVal = document.getElementById('exp-ratio-val');
  const threshSlider = document.getElementById('exp-thresh-slider');
  const threshLabel = document.getElementById('exp-thresh-label');
  const threshVal = document.getElementById('exp-thresh-val');
  const lockDevBtn = document.getElementById('exp-lock-dev-btn');
  const resetBtn = document.getElementById('exp-reset-btn');
  const takeawayEl = document.getElementById('exp-live-takeaway');

  const roModelCost = document.getElementById('ro-model-cost');
  const roFixedCost = document.getElementById('ro-fixed-cost');
  const roFailures = document.getElementById('ro-failures');

  const hostA = document.getElementById('fig5a-host');
  const hostB = document.getElementById('fig5b-host');
  const parentHost = document.getElementById('fig5-svg-host');

  const state = {
    modelId: 'rung4_trends_logreg',
    ratio: 100,
    threshIdx: 93, // 0.99 for prob models
    lockedToDev: true,
  };

  function bestDevIndexForRatio(model, ratio) {
    let bestIdx = 0;
    let bestCost = Infinity;
    for (let i = 0; i < model.thresholds.length; i++) {
      const c = Math.round(((model.dev_failures[i] * ratio + model.dev_wasted_sum[i]) / 80) * 100) / 100;
      if (c < bestCost) {
        bestCost = c;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function bestTestHindsightIndexForRatio(model, ratio) {
    let bestIdx = 0;
    let bestCost = Infinity;
    for (let i = 0; i < model.thresholds.length; i++) {
      const c = Math.round(((model.test_failures[i] * ratio + model.test_wasted_sum[i]) / 20) * 100) / 100;
      if (c < bestCost) {
        bestCost = c;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function bestFixedScheduleForRatio(ratio) {
    const fs = explorer.fixed_schedule;
    let bestIdx = 0;
    let bestCost = Infinity;
    for (let i = 0; i < fs.dev.k_values.length; i++) {
      const c = Math.round(((fs.dev.failures[i] * ratio + fs.dev.wasted_sum[i]) / 80) * 100) / 100;
      if (c < bestCost) {
        bestCost = c;
        bestIdx = i;
      }
    }
    return {
      kStar: fs.dev.k_values[bestIdx],
      devCost: (fs.dev.failures[bestIdx] * ratio + fs.dev.wasted_sum[bestIdx]) / 80,
      testCost: (fs.test.failures[bestIdx] * ratio + fs.test.wasted_sum[bestIdx]) / 20,
      testFail: fs.test.failures[bestIdx],
      testWasted: fs.test.wasted_sum[bestIdx] / 20,
    };
  }

  function syncSliderRange() {
    const model = explorer.models[state.modelId];
    threshSlider.min = '0';
    threshSlider.max = String(model.thresholds.length - 1);
    threshLabel.innerHTML = model.score_is_rul ? 'RUL alarm trigger (&tau; cycles)' : 'Alarm probability threshold (<em>p</em>)';
    if (state.lockedToDev) {
      state.threshIdx = bestDevIndexForRatio(model, state.ratio);
    }
    state.threshIdx = Math.max(0, Math.min(model.thresholds.length - 1, state.threshIdx));
    threshSlider.value = String(state.threshIdx);
  }

  function renderExplorer() {
    const model = explorer.models[state.modelId];
    const r = state.ratio;
    const idx = state.threshIdx;
    const thVal = model.thresholds[idx];
    const devOptIdx = bestDevIndexForRatio(model, r);
    const devOptVal = model.thresholds[devOptIdx];
    const testHindsightIdx = bestTestHindsightIndexForRatio(model, r);
    const testHindsightVal = model.thresholds[testHindsightIdx];
    const isDevLocked = idx === devOptIdx;

    const fixed = bestFixedScheduleForRatio(r);
    const modelTestCost = (model.test_failures[idx] * r + model.test_wasted_sum[idx]) / 20;
    const modelDevCost = (model.dev_failures[idx] * r + model.dev_wasted_sum[idx]) / 80;
    const testFail = model.test_failures[idx];
    const savings = fixed.testCost - modelTestCost;
    const savingsPct = fixed.testCost > 0 ? (savings / fixed.testCost) * 100 : 0;

    ratioVal.textContent = `${r} : 1`;
    threshVal.textContent = model.score_is_rul ? `<= ${thVal.toFixed(1)} cyc` : `>= ${thVal.toFixed(2)}`;
    if (isDevLocked) {
      threshVal.textContent += ' (dev-locked)';
    } else {
      threshVal.textContent += ' (manual exploratory)';
    }

    roModelCost.textContent = modelTestCost.toFixed(2);
    roFixedCost.textContent = fixed.testCost.toFixed(2);
    roFailures.textContent = `${testFail} / 20 (${(testFail * 5).toFixed(0)}%)`;

    const thText = model.score_is_rul ? `RUL &le; <strong>${thVal.toFixed(1)}</strong>` : `<em>p</em> &ge; <strong>${thVal.toFixed(2)}</strong>`;
    const devOptText = model.score_is_rul ? `&tau;*_dev = <strong>${devOptVal.toFixed(1)}</strong>` : `<em>p</em>*_dev = <strong>${devOptVal.toFixed(2)}</strong>`;
    const lockNote = isDevLocked
      ? `dev-locked (${devOptText})`
      : `manual ${thText} (dev-locked: ${devOptText})`;

    takeawayEl.innerHTML =
      `At <strong>${r}:1</strong> (${lockNote}): <strong>${model.label}</strong> costs <strong>${modelTestCost.toFixed(2)}</strong> vs. <strong>${fixed.testCost.toFixed(2)}</strong> fixed (<strong>${savings >= 0 ? `saves ${savingsPct.toFixed(1)}%` : `costs ${Math.abs(savings).toFixed(2)} more`}</strong>).`;

    // ---- Panel A: Cost Curve across ratios 10..300 ----
    const ratios = [10, 20, 30, 50, 75, 100, 150, 200, 300];
    const fixSeries = [];
    const modTestSeries = [];
    const modDevSeries = [];
    ratios.forEach((rr) => {
      fixSeries.push(bestFixedScheduleForRatio(rr).testCost);
      const useIdx = state.lockedToDev ? bestDevIndexForRatio(model, rr) : idx;
      modTestSeries.push((model.test_failures[useIdx] * rr + model.test_wasted_sum[useIdx]) / 20);
      modDevSeries.push((model.dev_failures[useIdx] * rr + model.dev_wasted_sum[useIdx]) / 80);
    });

    const maxY = Math.max(90, ...fixSeries, ...modTestSeries) * 1.1;
    const WA = 420;
    const HA = 285;
    const padA = { l: 56, r: 24, t: 30, b: 42 };
    const svgA = createSvg(WA, HA, 'Cost curve across failure-to-waste ratios');
    const xA = scaleLinear(10, 300, padA.l, WA - padA.r);
    const yA = scaleLinear(0, maxY, HA - padA.b, padA.t);

    drawAxes(svgA, {
      xMap: xA,
      yMap: yA,
      xMin: 10,
      xMax: 300,
      yMin: 0,
      yMax: maxY,
      xTicks: [10, 50, 100, 150, 200, 250, 300],
      yTicks: niceTicks(0, Math.ceil(maxY / 20) * 20, 5),
      xLabel: 'Failure-to-waste cost ratio (c_fail : c_waste)',
      yLabel: 'Mean cost / engine',
    });

    svgA.appendChild(el('path', {
      d: buildLinePath(ratios, fixSeries, xA, yA),
      fill: 'none',
      stroke: COLORS.fixed,
      'stroke-width': 2.1,
      'stroke-dasharray': '6,4',
    }));
    svgA.appendChild(el('path', {
      d: buildLinePath(ratios, modDevSeries, xA, yA),
      fill: 'none',
      stroke: COLORS.dev,
      'stroke-width': 1.7,
      'stroke-dasharray': '2,3',
    }));
    svgA.appendChild(el('path', {
      d: buildLinePath(ratios, modTestSeries, xA, yA),
      fill: 'none',
      stroke: COLORS.model,
      'stroke-width': 2.6,
    }));

    // Selected ratio vertical marker
    svgA.appendChild(el('line', {
      x1: xA(r),
      x2: xA(r),
      y1: padA.t,
      y2: HA - padA.b,
      stroke: COLORS.test,
      'stroke-width': 1.5,
      'stroke-dasharray': '3,3',
    }));
    svgA.appendChild(el('circle', { cx: xA(r), cy: yA(modelTestCost), r: 4.5, fill: COLORS.model, stroke: '#FFFFFF', 'stroke-width': 1.4 }));
    svgA.appendChild(el('circle', { cx: xA(r), cy: yA(fixed.testCost), r: 4.2, fill: COLORS.fixed, stroke: '#FFFFFF', 'stroke-width': 1.4 }));

    // Top key strip + collision-clamped direct labels
    const keyStrip = el('text', { x: padA.l, y: 14, 'font-size': 11.2, 'font-weight': 600 });
    keyStrip.appendChild(el('tspan', { fill: COLORS.fixedInk }, 'Fixed schedule (test)'));
    keyStrip.appendChild(el('tspan', { fill: COLORS.muted }, ' · '));
    keyStrip.appendChild(el('tspan', { fill: COLORS.model }, 'Model policy (test)'));
    keyStrip.appendChild(el('tspan', { fill: COLORS.muted }, ' · '));
    keyStrip.appendChild(el('tspan', { fill: COLORS.dev }, 'Model policy (dev CV)'));
    svgA.appendChild(keyStrip);

    const yFloor = yA(0) - 8;
    const upperModelY = Math.min(yA(modTestSeries[7]), yA(modDevSeries[7]));
    const lowerModelY = Math.max(yA(modTestSeries[7]), yA(modDevSeries[7]));
    const yModLbl = Math.max(padA.t + 22, Math.min(yFloor - 14, upperModelY - 7));
    let yDevLbl = lowerModelY + 14 <= yFloor ? lowerModelY + 14 : Math.max(padA.t + 10, yModLbl - 14);
    if (Math.abs(yDevLbl - yModLbl) < 14) {
      yDevLbl = Math.max(padA.t + 10, yModLbl - 14);
    }
    const yFixLbl = Math.max(padA.t + 10, Math.min(Math.min(yModLbl, yDevLbl) - 14, yA(fixSeries[7]) - 8));
    const haloAttrs = { stroke: '#FFFFFF', 'stroke-width': 3.5, 'paint-order': 'stroke' };
    svgA.appendChild(el('text', { x: xA(205), y: yFixLbl, fill: COLORS.fixedInk, 'font-size': 11.2, 'font-weight': 600, ...haloAttrs },
      'Fixed (test)'));
    svgA.appendChild(el('text', { x: xA(205), y: yModLbl, fill: COLORS.model, 'font-size': 11.2, 'font-weight': 600, ...haloAttrs },
      'Model (test)'));
    svgA.appendChild(el('text', { x: xA(205), y: yDevLbl, fill: COLORS.dev, 'font-size': 11.2, ...haloAttrs },
      'Model (dev CV)'));

    hostA.innerHTML = '';
    hostA.appendChild(svgA);

    // ---- Panel B: Threshold Curve + 20-Engine Outcome Strip ----
    const WB = 440;
    const HB = 305;
    const svgB = createSvg(WB, HB, 'Threshold sensitivity and per-engine alarm outcomes on 20 held-out test engines');

    // Top half: Threshold vs Cost (Dev vs Test)
    const padBTop = { l: 56, r: 18, t: 18, b: 172 };
    const ths = model.thresholds;
    const thMin = ths[0];
    const thMax = ths[ths.length - 1];
    const devCurve = ths.map((_, i) => (model.dev_failures[i] * r + model.dev_wasted_sum[i]) / 80);
    const testCurve = ths.map((_, i) => (model.test_failures[i] * r + model.test_wasted_sum[i]) / 20);
    const thMaxY = Math.min(r * 1.05, Math.max(40, ...devCurve, ...testCurve) * 1.08);

    const xB = scaleLinear(thMin, thMax, padBTop.l, WB - padBTop.r);
    const yB = scaleLinear(0, thMaxY, HB - padBTop.b, padBTop.t);

    drawAxes(svgB, {
      xMap: xB,
      yMap: yB,
      xMin: thMin,
      xMax: thMax,
      yMin: 0,
      yMax: thMaxY,
      xTicks: niceTicks(thMin, thMax, 5),
      yTicks: niceTicks(0, Math.round(thMaxY / 10) * 10, 4),
      xFmt: (v) => (model.score_is_rul ? v.toFixed(0) : v.toFixed(2)),
      yLabel: 'Cost / eng',
    });

    svgB.appendChild(el('path', { d: buildLinePath(ths, devCurve, xB, yB), fill: 'none', stroke: COLORS.dev, 'stroke-width': 2 }));
    svgB.appendChild(el('path', { d: buildLinePath(ths, testCurve, xB, yB), fill: 'none', stroke: COLORS.test, 'stroke-width': 1.9, 'stroke-dasharray': '4,3' }));

    // Dev-locked marker vs Test hindsight minimum marker
    svgB.appendChild(el('line', {
      x1: xB(devOptVal),
      x2: xB(devOptVal),
      y1: padBTop.t,
      y2: HB - padBTop.b,
      stroke: COLORS.model,
      'stroke-width': 1.7,
    }));
    svgB.appendChild(el('circle', { cx: xB(thVal), cy: yB(modelTestCost), r: 4.5, fill: COLORS.ink }));
    svgB.appendChild(el('polygon', {
      points: `${xB(testHindsightVal)},${yB(testCurve[testHindsightIdx]) - 5} ${xB(testHindsightVal) + 5},${yB(testCurve[testHindsightIdx])} ${xB(testHindsightVal)},${yB(testCurve[testHindsightIdx]) + 5} ${xB(testHindsightVal) - 5},${yB(testCurve[testHindsightIdx])}`,
      fill: 'none',
      stroke: COLORS.testInk,
      'stroke-width': 1.5,
    }));

    svgB.appendChild(el('text', {
      x: padBTop.l + 6,
      y: padBTop.t + 11,
      fill: COLORS.dev,
      'font-size': 11,
      'font-weight': 600,
      'paint-order': 'stroke',
      stroke: '#FFFFFF',
      'stroke-width': 3,
      'stroke-linejoin': 'round',
    }, `Dev CV (${devOptVal})`));
    svgB.appendChild(el('text', {
      x: padBTop.l + 6,
      y: padBTop.t + 25,
      fill: COLORS.testInk,
      'font-size': 11,
      'font-weight': 600,
      'paint-order': 'stroke',
      stroke: '#FFFFFF',
      'stroke-width': 3,
      'stroke-linejoin': 'round',
    }, `◆ Test hindsight opt (${testHindsightVal}, unavailable in practice)`));

    // Bottom half: 20 Held-out Test Engines Timeline
    const padBBot = { l: 56, r: 18, t: 180, b: 28 };
    const xEng = scaleLinear(0, 360, padBBot.l, WB - padBBot.r);
    const enginesSorted = model.test_per_engine_alarms
      .map((e) => ({ unit_id: e.unit_id, Te: e.true_lifetime_Te, alarm: e.alarm_cycles[idx] }))
      .sort((a, b) => a.Te - b.Te);

    // Fixed schedule vertical line
    svgB.appendChild(el('line', {
      x1: xEng(fixed.kStar),
      x2: xEng(fixed.kStar),
      y1: padBBot.t,
      y2: HB - padBBot.b,
      stroke: COLORS.fixed,
      'stroke-width': 1.6,
      'stroke-dasharray': '4,3',
    }));

    const rowH = (HB - padBBot.b - padBBot.t) / enginesSorted.length;
    enginesSorted.forEach((eng, rowIdx) => {
      const cy = padBBot.t + rowIdx * rowH + rowH / 2;
      svgB.appendChild(el('line', {
        x1: xEng(0),
        x2: xEng(eng.Te),
        y1: cy,
        y2: cy,
        stroke: COLORS.rule,
        'stroke-width': 2.2,
      }));
      if (eng.alarm === null) {
        const fx = xEng(eng.Te);
        svgB.appendChild(el('polygon', {
          points: `${fx},${cy - 3.2} ${fx + 3.2},${cy} ${fx},${cy + 3.2} ${fx - 3.2},${cy}`,
          fill: COLORS.fail,
        }));
      } else {
        const ax = xEng(eng.alarm);
        const wasted = eng.Te - eng.alarm;
        if (wasted > 30) {
          svgB.appendChild(el('rect', { x: ax - 2.5, y: cy - 2.5, width: 5, height: 5, fill: COLORS.test }));
        } else {
          svgB.appendChild(el('circle', { cx: ax, cy, r: 2.7, fill: COLORS.model }));
        }
      }
    });

    [0, 100, 179, 250, 341].forEach((tk) => {
      svgB.appendChild(el('text', {
        x: xEng(tk),
        y: HB - 11,
        'text-anchor': 'middle',
        fill: tk === 179 ? COLORS.fixedInk : COLORS.muted,
        'font-size': 11.2,
      }, String(tk)));
    });

    const stripLegend = el('text', { x: padBBot.l, y: padBBot.t - 6, fill: COLORS.ink, 'font-size': 11.2, 'font-weight': 600 });
    stripLegend.appendChild(el('tspan', { fill: COLORS.ink }, '20 Held-out test lifetimes: '));
    stripLegend.appendChild(el('tspan', { fill: COLORS.model }, '● timely (≤30c)  '));
    stripLegend.appendChild(el('tspan', { fill: COLORS.testInk }, '■ early (>30c)  '));
    stripLegend.appendChild(el('tspan', { fill: COLORS.fail }, '◆ failed'));
    svgB.appendChild(stripLegend);

    hostB.innerHTML = '';
    hostB.appendChild(svgB);
  }

  modelSel.addEventListener('change', () => {
    state.modelId = modelSel.value;
    state.lockedToDev = true;
    syncSliderRange();
    renderExplorer();
  });

  ratioSlider.addEventListener('input', () => {
    state.ratio = Number(ratioSlider.value);
    if (state.lockedToDev) syncSliderRange();
    renderExplorer();
  });

  threshSlider.addEventListener('input', () => {
    state.lockedToDev = false;
    state.threshIdx = Number(threshSlider.value);
    renderExplorer();
  });

  lockDevBtn.addEventListener('click', () => {
    state.lockedToDev = true;
    syncSliderRange();
    renderExplorer();
  });

  resetBtn.addEventListener('click', () => {
    state.modelId = 'rung4_trends_logreg';
    state.ratio = 100;
    state.lockedToDev = true;
    modelSel.value = state.modelId;
    ratioSlider.value = '100';
    syncSliderRange();
    renderExplorer();
  });

  syncSliderRange();
  renderExplorer();
  showSvgHost('fig5-svg-host', parentHost);

  return {
    jumpToModel(modelId) {
      if (!explorer.models[modelId]) return;
      state.modelId = modelId;
      state.lockedToDev = true;
      modelSel.value = modelId;
      syncSliderRange();
      renderExplorer();
    },
  };
}

/* ==========================================================================
   Leakage Comparison Widget (Figure 6)
   ========================================================================== */
function initLeakageWidget(data) {
  const host = document.getElementById('fig6-svg-host');
  const leak = data.leakage_experiment;
  if (!host || !leak) return;

  const btnProper = document.getElementById('leak-btn-proper');
  const btnLeaky = document.getElementById('leak-btn-leaky');
  const modelSel = document.getElementById('leak-model-select');
  const resetBtn = document.getElementById('leak-reset-btn');
  const takeawayEl = document.getElementById('leak-live-takeaway');

  let mode = 'proper';
  let modelId = 'rung5_trends_hgb';

  function render() {
    btnProper.setAttribute('aria-pressed', String(mode === 'proper'));
    btnLeaky.setAttribute('aria-pressed', String(mode === 'leaky'));

    const row = leak.summary.find((r) => r.model_id === modelId);
    const traj = leak.scatter_and_trajectory.trajectory_example;

    if (mode === 'proper') {
      takeawayEl.innerHTML =
        `Proper engine-level split holds out all cycles of unseen engines: <strong>${row.model_label}</strong> achieves <strong>${row.proper_heldout_engines_test_rmse} test RMSE</strong> (R&sup2; = <strong>${row.proper_test_r2}</strong>, NASA asym = <strong>${row.proper_test_nasa_asym}</strong>).`;
    } else {
      takeawayEl.innerHTML =
        `Flawed random row split leaks <strong>~80%</strong> of adjacent cycles into training: <strong>${row.model_label}</strong> reports an artificially deflated <strong>${row.leaky_random_row_split_test_rmse} RMSE</strong> — a fake <strong>${row.artificial_error_reduction_pct}% drop</strong> (<strong>${row.artificial_rmse_drop_cycles} cycles</strong>).`;
    }

    const W = 780;
    const H = 260;
    const svg = createSvg(W, H, 'Comparison between proper engine split and leaky random row split');

    // Panel A: Held-out Engine #21 trajectory (Random Forest)
    const padA = { l: 58, r: 422, t: 28, b: 44 };
    const xA = scaleLinear(1, traj.cycles.length, padA.l, W - padA.r);
    const yA = scaleLinear(0, 130, H - padA.b, padA.t);

    drawAxes(svg, {
      xMap: xA,
      yMap: yA,
      xMin: 1,
      xMax: traj.cycles.length,
      yMin: 0,
      yMax: 130,
      xTicks: [1, 50, 100, 150, 195],
      yTicks: [0, 30, 60, 90, 125],
      xLabel: `Engine #${traj.unit_id} operating cycle`,
      yLabel: 'RUL (cycles)',
    });

    svg.appendChild(el('path', {
      d: buildLinePath(traj.cycles, traj.true_rul, xA, yA),
      fill: 'none',
      stroke: COLORS.ink,
      'stroke-width': 2.2,
    }));

    if (mode === 'proper') {
      svg.appendChild(el('path', {
        d: buildLinePath(traj.cycles, traj.pred_proper_unseen_engine, xA, yA),
        fill: 'none',
        stroke: COLORS.model,
        'stroke-width': 2.0,
      }));
      svg.appendChild(el('text', { x: xA(20), y: yA(15), fill: COLORS.model, 'font-size': 11.5, 'font-weight': 600 },
        'Proper unseen-engine prediction (Random Forest)'));
    } else {
      svg.appendChild(el('path', {
        d: buildLinePath(traj.cycles, traj.pred_leaky_row_split, xA, yA),
        fill: 'none',
        stroke: COLORS.fail,
        'stroke-width': 2.0,
      }));
      // Show leaked training points along true trajectory
      traj.cycles.forEach((c, i) => {
        if (i % 3 === 0 && traj.in_leaky_train_mask[i]) {
          svg.appendChild(el('circle', { cx: xA(c), cy: yA(traj.true_rul[i]), r: 2.3, fill: COLORS.fail, opacity: 0.75 }));
        }
      });
      svg.appendChild(el('text', { x: xA(14), y: yA(14), fill: COLORS.fail, 'font-size': 11.5, 'font-weight': 600 },
        '● Neighboring cycles leaked into train (Random Forest)'));
    }
    svg.appendChild(el('text', { x: padA.l + 4, y: padA.t - 8, fill: COLORS.ink, 'font-size': 12, 'font-weight': 600 },
      `A. Engine #${traj.unit_id} Random Forest trace (${mode === 'proper' ? 'held-out' : 'interleaved rows'})`));

    // Panel B: Predicted vs True scatter (or Ridge bar comparison)
    const padB = { l: 452, r: 22, t: 28, b: 44 };
    if (modelId === 'rung4_trends_ridge') {
      const xB = scaleLinear(0, 3, padB.l, W - padB.r);
      const yB = scaleLinear(0, 18, H - padB.b, padB.t);
      drawAxes(svg, {
        xMap: xB,
        yMap: yB,
        xMin: 0,
        xMax: 3,
        yMin: 0,
        yMax: 18,
        xTicks: [1, 2],
        xFmt: (v) => (v === 1 ? 'Proper split' : 'Leaky row split'),
        yTicks: [0, 5, 10, 15],
        yLabel: 'Test RMSE (cycles)',
      });
      svg.appendChild(el('rect', { x: xB(1) - 26, y: yB(row.proper_heldout_engines_test_rmse), width: 52, height: yB(0) - yB(row.proper_heldout_engines_test_rmse), fill: COLORS.model }));
      svg.appendChild(el('rect', { x: xB(2) - 26, y: yB(row.leaky_random_row_split_test_rmse), width: 52, height: yB(0) - yB(row.leaky_random_row_split_test_rmse), fill: mode === 'leaky' ? COLORS.fail : COLORS.fixed }));
      svg.appendChild(el('text', { x: xB(1), y: yB(row.proper_heldout_engines_test_rmse) - 6, 'text-anchor': 'middle', fill: COLORS.ink, 'font-size': 12, 'font-weight': 600 }, `${row.proper_heldout_engines_test_rmse}`));
      svg.appendChild(el('text', { x: xB(2), y: yB(row.leaky_random_row_split_test_rmse) - 6, 'text-anchor': 'middle', fill: COLORS.ink, 'font-size': 12, 'font-weight': 600 }, `${row.leaky_random_row_split_test_rmse}`));
      svg.appendChild(el('text', { x: padB.l + 4, y: padB.t - 8, fill: COLORS.ink, 'font-size': 12, 'font-weight': 600 },
        'B. Linear Ridge resists row memorization (only 1.1% drop)'));
    } else {
      const sc = leak.scatter_and_trajectory[modelId][mode];
      const xB = scaleLinear(0, 130, padB.l, W - padB.r);
      const yB = scaleLinear(0, 130, H - padB.b, padB.t);
      drawAxes(svg, {
        xMap: xB,
        yMap: yB,
        xMin: 0,
        xMax: 130,
        yMin: 0,
        yMax: 130,
        xTicks: [0, 30, 60, 90, 125],
        yTicks: [0, 30, 60, 90, 125],
        xLabel: 'True capped RUL (cycles)',
        yLabel: 'Predicted RUL',
      });
      svg.appendChild(el('line', { x1: xB(0), x2: xB(125), y1: yB(0), y2: yB(125), stroke: COLORS.ink, 'stroke-dasharray': '4,3', 'stroke-width': 1.2 }));
      const dotColor = mode === 'proper' ? COLORS.model : COLORS.fail;
      for (let i = 0; i < sc.y_true.length; i++) {
        svg.appendChild(el('circle', {
          cx: xB(sc.y_true[i]),
          cy: yB(sc.y_pred[i]),
          r: 2.0,
          fill: dotColor,
          opacity: 0.45,
        }));
      }
      const rmseVal = mode === 'proper' ? row.proper_heldout_engines_test_rmse : row.leaky_random_row_split_test_rmse;
      const shortLbl = row.model_label.replace('HistGradientBoostingRegressor', 'HistGBR');
      svg.appendChild(el('text', { x: padB.l + 4, y: padB.t - 8, fill: COLORS.ink, 'font-size': 12, 'font-weight': 600 },
        `B. ${shortLbl} (${mode}, RMSE = ${rmseVal})`));
    }

    host.innerHTML = '';
    host.appendChild(svg);
    showSvgHost('fig6-svg-host', host);
  }

  btnProper.addEventListener('click', () => { mode = 'proper'; render(); });
  btnLeaky.addEventListener('click', () => { mode = 'leaky'; render(); });
  modelSel.addEventListener('change', () => { modelId = modelSel.value; render(); });
  resetBtn.addEventListener('click', () => {
    mode = 'proper';
    modelId = 'rung5_trends_hgb';
    modelSel.value = modelId;
    render();
  });

  render();
}

/* ==========================================================================
   Figure 7 & Figure 8: Ladder Progression and PR / Calibration
   ========================================================================== */
function renderFigure7(data) {
  const host = document.getElementById('fig7-svg-host');
  if (!host) return;

  const reg = data.test_regression_ladder;
  const labels = ['R0 Mean', 'R2 Age', 'R3 Ridge', 'R4 Ridge', 'R5 RF', 'R5 HGB'];
  const ids = ['rung0_mean_rul', 'rung2_age_linear', 'rung3_instant_ridge', 'rung4_trends_ridge', 'rung5_trends_rf', 'rung5_trends_hgb'];
  const rows = ids.map((id) => reg.find((r) => r.model_id === id));

  const W = 780;
  const H = 250;
  const pad = { l: 58, r: 24, t: 28, b: 46 };
  const svg = createSvg(W, H, 'RUL RMSE progression from Rung 0 to Rung 5');

  const xMap = scaleLinear(0, rows.length, pad.l, W - pad.r);
  const yMap = scaleLinear(0, 45, H - pad.b, pad.t);

  drawAxes(svg, {
    xMap,
    yMap,
    xMin: 0,
    xMax: rows.length,
    yMin: 0,
    yMax: 45,
    xTicks: rows.map((_, i) => i + 0.5),
    xFmt: (v) => labels[Math.floor(v)] || '',
    yTicks: [0, 10, 20, 30, 40],
    yLabel: 'RUL RMSE (cycles)',
  });

  rows.forEach((r, i) => {
    const cx = xMap(i + 0.5);
    const bw = 24;
    svg.appendChild(el('rect', { x: cx - bw - 1, y: yMap(r.dev_cv_rmse), width: bw, height: yMap(0) - yMap(r.dev_cv_rmse), fill: COLORS.dev }));
    svg.appendChild(el('rect', { x: cx + 1, y: yMap(r.rmse_all), width: bw, height: yMap(0) - yMap(r.rmse_all), fill: COLORS.test }));
    svg.appendChild(el('text', { x: cx - bw / 2 - 1, y: yMap(r.dev_cv_rmse) - 5, 'text-anchor': 'middle', fill: COLORS.dev, 'font-size': 11, 'font-weight': 600 }, r.dev_cv_rmse.toFixed(1)));
    svg.appendChild(el('text', { x: cx + bw / 2 + 1, y: yMap(r.rmse_all) - 5, 'text-anchor': 'middle', fill: COLORS.testInk, 'font-size': 11, 'font-weight': 600 }, r.rmse_all.toFixed(1)));
  });

  svg.appendChild(el('text', { x: W - pad.r, y: pad.t, 'text-anchor': 'end', fill: COLORS.dev, 'font-size': 11.8, 'font-weight': 600 }, '■ 80 Dev CV   '));
  svg.appendChild(el('text', { x: W - pad.r, y: pad.t + 16, 'text-anchor': 'end', fill: COLORS.testInk, 'font-size': 11.8, 'font-weight': 600 }, '■ 20 Held-Out Test'));

  host.innerHTML = '';
  host.appendChild(svg);
  showSvgHost('fig7-svg-host', host);
}

function renderFigure8(data) {
  const host = document.getElementById('fig8-svg-host');
  const prc = data.pr_and_calibration;
  if (!host || !prc) return;

  const W = 780;
  const H = 265;
  const svg = createSvg(W, H, 'Precision-Recall and Reliability Calibration curves on development CV');

  const specs = [
    { id: 'rung4_trends_logreg', label: 'R4 Trends Logistic (PR-AUC 0.986)', color: COLORS.model, dash: 'none' },
    { id: 'rung5_trends_hgb_clf', label: 'R5 Trends HistGB (0.978)', color: COLORS.test, dash: '5,3' },
    { id: 'rung3_instant_logreg', label: 'R3 Instant Logistic (0.952)', color: COLORS.dev, dash: '2,2' },
    { id: 'rung2_age_logreg', label: 'R2 Age-Only (0.512)', color: COLORS.fixed, dash: '6,4' },
  ];

  // Panel A: PR curve
  const padA = { l: 58, r: 422, t: 28, b: 44 };
  const xA = scaleLinear(0, 1, padA.l, W - padA.r);
  const yA = scaleLinear(0, 1.04, H - padA.b, padA.t);
  drawAxes(svg, {
    xMap: xA,
    yMap: yA,
    xMin: 0,
    xMax: 1,
    yMin: 0,
    yMax: 1.04,
    xTicks: [0, 0.25, 0.5, 0.75, 1.0],
    yTicks: [0, 0.25, 0.5, 0.75, 1.0],
    xFmt: (v) => v.toFixed(2),
    yFmt: (v) => v.toFixed(2),
    xLabel: 'Recall',
    yLabel: 'Precision',
  });
  svg.appendChild(el('line', { x1: xA(0), x2: xA(1), y1: yA(prc.base_rate), y2: yA(prc.base_rate), stroke: COLORS.rule, 'stroke-dasharray': '3,3' }));
  specs.forEach((s, idx) => {
    const m = prc.models[s.id];
    svg.appendChild(el('path', { d: buildLinePath(m.pr_recall, m.pr_precision, xA, yA), fill: 'none', stroke: s.color, 'stroke-width': 2.0, 'stroke-dasharray': s.dash }));
    svg.appendChild(el('text', {
      x: xA(0.05),
      y: yA(0.78 - idx * 0.11),
      fill: s.color === COLORS.test ? COLORS.testInk : s.color,
      'font-size': 11.2,
      'font-weight': 600,
      'paint-order': 'stroke',
      stroke: '#FFFFFF',
      'stroke-width': 3,
      'stroke-linejoin': 'round',
    }, s.label));
  });
  svg.appendChild(el('text', { x: padA.l + 4, y: padA.t - 8, fill: COLORS.ink, 'font-size': 12.2, 'font-weight': 600 }, 'A. Precision-Recall (dev CV)'));

  // Panel B: Calibration curve
  const padB = { l: 452, r: 22, t: 28, b: 44 };
  const xB = scaleLinear(0, 1, padB.l, W - padB.r);
  const yB = scaleLinear(0, 1.04, H - padB.b, padB.t);
  drawAxes(svg, {
    xMap: xB,
    yMap: yB,
    xMin: 0,
    xMax: 1,
    yMin: 0,
    yMax: 1.04,
    xTicks: [0, 0.25, 0.5, 0.75, 1.0],
    yTicks: [0, 0.25, 0.5, 0.75, 1.0],
    xFmt: (v) => v.toFixed(2),
    yFmt: (v) => v.toFixed(2),
    xLabel: 'Mean predicted alarm probability',
    yLabel: 'Observed positive fraction',
  });
  svg.appendChild(el('line', { x1: xB(0), x2: xB(1), y1: yB(0), y2: yB(1), stroke: COLORS.ink, 'stroke-dasharray': '4,3', 'stroke-width': 1.2 }));
  specs.forEach((s) => {
    const m = prc.models[s.id];
    svg.appendChild(el('path', { d: buildLinePath(m.cal_mean_pred, m.cal_frac_pos, xB, yB), fill: 'none', stroke: s.color, 'stroke-width': 2.0, 'stroke-dasharray': s.dash }));
  });
  svg.appendChild(el('text', { x: padB.l + 4, y: padB.t - 8, fill: COLORS.ink, 'font-size': 12.2, 'font-weight': 600 }, 'B. Calibration reliability (diagonal = ideal)'));

  host.innerHTML = '';
  host.appendChild(svg);
  showSvgHost('fig8-svg-host', host);
}

/* ==========================================================================
   Error Case Browser Widget (Figure 9)
   ========================================================================== */
function initErrorBrowser(data, explorerHandle) {
  const errData = data.error_analysis;
  const host = document.getElementById('fig9-svg-host');
  if (!errData || !host) return;

  const caseBtns = Array.from(document.querySelectorAll('.case-btn'));
  const engSel = document.getElementById('err-engine-select');
  const resetBtn = document.getElementById('err-reset-btn');
  const takeawayEl = document.getElementById('err-live-takeaway');
  const descEl = document.getElementById('case-desc');
  const wrongEl = document.getElementById('case-wrong');
  const reasonEl = document.getElementById('case-reason');

  // Populate dropdown with all 20 test engines
  engSel.innerHTML = '';
  errData.per_engine_error_audit.forEach((row) => {
    const opt = document.createElement('option');
    opt.value = String(row.unit_id);
    opt.textContent = `Engine #${row.unit_id} (${row.max_cycle} cyc, HGB RMSE ${row.hgb_rmse})`;
    engSel.appendChild(opt);
  });

  const caseCopy = {
    shortest_lived: {
      unitId: 24,
      jumpModel: 'rung4_trends_logreg',
      desc: '<strong>Engine #24</strong> fails at <strong>cycle 147</strong> (shortest lifetime in held-out test set). Fixed schedule <strong>K*=179</strong> misses it and pays failure cost <strong>100</strong>.',
      wrong: 'HistGBR over-predicts early wear by up to <strong>+40.08 cycles</strong> before converging tightly inside the final <strong>30 cycles</strong> (last-30 mean error <strong>+1.30 cycles</strong>).',
      reason: 'Steep late HPC wear triggers Logistic at <strong>cycle 130 (cost 17)</strong> and HistGBR at <strong>cycle 140 (cost 7)</strong>.',
    },
    longest_lived: {
      unitId: 92,
      jumpModel: 'rung5_trends_hgb',
      desc: '<strong>Engine #92</strong> survives <strong>341 cycles</strong> (longest in held-out test set). Fixed <strong>K*=179</strong> pulls it <strong>162 cycles early (cost 162)</strong>.',
      wrong: 'HistGBR with locked threshold <strong>tau=9.0</strong> fails to dip below <strong>9.0</strong> prior to <strong>cycle 341</strong>, causing its lone test failure (<strong>cost 100</strong>).',
      reason: 'Tree leaves average older training units conservatively near extreme cycle age <strong>341</strong>, whereas Logistic Regression fires at <strong>cycle 323 (cost 18)</strong>.',
    },
    worst_asymmetric_error: {
      unitId: 61,
      jumpModel: 'rung5_trends_hgb',
      desc: '<strong>Engine #61</strong> fails at <strong>cycle 185</strong> and posts the highest mean NASA asymmetric score (<strong>5.043</strong>) outside the shortest/longest extremes.',
      wrong: 'HistGBR over-predicts mid-wear RUL by up to <strong>+44.08 cycles</strong> before rapid late correction.',
      reason: 'Engine #61 starts healthier than average before wearing rapidly; both sensor policies still trigger alarms before failure (Logistic <strong>@166</strong>, HistGBR <strong>@177</strong>).',
    },
  };

  let activeUnitId = 24;

  function renderCase() {
    const csMap = errData.error_case_studies.case_studies;
    const activeCaseKey = Object.keys(caseCopy).find((k) => caseCopy[k].unitId === activeUnitId) || null;
    caseBtns.forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.case === activeCaseKey));
    });
    engSel.value = String(activeUnitId);

    const audit = errData.per_engine_error_audit.find((r) => r.unit_id === activeUnitId);

    if (activeCaseKey) {
      const cs = csMap[activeCaseKey];
      const cp = caseCopy[activeCaseKey];
      takeawayEl.innerHTML =
        `Case study — <strong>Engine #${cs.unit_id}</strong> (lifetime <strong>${cs.max_cycle} cyc</strong>): Fixed schedule costs <strong>${cs.fixed_cost.toFixed(0)}</strong>, ` +
        `Logistic costs <strong>${cs.logreg_cost.toFixed(0)}</strong> (alarm <strong>@${cs.logreg_alarm_cycle}</strong>), and HistGBR costs <strong>${cs.hgb_cost.toFixed(0)}</strong> ` +
        `(${cs.hgb_alarm_cycle !== null ? `alarm <strong>@${cs.hgb_alarm_cycle}</strong>` : '<strong>missed alarm</strong>'}).`;
      descEl.innerHTML = cp.desc;
      wrongEl.innerHTML = cp.wrong;
      reasonEl.innerHTML = `${cp.reason} <a href="#explorer-widget" class="jump-explorer" data-model="${cp.jumpModel}">Inspect in explorer</a>.`;
      renderCaseTrajectorySvg(cs);
    } else {
      takeawayEl.innerHTML =
        `Held-out <strong>Engine #${audit.unit_id}</strong> (lifetime <strong>${audit.max_cycle} cyc</strong>): HistGBR RMSE = <strong>${audit.hgb_rmse}</strong>, NASA asym = <strong>${audit.hgb_nasa_asym_mean}</strong>. ` +
        `Policy costs at <strong>100:1</strong> — Fixed: <strong>${audit.fixed_cost}</strong>, Logistic: <strong>${audit.logreg_cost}</strong>, HistGBR: <strong>${audit.hgb_cost}</strong>.`;
      descEl.innerHTML = `<strong>Engine #${audit.unit_id}</strong> runs <strong>${audit.max_cycle} cycles</strong>. Fixed schedule <strong>K*=179</strong> outcome: ${audit.fixed_outcome} (<strong>cost ${audit.fixed_cost}</strong>).`;
      wrongEl.innerHTML = `HistGBR RMSE is <strong>${audit.hgb_rmse}</strong> (Ridge RMSE <strong>${audit.ridge_rmse}</strong>), max over-prediction <strong>+${audit.hgb_max_overpred} cyc</strong>, last-30 mean error <strong>${audit.hgb_last30_mean_err} cyc</strong>.`;
      reasonEl.innerHTML = `Logistic alarms at cycle <strong>${audit.logreg_alarm_cycle ?? 'none'}</strong> (<strong>cost ${audit.logreg_cost}</strong>); HistGBR alarms at cycle <strong>${audit.hgb_alarm_cycle ?? 'none'}</strong> (<strong>cost ${audit.hgb_cost}</strong>). <a href="#explorer-widget" class="jump-explorer" data-model="rung4_trends_logreg">Inspect in explorer</a>.`;
      renderAuditBarSvg(audit);
    }

    const jumpLink = reasonEl.querySelector('.jump-explorer');
    if (jumpLink && explorerHandle) {
      jumpLink.addEventListener('click', () => {
        explorerHandle.jumpToModel(jumpLink.dataset.model);
      });
    }
  }

  function renderCaseTrajectorySvg(cs) {
    const W = 780;
    const H = 265;
    const pad = { l: 58, r: 44, t: 32, b: 42 };
    const svg = createSvg(W, H, `Engine ${cs.unit_id} predicted RUL and alarm trajectory`);
    const maxX = Math.max(cs.max_cycle, cs.fixed_k_dev) + 10;
    const xMap = scaleLinear(1, maxX, pad.l, W - pad.r);
    const yRul = scaleLinear(0, 132, H - pad.b, pad.t);
    const yProb = scaleLinear(0, 1, H - pad.b, pad.t);

    drawAxes(svg, {
      xMap,
      yMap: yRul,
      xMin: 1,
      xMax: maxX,
      yMin: 0,
      yMax: 132,
      xTicks: niceTicks(1, cs.max_cycle, 5),
      yTicks: [0, 30, 60, 90, 125],
      xLabel: `Engine #${cs.unit_id} cycle t (failure @${cs.max_cycle})`,
      yLabel: 'RUL (cycles)',
    });

    // Right axis for alarm probability
    const rx = W - pad.r;
    svg.appendChild(el('line', { x1: rx, x2: rx, y1: H - pad.b, y2: pad.t, stroke: COLORS.rule, 'stroke-width': 1 }));
    [0, 0.5, 1.0].forEach((p) => {
      svg.appendChild(el('text', { x: rx + 6, y: yProb(p) + 4, fill: COLORS.model, 'font-size': 11.5 }, p.toFixed(1)));
    });

    svg.appendChild(el('path', { d: buildLinePath(cs.cycles, cs.true_rul, xMap, yRul), fill: 'none', stroke: COLORS.ink, 'stroke-width': 2.3 }));
    svg.appendChild(el('path', { d: buildLinePath(cs.cycles, cs.pred_ridge, xMap, yRul), fill: 'none', stroke: COLORS.dev, 'stroke-width': 1.5, 'stroke-dasharray': '4,3' }));
    svg.appendChild(el('path', { d: buildLinePath(cs.cycles, cs.pred_hgb, xMap, yRul), fill: 'none', stroke: COLORS.test, 'stroke-width': 2.1 }));
    svg.appendChild(el('path', { d: buildLinePath(cs.cycles, cs.prob_logreg, xMap, yProb), fill: 'none', stroke: COLORS.model, 'stroke-width': 1.7, 'stroke-dasharray': '2,2' }));

    const halo = { 'paint-order': 'stroke', stroke: '#FFFFFF', 'stroke-width': 3, 'stroke-linejoin': 'round' };

    // Markers
    svg.appendChild(el('line', { x1: xMap(cs.fixed_k_dev), x2: xMap(cs.fixed_k_dev), y1: pad.t, y2: H - pad.b, stroke: COLORS.fixed, 'stroke-dasharray': '5,4', 'stroke-width': 1.6 }));
    const fixedInside = cs.fixed_k_dev > cs.max_cycle * 0.75;
    svg.appendChild(el('text', {
      x: fixedInside ? xMap(cs.fixed_k_dev) - 4 : xMap(cs.fixed_k_dev) + 4,
      y: fixedInside ? pad.t + 56 : pad.t + 12,
      'text-anchor': fixedInside ? 'end' : 'start',
      fill: COLORS.fixedInk,
      'font-size': 11.2,
      'font-weight': 600,
      ...halo,
    }, `Fixed K*=${cs.fixed_k_dev}`));

    if (cs.logreg_alarm_cycle !== null) {
      svg.appendChild(el('line', { x1: xMap(cs.logreg_alarm_cycle), x2: xMap(cs.logreg_alarm_cycle), y1: pad.t, y2: H - pad.b, stroke: COLORS.model, 'stroke-width': 1.6 }));
      const lrRight = cs.logreg_alarm_cycle < maxX * 0.25;
      svg.appendChild(el('text', {
        x: lrRight ? xMap(cs.logreg_alarm_cycle) + 4 : xMap(cs.logreg_alarm_cycle) - 4,
        y: pad.t + 26,
        'text-anchor': lrRight ? 'start' : 'end',
        fill: COLORS.model,
        'font-size': 11.2,
        'font-weight': 600,
        ...halo,
      }, `LogReg @${cs.logreg_alarm_cycle}`));
    }
    if (cs.hgb_alarm_cycle !== null) {
      svg.appendChild(el('line', { x1: xMap(cs.hgb_alarm_cycle), x2: xMap(cs.hgb_alarm_cycle), y1: pad.t, y2: H - pad.b, stroke: COLORS.test, 'stroke-dasharray': '3,2', 'stroke-width': 1.6 }));
      const hgbRight = cs.hgb_alarm_cycle < maxX * 0.25;
      svg.appendChild(el('text', {
        x: hgbRight ? xMap(cs.hgb_alarm_cycle) + 4 : xMap(cs.hgb_alarm_cycle) - 4,
        y: pad.t + 41,
        'text-anchor': hgbRight ? 'start' : 'end',
        fill: COLORS.testInk,
        'font-size': 11.2,
        'font-weight': 600,
        ...halo,
      }, `HistGBR @${cs.hgb_alarm_cycle}`));
    } else {
      svg.appendChild(el('text', { x: xMap(cs.max_cycle) - 6, y: pad.t + 41, 'text-anchor': 'end', fill: COLORS.fail, 'font-size': 11.5, 'font-weight': 700, ...halo }, 'HistGBR missed (cost 100)'));
    }

    svg.appendChild(el('text', { x: pad.l, y: 15, fill: COLORS.ink, 'font-size': 11.2, 'font-weight': 600 },
      'True RUL (black) · HistGBR pred (amber) · Ridge pred (slate dashed) · P(alarm) (teal dotted)'));

    host.innerHTML = '';
    host.appendChild(svg);
  }

  function renderAuditBarSvg(audit) {
    const W = 780;
    const H = 220;
    const pad = { l: 58, r: 24, t: 28, b: 44 };
    const svg = createSvg(W, H, `Policy cost comparison for Engine ${audit.unit_id}`);
    const maxY = Math.max(110, audit.fixed_cost, audit.logreg_cost, audit.hgb_cost) * 1.15;
    const xMap = scaleLinear(0, 3, pad.l, W - pad.r);
    const yMap = scaleLinear(0, maxY, H - pad.b, pad.t);
    const bars = [
      { label: `Fixed (${audit.fixed_outcome})`, val: audit.fixed_cost, color: COLORS.fixed },
      { label: `Logistic (@${audit.logreg_alarm_cycle ?? 'fail'})`, val: audit.logreg_cost, color: COLORS.model },
      { label: `HistGBR (@${audit.hgb_alarm_cycle ?? 'fail'})`, val: audit.hgb_cost, color: audit.hgb_alarm_cycle ? COLORS.test : COLORS.fail },
    ];
    drawAxes(svg, {
      xMap,
      yMap,
      xMin: 0,
      xMax: 3,
      yMin: 0,
      yMax: maxY,
      xTicks: [0.5, 1.5, 2.5],
      xFmt: (v) => bars[Math.floor(v)].label,
      yTicks: niceTicks(0, Math.round(maxY / 20) * 20, 4),
      yLabel: 'Engine cost (c=100)',
    });
    bars.forEach((b, i) => {
      const cx = xMap(i + 0.5);
      svg.appendChild(el('rect', { x: cx - 32, y: yMap(b.val), width: 64, height: yMap(0) - yMap(b.val), fill: b.color }));
      svg.appendChild(el('text', { x: cx, y: yMap(b.val) - 6, 'text-anchor': 'middle', fill: COLORS.ink, 'font-size': 12, 'font-weight': 600 }, b.val.toFixed(1)));
    });
    host.innerHTML = '';
    host.appendChild(svg);
  }

  caseBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      activeUnitId = caseCopy[btn.dataset.case].unitId;
      renderCase();
    });
  });

  engSel.addEventListener('change', () => {
    activeUnitId = Number(engSel.value);
    renderCase();
  });

  resetBtn.addEventListener('click', () => {
    activeUnitId = 24;
    renderCase();
  });

  renderCase();
  showSvgHost('fig9-svg-host', host);
}

/* ==========================================================================
   Populate Collapsed Details Tables from JSON
   ========================================================================== */
function populateDetailsTables(data) {
  const sensorHost = document.getElementById('sensor-screening-table-host');
  if (sensorHost && data.sensor_screening) {
    const rows = data.sensor_screening
      .map(
        (r) => `<tr>
          <td><code>${r.channel}</code></td>
          <td>${r.physical_description}</td>
          <td class="num">${Number(r.mean).toFixed(2)}</td>
          <td class="num">${Number(r.std).toFixed(4)}</td>
          <td class="num">${Number(r.corr_rul_capped).toFixed(3)}</td>
          <td>${r.status}</td>
        </tr>`
      )
      .join('');
    sensorHost.innerHTML = `<table class="data-table">
      <thead><tr><th>Channel</th><th>Description</th><th class="num">Mean</th><th class="num">Std</th><th class="num">Corr (RUL)</th><th>Screening decision</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  const attrHost = document.getElementById('feature-attribution-table-host');
  if (attrHost && data.feature_attribution) {
    const rows = data.feature_attribution
      .slice(0, 15)
      .map(
        (r) => `<tr>
          <td><code>${r.feature}</code></td>
          <td class="num">${Number(r.ridge_coef).toFixed(3)}</td>
          <td class="num">${Number(r.lasso_coef).toFixed(3)}</td>
          <td class="num">${Number(r.logreg_alarm_coef).toFixed(3)}</td>
          <td class="num">${Number(r.hgb_perm_importance_rmse).toFixed(3)}</td>
        </tr>`
      )
      .join('');
    attrHost.innerHTML = `<table class="data-table">
      <thead><tr><th>Feature (top 15 by HGB permutation)</th><th class="num">Ridge coef</th><th class="num">Lasso coef</th><th class="num">LogReg coef</th><th class="num">HGB perm RMSE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  const sensHost = document.getElementById('sensitivity-table-host');
  if (sensHost && data.sensitivity_summary) {
    const rows = data.sensitivity_summary
      .map(
        (r) => `<tr>
          <td>${r.sweep_dimension}</td>
          <td>${r.setting_label}</td>
          <td>${r.model_name}</td>
          <td class="num">${r.rmse_all !== null ? Number(r.rmse_all).toFixed(3) : '&mdash;'}</td>
          <td class="num">${r.pr_auc !== null ? Number(r.pr_auc).toFixed(4) : '&mdash;'}</td>
          <td class="num">${Number(r.oof_policy_cost_c100).toFixed(2)}</td>
        </tr>`
      )
      .join('');
    sensHost.innerHTML = `<table class="data-table">
      <thead><tr><th>Dimension</th><th>Setting</th><th>Model</th><th class="num">Dev RMSE</th><th class="num">Dev PR-AUC</th><th class="num">Dev Policy Cost (100:1)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  const auditHost = document.getElementById('per-engine-audit-table-host');
  if (auditHost && data.error_analysis?.per_engine_error_audit) {
    const rows = data.error_analysis.per_engine_error_audit
      .map(
        (r) => `<tr>
          <td class="num">#${r.unit_id}</td>
          <td class="num">${r.max_cycle}</td>
          <td class="num">${Number(r.hgb_rmse).toFixed(2)}</td>
          <td class="num">${Number(r.hgb_nasa_asym_mean).toFixed(2)}</td>
          <td>${r.fixed_outcome}</td>
          <td class="num">${Number(r.fixed_cost).toFixed(0)}</td>
          <td class="num">${r.logreg_alarm_cycle ?? 'Fail'} (${Number(r.logreg_cost).toFixed(0)})</td>
          <td class="num">${r.hgb_alarm_cycle ?? 'Fail'} (${Number(r.hgb_cost).toFixed(0)})</td>
        </tr>`
      )
      .join('');
    auditHost.innerHTML = `<table class="data-table">
      <thead><tr><th class="num">Unit</th><th class="num">Lifetime</th><th class="num">HGB RMSE</th><th class="num">HGB Asym</th><th>Fixed K*=179</th><th class="num">Fixed Cost</th><th class="num">LogReg (Cost)</th><th class="num">HistGBR (Cost)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }
}

/* ==========================================================================
   Navigation, Margin-Note Collapse Behavior & Copy Buttons
   ========================================================================== */
function initUIControls() {
  // Margin notes: open by default on wide desktop (>=1080px), collapsed expandable inline on mobile
  const mql = window.matchMedia('(min-width: 1080px)');
  const notes = Array.from(document.querySelectorAll('.margin-note-box'));
  function syncNotes() {
    notes.forEach((d) => {
      d.open = mql.matches;
    });
  }
  syncNotes();
  mql.addEventListener('change', syncNotes);

  // Close mobile menu after clicking a section anchor
  const mobileNav = document.querySelector('.mobile-nav');
  if (mobileNav) {
    mobileNav.querySelectorAll('a').forEach((a) => {
      a.addEventListener('click', () => {
        mobileNav.open = false;
      });
    });
  }

  // Highlight active section in sticky left rail on scroll
  const railLinks = Array.from(document.querySelectorAll('.rail-list a'));
  const sections = railLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  if ('IntersectionObserver' in window && sections.length > 0) {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length === 0) return;
        visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        const activeId = `#${visible[0].target.id}`;
        railLinks.forEach((a) => {
          if (a.getAttribute('href') === activeId) {
            a.setAttribute('aria-current', 'location');
          } else {
            a.removeAttribute('aria-current');
          }
        });
      },
      { rootMargin: '-12% 0px -68% 0px', threshold: 0.01 }
    );
    sections.forEach((sec) => obs.observe(sec));
  }

  // Sequential scroll reveal of sections
  const revealSections = Array.from(document.querySelectorAll('.reveal-section'));
  const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!prefersReducedMotion && 'IntersectionObserver' in window && revealSections.length > 0) {
    document.documentElement.classList.add('js-reveal-active');
    const revealObs = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-revealed');
            obs.unobserve(entry.target);
          }
        });
      },
      { rootMargin: '0px 0px -6% 0px', threshold: 0.04 }
    );
    revealSections.forEach((sec) => revealObs.observe(sec));

    // Reveal target section on anchor click or hash change immediately
    const revealTargetHash = () => {
      const hash = window.location.hash.slice(1);
      if (!hash) return;
      const target = document.getElementById(hash);
      if (target) {
        target.classList.add('is-revealed');
        const parentSec = target.closest('.reveal-section');
        if (parentSec) parentSec.classList.add('is-revealed');
      }
    };
    window.addEventListener('hashchange', revealTargetHash);
    if (window.location.hash) {
      setTimeout(revealTargetHash, 60);
    }
  } else {
    revealSections.forEach((sec) => sec.classList.add('is-revealed'));
  }

  // Copy buttons on command blocks
  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const target = document.getElementById(btn.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = 'Copied';
      } catch {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.textContent = 'Copied';
      }
      setTimeout(() => {
        btn.textContent = 'Copy';
      }, 1500);
    });
  });

  // Record default open state on grouped drawers so collapse resets cleanly to initial defaults
  document.querySelectorAll('details[data-drawer-group]').forEach((d) => {
    d.dataset.defaultOpen = d.open ? '1' : '0';
  });

  // Expand / reset toggle buttons on collapsible tables (Table 3, 5, 6, 8)
  function syncTable3Switcher(mode) {
    const sw = document.querySelector('[data-rung-switcher="table3-wrap"]');
    if (!sw) return;
    sw.querySelectorAll('button[data-rung-filter]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.rungFilter === mode));
    });
  }

  document.querySelectorAll('.table-toggle-btn[data-table-target]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const wrap = document.getElementById(btn.dataset.tableTarget);
      if (!wrap) return;
      wrap.classList.remove('filter-baselines', 'filter-sensors');
      const expanded = wrap.classList.toggle('is-expanded');
      btn.setAttribute('aria-expanded', String(expanded));
      btn.textContent = expanded ? btn.dataset.labelCollapse : btn.dataset.labelExpand;
      const statusEl = document.getElementById(btn.dataset.statusTarget);
      if (statusEl) {
        statusEl.textContent = expanded ? btn.dataset.statusExpanded : btn.dataset.statusCollapsed;
      }
      if (wrap.id === 'table3-wrap') {
        syncTable3Switcher(expanded ? 'all' : 'key');
      }
    });
  });

  // Model ladder filter bar for Table 3
  document.querySelectorAll('[data-rung-switcher]').forEach((sw) => {
    const wrap = document.getElementById(sw.dataset.rungSwitcher);
    if (!wrap) return;
    const footerBtn = wrap.querySelector('.table-toggle-btn[data-table-target="table3-wrap"]');
    const statusEl = document.getElementById('table3-toggle-status');
    sw.querySelectorAll('button[data-rung-filter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.rungFilter;
        syncTable3Switcher(mode);
        wrap.classList.remove('filter-baselines', 'filter-sensors', 'is-expanded');
        if (mode === 'all') {
          wrap.classList.add('is-expanded');
        } else if (mode === 'baselines') {
          wrap.classList.add('is-expanded', 'filter-baselines');
        } else if (mode === 'sensors') {
          wrap.classList.add('is-expanded', 'filter-sensors');
        }
        const isAll = mode !== 'key';
        if (footerBtn) {
          footerBtn.setAttribute('aria-expanded', String(isAll));
          footerBtn.textContent = isAll ? footerBtn.dataset.labelCollapse : footerBtn.dataset.labelExpand;
        }
        if (statusEl) {
          const map = {
            key: 'Showing 4 key rungs of 6 (summary view)',
            all: 'Showing all 6 model rungs',
            baselines: 'Showing 3 baseline rungs (Rungs 0–2)',
            sensors: 'Showing 3 sensor model rungs (Rungs 3–5)',
          };
          statusEl.textContent = map[mode] || '';
        }
      });
    });
  });

  // Segmented figure switchers (Section 2 Data & Section 6 Results)
  document.querySelectorAll('[data-fig-switcher]').forEach((sw) => {
    const group = sw.dataset.figSwitcher;
    const figs = Array.from(document.querySelectorAll(`figure[data-fig-group="${group}"]`));
    const btns = Array.from(sw.querySelectorAll('button[data-fig-mode]'));
    btns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.figMode;
        btns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
        figs.forEach((fig) => {
          fig.hidden = mode !== 'all' && fig.id !== mode;
        });
      });
    });
  });

  // Segmented 3-part switcher in Conclusion
  document.querySelectorAll('[data-conc-switcher]').forEach((sw) => {
    const parts = Array.from(document.querySelectorAll('.conclusion-part[data-conc-part]'));
    const btns = Array.from(sw.querySelectorAll('button[data-conc-mode]'));
    btns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.concMode;
        btns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
        parts.forEach((p) => {
          p.hidden = mode !== 'all' && p.dataset.concPart !== mode;
        });
      });
    });
  });

  // Group expand / reset buttons for progressive-disclosure drawers
  document.querySelectorAll('[data-group-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.groupToggle;
      const drawers = Array.from(document.querySelectorAll(`details[data-drawer-group="${group}"]`));
      const nextExpanded = btn.getAttribute('aria-expanded') !== 'true';
      drawers.forEach((d) => {
        d.open = nextExpanded ? true : d.dataset.defaultOpen === '1';
      });
      btn.setAttribute('aria-expanded', String(nextExpanded));
      btn.textContent = nextExpanded ? btn.dataset.labelCollapse : btn.dataset.labelExpand;
    });
  });

  // Toggle buttons on Reproduction command blocks
  function setCmdBlockOpen(block, open) {
    const pre = block.querySelector('pre');
    const toggleBtn = block.querySelector('.cmd-toggle-btn');
    if (pre) pre.hidden = !open;
    block.classList.toggle('is-collapsed', !open);
    if (toggleBtn) {
      toggleBtn.setAttribute('aria-expanded', String(open));
      toggleBtn.textContent = open ? 'Hide commands' : 'Show commands';
    }
  }

  document.querySelectorAll('.cmd-toggle-btn[data-cmd-target]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const pre = document.getElementById(btn.dataset.cmdTarget);
      const block = btn.closest('.cmd-block');
      if (!pre || !block) return;
      setCmdBlockOpen(block, pre.hidden);
    });
  });

  document.querySelectorAll('[data-cmd-group-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.cmdGroupToggle;
      const blocks = Array.from(document.querySelectorAll(`.cmd-block[data-cmd-group="${group}"]`));
      const nextExpanded = btn.getAttribute('aria-expanded') !== 'true';
      blocks.forEach((block, idx) => {
        setCmdBlockOpen(block, nextExpanded ? true : idx === 0);
      });
      btn.setAttribute('aria-expanded', String(nextExpanded));
      btn.textContent = nextExpanded ? btn.dataset.labelCollapse : btn.dataset.labelExpand;
    });
  });
}

/* ==========================================================================
   Bootstrap
   ========================================================================== */
async function init() {
  initUIControls();

  try {
    const resp = await fetch('data/report_data.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    renderFigure1(data);
    renderFigure2(data);
    renderFigure3(data);
    renderFigure4();
    const explorerHandle = initMainExplorer(data);
    initLeakageWidget(data);
    renderFigure7(data);
    renderFigure8(data);
    initErrorBrowser(data, explorerHandle);
    populateDetailsTables(data);
  } catch (err) {
    // Static fallback images and static HTML tables remain visible if JSON fails to load
    console.error('Failed to load report_data.json; retaining static fallbacks:', err);
  }
}

init();
