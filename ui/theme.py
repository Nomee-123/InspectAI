"""VISION CORE visual system: palette, fonts and the global CSS layer."""

from __future__ import annotations

PALETTE = {
    "navy": "#050B18",
    "panel": "#0A1426",
    "cyan": "#00E5FF",
    "blue": "#2979FF",
    "amber": "#FFB300",
    "red": "#FF5252",
    "text": "#E6F1FF",
    "muted": "#8FA6C2",
}

FONT_STACK = {
    "heading": "'Orbitron', 'Rajdhani', 'Segoe UI', sans-serif",
    "data": "'Rajdhani', 'Share Tech Mono', 'Consolas', monospace",
    "mono": "'Share Tech Mono', 'Consolas', 'Courier New', monospace",
}

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

@property --vc-angle {
  syntax: '<angle>';
  initial-value: 0deg;
  inherits: false;
}

:root {
  --vc-navy: #050B18;
  --vc-panel: #0A1426;
  --vc-panel-2: #07101f;
  --vc-cyan: #00E5FF;
  --vc-blue: #2979FF;
  --vc-amber: #FFB300;
  --vc-red: #FF5252;
  --vc-text: #E6F1FF;
  --vc-muted: #8FA6C2;
  --vc-line: rgba(0, 229, 255, 0.18);
  --vc-line-strong: rgba(0, 229, 255, 0.42);
  --vc-heading: 'Orbitron', 'Rajdhani', 'Segoe UI', sans-serif;
  --vc-data: 'Rajdhani', 'Share Tech Mono', 'Consolas', monospace;
  --vc-mono: 'Share Tech Mono', 'Consolas', 'Courier New', monospace;
  --vc-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
}

/* ------------------------------------------------------------------ */
/* Global shell                                                        */
/* ------------------------------------------------------------------ */

html, body, .stApp {
  background: var(--vc-navy);
  color: var(--vc-text);
}

.stApp {
  position: relative;
  isolation: isolate;
  min-height: 100vh;
  min-height: 100dvh;
  overflow-x: hidden;
}

/* Perspective grid + pulsing radial glow. */
.stApp::before {
  content: "";
  position: fixed;
  inset: -22vh -18vw -12vh -18vw;
  z-index: 0;
  pointer-events: none;
  background-image:
    radial-gradient(circle at 22% 18%, rgba(41, 121, 255, 0.16), transparent 34%),
    radial-gradient(circle at 78% 26%, rgba(0, 229, 255, 0.12), transparent 32%),
    radial-gradient(circle at 50% 100%, rgba(255, 179, 0, 0.05), transparent 45%),
    linear-gradient(rgba(0, 229, 255, 0.11) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 229, 255, 0.11) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 100% 100%, 64px 64px, 64px 64px;
  transform: perspective(780px) rotateX(61deg) translate3d(0, -10%, 0) scale(2.1);
  transform-origin: 50% 100%;
  animation: vc-grid-drift 28s linear infinite, vc-glow-pulse 9s ease-in-out infinite;
  opacity: 0.62;
  mask-image: linear-gradient(to bottom, transparent 0%, black 24%, black 78%, transparent 100%);
}

/* Scanlines and a slow vertical scan sweep. */
.stApp::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    repeating-linear-gradient(
      to bottom,
      rgba(230, 241, 255, 0.035) 0,
      rgba(230, 241, 255, 0.035) 1px,
      transparent 1px,
      transparent 4px
    ),
    linear-gradient(
      to bottom,
      transparent 0%,
      rgba(0, 229, 255, 0.035) 46%,
      rgba(0, 229, 255, 0.10) 50%,
      rgba(0, 229, 255, 0.035) 54%,
      transparent 100%
    );
  background-size: 100% 100%, 100% 220%;
  animation: vc-scan-drift 12s linear infinite;
  opacity: 0.9;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main,
.block-container {
  background: transparent !important;
}

.block-container {
  position: relative;
  z-index: 2;
  padding-top: 1.25rem;
  padding-bottom: 3rem;
  max-width: 1500px;
}

[data-testid="stHeader"] {
  background: transparent;
}

#MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4, h5, h6 {
  font-family: var(--vc-heading) !important;
  letter-spacing: 0.055em;
  color: var(--vc-text) !important;
}

p, li, label, span, div {
  font-family: var(--vc-data);
}

code, pre, .stCode, [data-testid="stJson"] {
  font-family: var(--vc-mono) !important;
}

a { color: var(--vc-cyan); }

/* Visible focus for every interactive control. */
a:focus-visible,
button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
summary:focus-visible,
[role="slider"]:focus-visible,
[role="button"]:focus-visible,
[role="tab"]:focus-visible {
  outline: 2px solid var(--vc-cyan) !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 4px rgba(0, 229, 255, 0.16) !important;
}

/* Streamlit widget skins.  Streamlit wraps buttons with a tooltip element when
   a help string is supplied, so the skin must target any descendant button. */
.stButton button,
.stDownloadButton button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
  font-family: var(--vc-heading);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--vc-cyan);
  background: linear-gradient(180deg, rgba(0, 229, 255, 0.10), rgba(0, 229, 255, 0.03));
  border: 1px solid rgba(0, 229, 255, 0.42);
  border-radius: 2px;
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

.stButton button:hover,
.stDownloadButton button:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--vc-cyan);
  box-shadow: 0 0 24px rgba(0, 229, 255, 0.22), inset 0 0 18px rgba(0, 229, 255, 0.10);
  transform: translateY(-1px);
  color: #ffffff;
}

.stButton button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(100deg, rgba(0, 229, 255, 0.26), rgba(41, 121, 255, 0.18));
  color: #f2fdff;
  border-color: var(--vc-cyan);
}

.stButton button:disabled,
.stDownloadButton button:disabled {
  opacity: 0.38;
  box-shadow: none;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(8, 18, 34, 0.78);
  border: 1px dashed rgba(0, 229, 255, 0.35);
  border-radius: 2px;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--vc-cyan);
  box-shadow: inset 0 0 24px rgba(0, 229, 255, 0.08);
}

.stSelectbox [data-baseweb="select"] > div,
.stCheckbox label,
.stTextInput input,
.stNumberInput input {
  background: rgba(7, 16, 31, 0.92) !important;
  border-color: rgba(0, 229, 255, 0.28) !important;
  color: var(--vc-text) !important;
}

[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(10, 20, 38, 0.98), rgba(5, 11, 24, 0.99));
  border-right: 1px solid rgba(0, 229, 255, 0.18);
  box-shadow: 12px 0 40px rgba(0, 0, 0, 0.32);
}

[data-testid="stSidebar"] .block-container {
  padding-top: 1rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 0.82rem;
  color: var(--vc-cyan) !important;
}

[data-testid="stExpander"] {
  background: rgba(7, 16, 31, 0.72);
  border: 1px solid rgba(0, 229, 255, 0.20);
  border-radius: 2px;
}

[data-testid="stAlert"] {
  border-radius: 2px;
  border: 1px solid rgba(255, 179, 0, 0.35);
  background: rgba(36, 27, 8, 0.72);
  color: #ffe7b0;
}

/* Tabs: short slide/opacity wipe, never faster than 3 Hz.
   Streamlit 1.6x renders React Aria tabs (role="tab"/"tablist"/"tabpanel")
   instead of BaseWeb nodes, so target the accessible roles directly. */
.stTabs [role="tablist"],
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px;
  border-bottom: 1px solid rgba(0, 229, 255, 0.18);
}

.stTabs [role="tab"],
[data-testid="stTabs"] [role="tab"] {
  font-family: var(--vc-heading);
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  color: var(--vc-muted);
  background: rgba(7, 16, 31, 0.55);
  border: 1px solid transparent;
  border-bottom: 2px solid transparent;
  border-radius: 2px 2px 0 0;
  padding: 0.62rem 1.05rem;
  transition: color 160ms ease, background 160ms ease, border-color 160ms ease;
}

.stTabs [role="tab"]:hover,
[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--vc-text);
  background: rgba(0, 229, 255, 0.07);
}

.stTabs [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--vc-cyan) !important;
  background: rgba(0, 229, 255, 0.09) !important;
  border-color: rgba(0, 229, 255, 0.32) !important;
  border-bottom-color: var(--vc-cyan) !important;
  box-shadow: inset 0 -1px 0 rgba(0, 229, 255, 0.75);
}

.stTabs [role="tabpanel"],
.stTabs [data-testid="stTabPanel"] {
  animation: vc-tab-in 280ms ease-out both;
}

/* ------------------------------------------------------------------ */
/* Typography + section chrome                                         */
/* ------------------------------------------------------------------ */

.vc-kicker {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-family: var(--vc-mono);
  font-size: 0.68rem;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--vc-cyan);
  margin: 1.3rem 0 0.65rem;
}

.vc-kicker::before {
  content: "";
  width: 26px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--vc-cyan));
  box-shadow: 0 0 10px var(--vc-cyan);
}

.vc-micro {
  font-family: var(--vc-mono);
  font-size: 0.66rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--vc-muted);
}

.vc-mono { font-family: var(--vc-mono); }
.vc-cyan { color: var(--vc-cyan); }
.vc-amber { color: var(--vc-amber); }
.vc-red { color: var(--vc-red); }
.vc-muted { color: var(--vc-muted); }

/* ------------------------------------------------------------------ */
/* Chamfered HUD panels with a rotating border highlight               */
/* ------------------------------------------------------------------ */

.hud-panel {
  position: relative;
  background:
    linear-gradient(160deg, rgba(12, 25, 45, 0.94), rgba(6, 13, 27, 0.97));
  border: 1px solid rgba(0, 229, 255, 0.16);
  padding: 1rem 1.05rem;
  margin: 0.55rem 0 0.9rem;
  box-shadow: var(--vc-shadow);
  clip-path: polygon(0 12px, 12px 0, calc(100% - 12px) 0, 100% 12px, 100% calc(100% - 12px), calc(100% - 12px) 100%, 12px 100%, 0 calc(100% - 12px));
  transition: transform 220ms ease, box-shadow 220ms ease;
  overflow: hidden;
}

.hud-panel::before {
  content: "";
  position: absolute;
  inset: -1px;
  padding: 1px;
  background: conic-gradient(from var(--vc-angle), transparent 0deg 250deg, rgba(0, 229, 255, 0.95) 300deg, transparent 344deg);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  mask-composite: exclude;
  animation: vc-border-spin 9s linear infinite;
  pointer-events: none;
  opacity: 0.9;
}

.hud-panel::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) left top / 18px 1px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) left top / 1px 18px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) right top / 18px 1px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) right top / 1px 18px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) left bottom / 18px 1px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) left bottom / 1px 18px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) right bottom / 18px 1px no-repeat,
    linear-gradient(var(--vc-cyan), var(--vc-cyan)) right bottom / 1px 18px no-repeat;
  opacity: 0.55;
}

.hud-panel:hover {
  transform: translateY(-2px);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 229, 255, 0.07);
}

.hud-panel.quiet::before { animation: none; opacity: 0.35; }
.hud-panel.quiet:hover { transform: none; }

/* ------------------------------------------------------------------ */
/* Header: rotating reticle + brand telemetry                           */
/* ------------------------------------------------------------------ */

.vc-header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 1.15rem;
  padding: 1.05rem 1.25rem;
  margin-bottom: 0.5rem;
  background:
    linear-gradient(100deg, rgba(8, 18, 34, 0.95), rgba(5, 11, 24, 0.82) 58%, rgba(0, 229, 255, 0.05));
  border: 1px solid rgba(0, 229, 255, 0.18);
  clip-path: polygon(0 16px, 16px 0, calc(100% - 40px) 0, 100% 40px, 100% 100%, 0 100%);
  overflow: hidden;
}

.vc-header::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.07), transparent);
  transform: translateX(-100%);
  animation: vc-header-sweep 7s ease-in-out infinite;
  pointer-events: none;
}

.vc-reticle {
  flex: 0 0 96px;
  width: 96px;
  height: 96px;
  filter: drop-shadow(0 0 12px rgba(0, 229, 255, 0.35));
}

.vc-reticle .vc-orbit { transform-origin: 60px 60px; animation: vc-spin 16s linear infinite; }
.vc-reticle .vc-orbit-fast { transform-origin: 60px 60px; animation: vc-spin 7s linear infinite reverse; }
.vc-reticle .vc-arc { transform-origin: 60px 60px; animation: vc-spin 3.6s linear infinite; }
.vc-reticle .vc-tick { stroke: rgba(0, 229, 255, 0.65); }

.vc-brand {
  min-width: 0;
  flex: 1 1 auto;
}

.vc-brand-title {
  font-family: var(--vc-heading);
  font-size: clamp(1.45rem, 3.1vw, 2.35rem);
  font-weight: 900;
  letter-spacing: 0.08em;
  line-height: 1;
  color: #f4fbff;
  text-shadow: 0 0 28px rgba(0, 229, 255, 0.35);
}

.vc-brand-title .slash { color: var(--vc-cyan); }

.vc-brand-sub {
  margin-top: 0.38rem;
  font-family: var(--vc-mono);
  font-size: 0.68rem;
  letter-spacing: 0.23em;
  text-transform: uppercase;
  color: var(--vc-cyan);
}

.vc-statusline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.1rem;
  margin-top: 0.52rem;
  font-family: var(--vc-mono);
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  color: var(--vc-muted);
}

.vc-statusline .dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-right: 6px;
  background: var(--vc-cyan);
  box-shadow: 0 0 9px var(--vc-cyan);
  animation: vc-dot-pulse 2.4s ease-in-out infinite;
}

.vc-statusline .dot.amber { background: var(--vc-amber); box-shadow: 0 0 9px var(--vc-amber); }
.vc-statusline .dot.red { background: var(--vc-red); box-shadow: 0 0 9px var(--vc-red); }

.vc-badge {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.4rem 0.7rem;
  border: 1px solid rgba(0, 229, 255, 0.32);
  background: rgba(0, 229, 255, 0.05);
  font-family: var(--vc-mono);
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--vc-text);
  white-space: nowrap;
}

.vc-badge img {
  height: 22px;
  width: auto;
  display: block;
}

.vc-badge .vc-logo-text { color: var(--vc-cyan); font-weight: 700; }

/* ------------------------------------------------------------------ */
/* Pipeline strip                                                       */
/* ------------------------------------------------------------------ */

.vc-pipeline {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 0;
  margin: 0.65rem 0 0.9rem;
  padding: 0.7rem 0.75rem;
  background: rgba(6, 14, 28, 0.86);
  border: 1px solid rgba(0, 229, 255, 0.16);
  position: relative;
  overflow: hidden;
}

.vc-node {
  position: relative;
  min-width: 0;
  padding: 0.25rem 0.25rem 0.15rem;
  text-align: center;
}

.vc-node:not(:last-child)::after {
  content: "";
  position: absolute;
  top: 17px;
  left: calc(50% + 17px);
  width: calc(100% - 34px);
  height: 1px;
  background: rgba(143, 166, 194, 0.22);
  transition: background 220ms ease;
}

.vc-node.done:not(:last-child)::after,
.vc-node.running:not(:last-child)::after {
  background: linear-gradient(90deg, var(--vc-cyan), rgba(0, 229, 255, 0.22));
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.45);
}

.vc-node-ring {
  width: 34px;
  height: 34px;
  margin: 0 auto 0.32rem;
  border-radius: 50%;
  border: 1px solid rgba(143, 166, 194, 0.32);
  background: rgba(7, 16, 31, 0.9);
  color: var(--vc-muted);
  font-family: var(--vc-mono);
  font-size: 0.62rem;
  line-height: 32px;
  transition: color 200ms ease, border-color 200ms ease, box-shadow 200ms ease;
}

.vc-node.running .vc-node-ring {
  border-color: transparent;
  background:
    radial-gradient(circle at center, rgba(7, 16, 31, 1) 56%, transparent 57%),
    conic-gradient(from 0deg, transparent 0 30%, var(--vc-cyan) 55%, transparent 80%);
  color: var(--vc-cyan);
  box-shadow: 0 0 18px rgba(0, 229, 255, 0.32);
  animation: vc-node-spin 1.8s linear infinite;
}

.vc-node.done .vc-node-ring {
  border-color: var(--vc-cyan);
  color: var(--vc-cyan);
  box-shadow: 0 0 14px rgba(0, 229, 255, 0.22);
}

.vc-node.ambiguous .vc-node-ring {
  border-color: var(--vc-amber);
  color: var(--vc-amber);
  box-shadow: 0 0 14px rgba(255, 179, 0, 0.22);
}

.vc-node.rejected .vc-node-ring {
  border-color: var(--vc-red);
  color: var(--vc-red);
  box-shadow: 0 0 14px rgba(255, 82, 82, 0.18);
}

.vc-node-name {
  font-family: var(--vc-mono);
  font-size: 0.58rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--vc-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.vc-node.done .vc-node-name,
.vc-node.running .vc-node-name { color: var(--vc-text); }

.vc-node-fn {
  display: block;
  margin-top: 0.18rem;
  font-family: var(--vc-mono);
  font-size: 0.5rem;
  letter-spacing: 0.02em;
  color: rgba(143, 166, 194, 0.72);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ------------------------------------------------------------------ */
/* Radar sweep while the real pipeline runs                             */
/* ------------------------------------------------------------------ */

.vc-radar-panel {
  display: grid;
  grid-template-columns: 150px 1fr;
  align-items: center;
  gap: 1.2rem;
  padding: 1rem 1.2rem;
  border: 1px solid rgba(0, 229, 255, 0.22);
  background: rgba(6, 14, 28, 0.9);
  margin: 0.7rem 0;
}

.vc-radar {
  position: relative;
  width: 140px;
  height: 140px;
  border-radius: 50%;
  border: 1px solid rgba(0, 229, 255, 0.34);
  background:
    radial-gradient(circle, rgba(0, 229, 255, 0.10) 0 1px, transparent 2px) 0 0 / 22px 22px,
    radial-gradient(circle, rgba(0, 229, 255, 0.06), transparent 68%);
  box-shadow: inset 0 0 32px rgba(0, 229, 255, 0.10), 0 0 28px rgba(0, 229, 255, 0.10);
  overflow: hidden;
}

.vc-radar::before {
  content: "";
  position: absolute;
  inset: 0;
  background: conic-gradient(from 0deg, rgba(0, 229, 255, 0.48), transparent 22%, transparent 100%);
  animation: vc-spin 2.4s linear infinite;
}

.vc-radar::after {
  content: "";
  position: absolute;
  inset: 34%;
  border: 1px solid rgba(0, 229, 255, 0.35);
  border-radius: 50%;
}

.vc-radar-text { min-width: 0; }
.vc-radar-title {
  font-family: var(--vc-heading);
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  color: var(--vc-cyan);
  text-transform: uppercase;
}
.vc-radar-stage {
  margin-top: 0.35rem;
  font-family: var(--vc-mono);
  font-size: 0.72rem;
  color: var(--vc-text);
}
.vc-radar-fn {
  margin-top: 0.28rem;
  font-family: var(--vc-mono);
  font-size: 0.6rem;
  color: var(--vc-muted);
  word-break: break-word;
}

/* ------------------------------------------------------------------ */
/* Gauges                                                               */
/* ------------------------------------------------------------------ */

.vc-gauge-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 0.7rem 0;
}

.vc-gauge {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.75rem 0.5rem 0.65rem;
  background: rgba(7, 16, 31, 0.88);
  border: 1px solid rgba(0, 229, 255, 0.16);
  clip-path: polygon(0 9px, 9px 0, 100% 0, 100% calc(100% - 9px), calc(100% - 9px) 100%, 0 100%);
  transition: border-color 180ms ease, box-shadow 180ms ease;
}

.vc-gauge:hover {
  border-color: rgba(0, 229, 255, 0.42);
  box-shadow: 0 0 24px rgba(0, 229, 255, 0.08);
}

.vc-gauge svg { width: 108px; height: 108px; overflow: visible; }

.vc-gauge .gauge-track { stroke: rgba(143, 166, 194, 0.18); }
.vc-gauge .gauge-fill {
  stroke: var(--vc-cyan);
  filter: drop-shadow(0 0 5px rgba(0, 229, 255, 0.7));
  transition: stroke-dashoffset 900ms cubic-bezier(.2,.7,.2,1);
}
.vc-gauge.amber .gauge-fill { stroke: var(--vc-amber); filter: drop-shadow(0 0 5px rgba(255, 179, 0, 0.65)); }
.vc-gauge.red .gauge-fill { stroke: var(--vc-red); filter: drop-shadow(0 0 5px rgba(255, 82, 82, 0.6)); }

.vc-gauge-value {
  position: absolute;
  top: 3.2rem;
  left: 50%;
  transform: translateX(-50%);
  font-family: var(--vc-mono);
  font-size: 1.18rem;
  color: var(--vc-text);
  white-space: nowrap;
}
.vc-gauge-unit { font-size: 0.62rem; color: var(--vc-muted); }
.vc-gauge-label {
  margin-top: 0.15rem;
  font-family: var(--vc-mono);
  font-size: 0.58rem;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--vc-muted);
  text-align: center;
}
.vc-gauge-sub {
  margin-top: 0.2rem;
  font-family: var(--vc-mono);
  font-size: 0.52rem;
  color: rgba(143, 166, 194, 0.7);
  text-align: center;
}

/* ------------------------------------------------------------------ */
/* Image panels, scan lines, tags                                      */
/* ------------------------------------------------------------------ */

.vc-image-panel {
  position: relative;
  background: rgba(6, 14, 28, 0.9);
  border: 1px solid rgba(0, 229, 255, 0.18);
  padding: 0.55rem;
  margin-bottom: 0.7rem;
  clip-path: polygon(0 9px, 9px 0, 100% 0, 100% calc(100% - 9px), calc(100% - 9px) 100%, 0 100%);
}

.vc-image-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.6rem;
  padding: 0.1rem 0.1rem 0.45rem;
}
.vc-image-title {
  font-family: var(--vc-mono);
  font-size: 0.64rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--vc-cyan);
}
.vc-image-meta {
  font-family: var(--vc-mono);
  font-size: 0.56rem;
  color: var(--vc-muted);
  white-space: nowrap;
}

.vc-img-wrap {
  position: relative;
  overflow: hidden;
  background: #030812;
  aspect-ratio: var(--vc-aspect, 4 / 3);
}

.vc-img-wrap img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.vc-scan {
  position: absolute;
  left: 0;
  right: 0;
  top: -28%;
  height: 20%;
  pointer-events: none;
  background: linear-gradient(to bottom, transparent, rgba(0, 229, 255, 0.24), rgba(230, 241, 255, 0.55), transparent);
  mix-blend-mode: screen;
  animation: vc-image-scan 2.8s ease-in-out 1 both;
}

.vc-img-wrap:hover .vc-scan { animation: vc-image-scan 1.15s ease-in-out 1 both; }

.vc-img-wrap::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  border: 1px solid rgba(0, 229, 255, 0.16);
  box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.45);
}

.vc-image-caption {
  margin-top: 0.42rem;
  font-family: var(--vc-mono);
  font-size: 0.58rem;
  color: var(--vc-muted);
}

.vc-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 26px;
  height: 22px;
  padding: 0 0.35rem;
  background: var(--vc-amber);
  color: #07101f;
  font-family: var(--vc-mono);
  font-size: 0.66rem;
  font-weight: 700;
  box-shadow: 0 0 14px rgba(255, 179, 0, 0.4);
  animation: vc-tag-in 280ms ease-out both;
}

/* ------------------------------------------------------------------ */
/* Agent feed + status chips                                           */
/* ------------------------------------------------------------------ */

.vc-feed {
  position: relative;
  margin: 0.6rem 0 0.2rem;
  padding-left: 1.05rem;
}

.vc-feed::before {
  content: "";
  position: absolute;
  left: 4px;
  top: 4px;
  bottom: 4px;
  width: 1px;
  background: linear-gradient(to bottom, rgba(0, 229, 255, 0.55), rgba(0, 229, 255, 0.05));
}

.vc-feed-entry {
  position: relative;
  padding: 0.55rem 0.65rem;
  margin-bottom: 0.45rem;
  background: rgba(7, 16, 31, 0.82);
  border: 1px solid rgba(0, 229, 255, 0.12);
  animation: vc-feed-in 360ms ease-out both;
  animation-delay: calc(var(--i, 0) * 55ms);
}

.vc-feed-entry::before {
  content: "";
  position: absolute;
  left: -1.05rem;
  top: 0.85rem;
  width: 9px;
  height: 9px;
  transform: translateX(-3px);
  border-radius: 50%;
  background: var(--vc-cyan);
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.7);
}

.vc-feed-entry.amber::before { background: var(--vc-amber); box-shadow: 0 0 10px rgba(255, 179, 0, 0.6); }
.vc-feed-entry.red::before { background: var(--vc-red); box-shadow: 0 0 10px rgba(255, 82, 82, 0.55); }

.vc-feed-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
}
.vc-feed-action {
  font-family: var(--vc-mono);
  font-size: 0.64rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--vc-cyan);
}
.vc-feed-outcome {
  font-family: var(--vc-mono);
  font-size: 0.54rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0.12rem 0.36rem;
  border: 1px solid rgba(143, 166, 194, 0.32);
  color: var(--vc-muted);
  white-space: nowrap;
}
.vc-feed-outcome.accepted { border-color: var(--vc-cyan); color: var(--vc-cyan); }
.vc-feed-outcome.rejected { border-color: var(--vc-red); color: var(--vc-red); }
.vc-feed-outcome.ambiguous { border-color: var(--vc-amber); color: var(--vc-amber); }

.vc-feed-body {
  margin-top: 0.28rem;
  font-size: 0.82rem;
  line-height: 1.42;
  color: var(--vc-text);
}
.vc-feed-note {
  margin-top: 0.24rem;
  font-family: var(--vc-mono);
  font-size: 0.56rem;
  color: var(--vc-muted);
}

/* ------------------------------------------------------------------ */
/* Misc HUD blocks                                                     */
/* ------------------------------------------------------------------ */

.vc-chip-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.35rem 0 0.7rem; }

.vc-chip {
  font-family: var(--vc-mono);
  font-size: 0.58rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.22rem 0.5rem;
  border: 1px solid rgba(0, 229, 255, 0.26);
  background: rgba(0, 229, 255, 0.05);
  color: var(--vc-muted);
}
.vc-chip.ok { border-color: rgba(0, 229, 255, 0.55); color: var(--vc-cyan); }
.vc-chip.amber { border-color: rgba(255, 179, 0, 0.55); color: var(--vc-amber); }
.vc-chip.red { border-color: rgba(255, 82, 82, 0.55); color: var(--vc-red); }

.vc-empty {
  padding: 2.2rem 1.4rem;
  border: 1px dashed rgba(0, 229, 255, 0.24);
  background: rgba(7, 16, 31, 0.55);
  text-align: center;
}
.vc-empty-title {
  font-family: var(--vc-heading);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--vc-cyan);
  font-size: 0.85rem;
}
.vc-empty-body { margin-top: 0.45rem; color: var(--vc-muted); font-size: 0.9rem; }

.vc-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; }
.vc-three-col { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }

.vc-readout-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.55rem;
}
.vc-readout {
  border: 1px solid rgba(0, 229, 255, 0.14);
  background: rgba(7, 16, 31, 0.72);
  padding: 0.55rem 0.6rem;
}
.vc-readout-label {
  font-family: var(--vc-mono);
  font-size: 0.54rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--vc-muted);
}
.vc-readout-value {
  margin-top: 0.18rem;
  font-family: var(--vc-mono);
  font-size: 1rem;
  color: var(--vc-text);
}
.vc-readout-value small { font-size: 0.58rem; color: var(--vc-muted); }

.vc-report-preview {
  max-height: 320px;
  overflow: auto;
  padding: 0.75rem;
  background: rgba(3, 8, 18, 0.9);
  border: 1px solid rgba(0, 229, 255, 0.14);
  font-family: var(--vc-mono);
  font-size: 0.62rem;
  line-height: 1.45;
  color: #b9d5ea;
  white-space: pre-wrap;
  word-break: break-word;
}

.vc-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 0.9rem;
  font-family: var(--vc-mono);
  font-size: 0.55rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--vc-muted);
}
.vc-legend span::before {
  content: "";
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-right: 5px;
  background: currentColor;
  box-shadow: 0 0 8px currentColor;
}
.vc-legend .nice { color: var(--vc-cyan); }
.vc-legend .warn { color: var(--vc-amber); }
.vc-legend .bad { color: var(--vc-red); }

.vc-note {
  font-family: var(--vc-mono);
  font-size: 0.6rem;
  line-height: 1.5;
  color: var(--vc-muted);
  border-left: 2px solid rgba(0, 229, 255, 0.32);
  padding-left: 0.65rem;
  margin: 0.55rem 0;
}

/* ------------------------------------------------------------------ */
/* Keyframes (all slow; nothing flashes faster than 3 Hz)              */
/* ------------------------------------------------------------------ */

@keyframes vc-grid-drift {
  0% { transform: perspective(780px) rotateX(61deg) translate3d(0, -10%, 0) scale(2.1); }
  50% { transform: perspective(780px) rotateX(61deg) translate3d(0, -4%, 0) scale(2.1); }
  100% { transform: perspective(780px) rotateX(61deg) translate3d(0, -10%, 0) scale(2.1); }
}

@keyframes vc-glow-pulse {
  0%, 100% { opacity: 0.52; }
  50% { opacity: 0.76; }
}

@keyframes vc-scan-drift {
  0% { background-position: 0 0, 0 -120%; }
  100% { background-position: 0 0, 0 120%; }
}

@keyframes vc-border-spin {
  to { --vc-angle: 360deg; }
}

@keyframes vc-header-sweep {
  0%, 12% { transform: translateX(-100%); opacity: 0; }
  30% { opacity: 0.8; }
  58%, 100% { transform: translateX(100%); opacity: 0; }
}

@keyframes vc-spin {
  to { transform: rotate(360deg); }
}

@keyframes vc-node-spin {
  to { transform: rotate(360deg); }
}

@keyframes vc-dot-pulse {
  0%, 100% { opacity: 0.55; }
  50% { opacity: 1; }
}

@keyframes vc-tab-in {
  from { opacity: 0; transform: translateY(7px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes vc-image-scan {
  0% { transform: translateY(0); opacity: 0; }
  12% { opacity: 1; }
  88% { opacity: 1; }
  100% { transform: translateY(650%); opacity: 0; }
}

@keyframes vc-tag-in {
  from { opacity: 0; transform: scale(0.72); }
  to { opacity: 1; transform: scale(1); }
}

@keyframes vc-feed-in {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ------------------------------------------------------------------ */
/* Reduced motion                                                      */
/* ------------------------------------------------------------------ */

html.reduce-motion *,
html.reduce-motion *::before,
html.reduce-motion *::after {
  animation-duration: 0.001ms !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0.001ms !important;
  scroll-behavior: auto !important;
}

html.reduce-motion .stApp::before { opacity: 0.45; }
html.reduce-motion .stApp::after { opacity: 0.45; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}

/* ------------------------------------------------------------------ */
/* Responsive                                                          */
/* ------------------------------------------------------------------ */

@media (max-width: 1180px) {
  .vc-gauge-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .vc-pipeline { grid-template-columns: repeat(4, minmax(0, 1fr)); row-gap: 0.7rem; }
  .vc-node:nth-child(4)::after { display: none; }
}

@media (max-width: 760px) {
  .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
  .vc-header { align-items: flex-start; gap: 0.7rem; padding: 0.85rem; clip-path: polygon(0 12px, 12px 0, calc(100% - 24px) 0, 100% 24px, 100% 100%, 0 100%); }
  .vc-reticle { flex-basis: 66px; width: 66px; height: 66px; }
  .vc-brand-title { font-size: 1.28rem; }
  .vc-brand-sub { font-size: 0.58rem; letter-spacing: 0.15em; }
  .vc-badge { display: none; }
  .vc-pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 0.85rem; }
  .vc-node::after { display: none !important; }
  .vc-gauge-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .vc-two-col, .vc-three-col, .vc-readout-grid { grid-template-columns: 1fr; }
  .vc-radar-panel { grid-template-columns: 1fr; justify-items: center; text-align: center; }
  .vc-radar { width: 112px; height: 112px; }
  .vc-image-head { flex-direction: column; align-items: flex-start; gap: 0.15rem; }
  .vc-gauge svg { width: 92px; height: 92px; }
  .vc-gauge-value { top: 2.65rem; font-size: 1rem; }
  .stTabs [role="tab"],
  [data-testid="stTabs"] [role="tab"] { padding: 0.5rem 0.62rem; font-size: 0.6rem; }
}

@media (max-width: 430px) {
  .vc-brand-title { font-size: 1.08rem; }
  .vc-statusline { font-size: 0.58rem; gap: 0.35rem 0.7rem; }
  .vc-gauge-grid { grid-template-columns: 1fr; }
  .vc-kicker { font-size: 0.6rem; }
}
"""
