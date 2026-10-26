# VISION CORE motion checklist

Scope: every motion and interaction item in the VISION CORE brief. Evidence is
given as `path:line`. Tests were run with `.venv/bin/python` (Python 3.14.4,
Streamlit 1.66.1, OpenCV 5.0.0, NumPy 2.5.4).

## Ambient motion

| Item | Status | Evidence |
| --- | --- | --- |
| Animated perspective background grid | DONE | `ui/theme.py:67` (`.stApp::before` grid layers), `ui/theme.py:82` (`vc-grid-drift`), keyframes `ui/theme.py:1046` |
| Pulsing background glow | DONE | `ui/theme.py:82` (`vc-glow-pulse`), keyframes `ui/theme.py:1052` |
| Ambient scanlines / vertical sweep | DONE | `ui/theme.py:88` (scanline layer), `ui/theme.py:111` (`vc-scan-drift`), `ui/theme.py:1057` |
| Particle canvas | DONE | `ui/templates/hero.html:207`–`294`, rendered from `ui/components.py:115` |
| Mouse parallax on particles | DONE | `ui/templates/hero.html:298` (`mousemove`), applied in `ui/templates/hero.html:258` |
| Particle DPR cap of 2 | DONE | `ui/templates/hero.html:220` (`Math.min(2, devicePixelRatio)`) |
| Particle pause when tab hidden | DONE | `ui/templates/hero.html:303` (`visibilitychange` → `stop()` / `start()`) |
| Static particle fallback under reduced motion | DONE | `ui/templates/hero.html:142` (`REDUCED`), `ui/templates/hero.html:227` (`drawStatic`), `ui/templates/hero.html:243` |
| First-load boot sequence | DONE | `ui/components.py:115` (`boot` payload), `app.py:190` (first-load flag), `ui/templates/hero.html:167`–`204` |
| Boot uses real runtime values only | DONE | `ui/components.py:106`–`115` (versions/detectors/threads passed from `ui/pipeline.py:capabilities`), `ui/templates/hero.html:174`–`179` |
| Rotating acquisition reticle | DONE | `ui/templates/hero.html:101`, orbit/arc animation rules `ui/theme.py:431`–`433` (the iframe also has its own rules at `ui/templates/hero.html:48`–`54`) |
| Animated panel borders | DONE | border implementation `ui/theme.py:343`–`366` (`vc-border-spin`, keyframes `ui/theme.py:1062`), applied to the live radar panel at `ui/components.py:195` |
| No flashing faster than 3 Hz | DONE | A/B toggle is 500 ms half-period = 1 Hz full cycle, 2 Hz visible switching: `ui/templates/register.html:520`; CSS loops are 1.8 s or slower: `ui/theme.py:1046`–`1099` |
| Mostly transform/opacity motion | DONE | keyframes use `transform`/`opacity`: `ui/theme.py:1046`–`1099`; scan line uses `transform`: `ui/templates/swipe.html:44`; region tag uses `transform`: `ui/templates/regions.html:54` |
| Reduced-motion CSS kill switch | DONE | `ui/theme.py:1111`–`1121`; component JS also checks `prefers-reduced-motion`: `ui/templates/hero.html:142`, `ui/templates/gauges.html:37`, `ui/templates/register.html:193`, `ui/templates/swipe.html:71`, `ui/templates/regions.html:74` |
| Sidebar Reduce motion toggle | DONE | `app.py:163`–`170`; CSS injection at `ui/components.py:38`–`39` |
| Responsive at 1440×900 | DONE | Headless Chrome viewport 1440×900 check: 0 px horizontal overflow, hero 980×178, gauges 980×177, REGISTER 980×631. Breakpoints: `ui/theme.py:1136`–`1167` |
| Responsive at ~390 px | DONE | Headless Chrome 390×844 check: 0 px horizontal overflow, hero 368×150, gauges 368×617, REGISTER 368×930. Breakpoints: `ui/theme.py:1142`–`1167`, widget rules `ui/templates/gauges.html:28`, `ui/templates/register.html:112` |
| Full-height app shell | DONE | `.stApp` min-height at `ui/theme.py:58`–`64`; headless check: `.stApp`, `#root`, main and sidebar measure 900 px at 1440×900 and 844 px at 390×844, and pixel probes confirm painted content |

## Live telemetry and result motion

| Item | Status | Evidence |
| --- | --- | --- |
| Real-time pipeline node strip | DONE | `ui/components.py:126`; state derived from real progress events at `ui/pipeline.py:233`; wired to the real callback at `app.py:376`–`436` |
| Real-time radar | DONE | `ui/components.py:167`; stage, phase, status, reason and function names from the real event, not a timer |
| Pipeline shows real OpenCV/NumPy function names | DONE | `ui/components.py:130`–`157`; event `functions` copied from `inspect_images` at `ui/pipeline.py:233`–`260`; a decision-only stage such as VERIFY shows its status and its tooltip/radar says `no OpenCV call recorded` instead of `awaiting` |
| Gauge count-up | DONE | `ui/templates/gauges.html:70`–`85`; numeric detection and unit guard at `ui/templates/gauges.html:48`–`52`; called from `ui/components.py:224` |
| Gauge values are real measurements | DONE | `ui/pipeline.py:263`–`352` reads `final_comparison`, `independent_comparison`, agreement and alignment fields |
| Gauge unit bug (`SPLIT%`, `n/a%`) fixed | DONE | explicit `numeric` flags at `ui/pipeline.py:296`, `:311`, `:326`, `:341`; template guard at `ui/templates/gauges.html:48`–`52` |
| Image scan line on panels | DONE | `ui/theme.py:800`–`812`; renderer `ui/components.py:235`–`247` |
| Image scan line on swipe and region widgets | DONE | `ui/templates/swipe.html:33`–`35`, `ui/templates/regions.html:22`–`24` |
| Alignment lock-on markers | DONE | `ui/templates/register.html:383`–`435`; marker rendering and transforms; marker styles `ui/templates/register.html:43`–`46` |
| Step transition animation | DONE | timeline/step select at `ui/templates/register.html:351`–`363`; main transform animation `ui/templates/register.html:527`–`559` |
| Tab transition animation | DONE | `ui/theme.py:298` (`vc-tab-in`), keyframes `ui/theme.py:1085`; tab markup at `app.py:519` |
| Agent-feed entry animation | DONE | `ui/theme.py:866`, keyframes `ui/theme.py:1102`; renderer `ui/components.py:366` |
| Region tag animations | DONE | `ui/templates/regions.html:30`–`31` (staggered `tagIn`), keyframes `ui/templates/regions.html:54`; CSS tag animation `ui/theme.py:830`–`843`; renderer `ui/components.py:301` |
| No simulated numbers | DONE | all payload values come from `inspect_images` / `build_inspection_report` in `ui/pipeline.py:355`–`542`; the cached path never reruns UI callbacks (`ui/pipeline.py:502`–`534`) |

## REGISTER interaction

| Item | Status | Evidence |
| --- | --- | --- |
| Step timeline | DONE | `ui/templates/register.html:168`, built at `ui/templates/register.html:351` |
| Prev / Next / Play controls | DONE | markup `ui/templates/register.html:170`–`172`, handlers `ui/templates/register.html:622`–`624` |
| Scrubber | DONE | markup `ui/templates/register.html:173`–`174`, handler `ui/templates/register.html:613`–`617` |
| Blend overlay | DONE | button `ui/templates/register.html:177`, mode handling `ui/templates/register.html:485`–`490` |
| Checkerboard overlay | DONE | button `ui/templates/register.html:178`, mode handling `ui/templates/register.html:487`–`491` |
| Difference overlay | DONE | button `ui/templates/register.html:179`, mode handling `ui/templates/register.html:492`–`494` |
| Edge overlay | DONE | button `ui/templates/register.html:180`, edge layers `ui/templates/register.html:39`–`40`, mode `ui/templates/register.html:495`–`500` |
| A/B flicker at ≤2 Hz | DONE | button `ui/templates/register.html:181`, interval `ui/templates/register.html:520` |
| Swipe overlay | DONE | button `ui/templates/register.html:182`, control `ui/templates/register.html:142`–`143`, handler `ui/templates/register.html:511`–`514` |
| OpenCV fidelity toggle | DONE | `ui/templates/register.html:186`; toggles real step preview vs animated transform at `ui/templates/register.html:470`–`483` |
| LOCKED / AMBIGUOUS / REJECTED / NO ALIGNMENT badges | DONE | status labels `ui/templates/register.html:248`–`255`, badge markup `ui/templates/register.html:137`; result chips `app.py:504`–`516` via `app.py:77` |
| Metrics + reason | DONE | `ui/templates/register.html:451`–`466` |
| Residual chart | DONE | `ui/templates/register.html:365`–`374` |
| Visible orientation step | DONE | real step rows carry `model == "orientation"` from `inspectai/agent.py:359`–`390`; REGISTER lists every step in the timeline and the page adds a caption at `app.py:539`–`548` |
| Ambiguous result opens on an ambiguous candidate | DONE | `ui/pipeline.py:198`–`230` selects the last ambiguous preview; headless audit of the grid pair opened STEP 02 with badge AMBIGUOUS |
| Skipped Final step is labelled NO ALIGNMENT | DONE | `ui/templates/register.html:248`–`255`; a skipped Final step no longer claims SKIPPED in the badge or metrics row |

## Honesty, disclosure and accessibility

| Item | Status | Evidence |
| --- | --- | --- |
| Gauges labelled uncalibrated heuristics | DONE | `ui/templates/gauges.html:34`; gauge aria strings `ui/pipeline.py:283`–`343` |
| LIMITATIONS always reachable | DONE | sidebar expander `app.py:173`–`183`; result limitations expander `app.py:683`; footer disclosure `app.py:710` |
| Agent decisions linked into VERIFY | DONE | payload fields `ui/pipeline.py:468`–`470`; rendered at `app.py:669` with a real no-silent-correction message (for example the 270° orientation fix) |
| Method-disagreement message names the actual methods | DONE | `app.py:621`–`632`; SSIM indication is serialized at `ui/pipeline.py:466`; real states: shifted = Lab no / Canny no / SSIM yes |
| Current Streamlit button and tab DOM skinned | DONE | tooltip-wrapped button selectors `ui/theme.py:169`–`202`; ARIA tab selectors `ui/theme.py:261`–`299`; headless audit read 2 px radius, Orbitron labels and a cyan selected-tab underline |
| iframe widgets have accessible titles | DONE | `ui/components.py:69`–`70` passes `alt` to `st.iframe`; hero, gauges, REGISTER, regions and swipe call sites supply real descriptions |
| Forbidden verdict wording guard | DONE | `tests/test_progress_and_report.py:224`–`247` |
| Keyboard focus rings | DONE | `ui/theme.py:153`–`165`; in-widget rules `ui/templates/register.html:18`, `ui/templates/swipe.html:40` |
| SVG/gauge aria labels | DONE | reticle `ui/templates/hero.html:101`, gauges `ui/templates/gauges.html:54`, residuals `ui/templates/register.html:374`, regions `ui/templates/regions.html:65` and `:109` |
| Keyboard-operable region boxes | DONE | `ui/templates/regions.html:108` (`tabindex`), `ui/templates/regions.html:112`–`115` (Enter/Space), selection at `:150`–`155` |
| Interactive region boxes wired | DONE | `boxes`/`tags` declared outside the empty branch at `ui/templates/regions.html:92`–`93`; headless audit dispatched a click and an Enter keydown, selecting the clicked box and the keyboard-focused box and printing their measured bounds |
| Keyboard step navigation | DONE | `ui/templates/register.html:633`–`636` (arrow keys) |
| Pause hidden-tab registration playback / flicker | DONE | `ui/templates/register.html:637`–`639` |
| OpenCV logo not generated/hotlinked | DONE | `ui/pipeline.py:67`–`78` returns only a user-supplied local asset; fallback badge `ui/templates/hero.html:153`–`159`; one warning `app.py:44`–`48` |

## Verification performed

- `python -m unittest discover -s tests -q` → **43 tests, OK**.
- AppTest clicked every demo button; all eight demo cases rendered three tabs,
  five iframes and zero exceptions. Measured results: shifted →
  locked/translation, 0.00%, 0 regions; rotated 15° → locked/similarity, 0.08%,
  4 regions; rotated 90° → locked/orientation, 0.00%, 0 regions; rotated 180° →
  locked/orientation, 0.00%, 0 regions; brighter → not needed / photometric,
  4.77%, 30 regions; perspective → locked/affine, 0.79%, 21 regions; pasted
  patch → not needed / none, 0.78%, 1 region; periodic grid →
  **ambiguous/none**, 0.00%, 0 regions.
- The 90° demo now autoruns without enabling crop: the aspect gate recognises
  the reciprocal quarter-turn shape and the result summary reads
  `ALIGNMENT LOCKED · METHOD ORIENTATION · CHANGED 0.00%`.
- A screenshot pixel probe caught a regression that DOM-only checks missed:
  `.stApp` had collapsed to 0 px and clipped every panel, leaving only the themed
  background. After adding `min-height: 100vh` / `100dvh`
  (`ui/theme.py:58`–`64`), `.stApp`, `#root`, main and sidebar measure 900 px at
  1440×900 and 844 px at 390×844. The same screenshots contain painted
  text/panels (3,204 bright samples on desktop; 1,020 on mobile after collapsing
  the sidebar).
- Headless Chrome via CDP after the shell fix: REGISTER, ANALYZE and VERIFY each
  size their iframes when selected (desktop REGISTER 980×631, ANALYZE regions
  980×854 and swipe 980×806). A 390×844 viewport reported
  `scrollWidth == clientWidth` (0 px overflow); with the sidebar open it overlays
  the app, and collapsing it returns a 390 px main column. Hidden tabs stay 0×0
  by design.
- The ambiguous grid result opens on STEP 02 with badge `AMBIGUOUS`; a skipped
  Final step shows `NO ALIGNMENT` instead of implying a lock.
- Region interaction was exercised in the DOM: a dispatched click selected R1
  (`R1 · x 383, y 385 · 11 x 10 px · area 73 px`) and an Enter keydown on the
  next box selected R2 (`R2 · x 445, y 381 · 13 x 12 px · area 63 px`), each
  with its measured bounds.
- VERIFY exposes `Agent decisions` and `Limitations` expanders; the 90° run
  listed “Applied a 270-degree clockwise orientation fix before shape checks.”
- The current Streamlit DOM is skinned: the help-wrapped Run button computed to a
  2 px radius with an Orbitron label, and selected/unselected tabs used the cyan
  underline, muted text and Orbitron label intended by `ui/theme.py`.
- No browser exception was recorded across the audited demos. The only console
  entries were Chrome warnings about unrecognised iframe features and
  Streamlit's own sandbox note.
- Automated evidence above came from the DOM plus screenshot pixel probes, not
  from hand-inspecting images. The temporary headless-Chrome scripts and
  screenshots are not committed; the reproducible checks are the AppTest sweep
  and the unit test suite.

