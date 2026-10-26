# VISION CORE motion checklist

Scope: every motion and interaction item in the VISION CORE brief. Evidence is
given as `path:line`. Tests were run with `.venv/bin/python` (Python 3.14.4,
Streamlit 1.66.1, OpenCV 5.0.0, NumPy 2.5.4).

## Ambient motion

| Item | Status | Evidence |
| --- | --- | --- |
| Animated perspective background grid | DONE | `ui/theme.py:65` (fixed grid layers), `ui/theme.py:80` (`vc-grid-drift`), keyframes `ui/theme.py:1033` |
| Pulsing background glow | DONE | `ui/theme.py:80` (`vc-glow-pulse`), keyframes `ui/theme.py:1039` |
| Ambient scanlines / vertical sweep | DONE | `ui/theme.py:86` (scanline layer), `ui/theme.py:109` (`vc-scan-drift`), `ui/theme.py:1044` |
| Particle canvas | DONE | `ui/templates/hero.html:207`–`294`, rendered from `ui/components.py:93` |
| Mouse parallax on particles | DONE | `ui/templates/hero.html:298` (`mousemove`), applied in `ui/templates/hero.html:258` |
| Particle DPR cap of 2 | DONE | `ui/templates/hero.html:220` (`Math.min(2, devicePixelRatio)`) |
| Particle pause when tab hidden | DONE | `ui/templates/hero.html:303` (`visibilitychange` → `stop()` / `start()`) |
| Static particle fallback under reduced motion | DONE | `ui/templates/hero.html:142` (`REDUCED`), `ui/templates/hero.html:227` (`drawStatic`), `ui/templates/hero.html:243` |
| First-load boot sequence | DONE | `ui/components.py:93` (`boot` payload), `app.py:190` (first-load flag), `ui/templates/hero.html:167`–`204` |
| Boot uses real runtime values only | DONE | `ui/components.py:98`–`106` (versions/detectors/threads passed from `ui/pipeline.py:capabilities`), `ui/templates/hero.html:174`–`179` |
| Rotating acquisition reticle | DONE | `ui/templates/hero.html:101`, orbit/arc animation rules `ui/theme.py:411`–`421` (the iframe also has its own rules at `ui/templates/hero.html:48`–`54`) |
| Animated panel borders | DONE | border implementation `ui/theme.py:343`–`356` (`vc-border-spin`, `ui/theme.py:1049`), applied to the live radar panel at `ui/components.py:171` |
| No flashing faster than 3 Hz | DONE | A/B toggle is 500 ms half-period = 1 Hz full cycle, 2 Hz visible switching: `ui/templates/register.html:520`; CSS loops are 1.8 s or slower: `ui/theme.py:1033`–`1096` |
| Mostly transform/opacity motion | DONE | keyframes use `transform`/`opacity`: `ui/theme.py:1033`–`1096`; scan line uses `transform`: `ui/templates/swipe.html:44`; region tag uses `transform`: `ui/templates/regions.html:54` |
| Reduced-motion CSS kill switch | DONE | `ui/theme.py:1098`–`1116`; component JS also checks `prefers-reduced-motion`: `ui/templates/hero.html:142`, `ui/templates/gauges.html:37`, `ui/templates/register.html:193`, `ui/templates/swipe.html:71`, `ui/templates/regions.html:74` |
| Sidebar Reduce motion toggle | DONE | `app.py:163`–`170`; CSS injection at `ui/components.py:38`–`39` |
| Responsive at 1440×900 | DONE | Headless Chrome viewport 1440×900 check: 0 px horizontal overflow, hero 980×178, gauges 980×166, REGISTER 980×631. Breakpoints: `ui/theme.py:1123`–`1158` |
| Responsive at ~390 px | DONE | Headless Chrome 390×844 check: 0 px horizontal overflow, hero 368×178, gauges 368×607, REGISTER 368×908. Breakpoints: `ui/theme.py:1129`–`1158`, widget rules `ui/templates/gauges.html:28`, `ui/templates/register.html:112` |

## Live telemetry and result motion

| Item | Status | Evidence |
| --- | --- | --- |
| Real-time pipeline node strip | DONE | `ui/components.py:123`; state derived from real progress events at `ui/pipeline.py:227`; wired to the real callback at `app.py:381`–`401` |
| Real-time radar | DONE | `ui/components.py:151`; stage, phase, status, reason and function names from the real event, not a timer |
| Pipeline shows real OpenCV/NumPy function names | DONE | `ui/components.py:127`–`139`; event `functions` copied from `inspect_images` at `ui/pipeline.py:356`–`364`; names emitted by `inspectai/agent.py:318`–`322` |
| Gauge count-up | DONE | `ui/templates/gauges.html:70`–`85`; numeric detection and unit guard at `ui/templates/gauges.html:48`–`52`; called from `ui/components.py:191` |
| Gauge values are real measurements | DONE | `ui/pipeline.py:257`–`344` reads `final_comparison`, `independent_comparison`, agreement and alignment fields |
| Gauge unit bug (`SPLIT%`, `n/a%`) fixed | DONE | explicit `numeric` flags at `ui/pipeline.py:290`, `:305`, `:320`, `:335`; template guard at `ui/templates/gauges.html:48`–`52` |
| Image scan line on panels | DONE | `ui/theme.py:787`–`799`; renderer `ui/components.py:208` |
| Image scan line on swipe and region widgets | DONE | `ui/templates/swipe.html:33`–`35`, `ui/templates/regions.html:22`–`24` |
| Alignment lock-on markers | DONE | `ui/templates/register.html:383`–`435`; marker rendering and transforms; marker styles `ui/templates/register.html:43`–`46` |
| Step transition animation | DONE | timeline/step select at `ui/templates/register.html:351`–`363`; main transform animation `ui/templates/register.html:527`–`559` |
| Tab transition animation | DONE | `ui/theme.py:285` (`vc-tab-in`), `ui/theme.py:1072`; tab markup at `app.py:491` |
| Agent-feed entry animation | DONE | `ui/theme.py:859`, keyframes `ui/theme.py:1089`; renderer `ui/components.py:334` |
| Region tag animations | DONE | `ui/templates/regions.html:30`–`31` (staggered `tagIn`), keyframes `ui/templates/regions.html:54`; CSS tag animation `ui/theme.py:830`; renderer `ui/components.py:273` |
| No simulated numbers | DONE | all payload values come from `inspect_images` / `build_inspection_report` in `ui/pipeline.py:362`–`489`; the cached path never reruns UI callbacks (`ui/pipeline.py:506`–`538`) |

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
| LOCKED / AMBIGUOUS / REJECTED badges | DONE | `ui/templates/register.html:249`–`251`, badge markup `ui/templates/register.html:137`; result chips `app.py:478` via `app.py:77` |
| Metrics + reason | DONE | `ui/templates/register.html:451`–`466` |
| Residual chart | DONE | `ui/templates/register.html:365`–`374` |
| Visible orientation step | DONE | real step rows carry `model == "orientation"` from `inspectai/agent.py:359`–`390`; REGISTER lists every step in the timeline and the page adds a caption at `app.py:511`–`518` |

## Honesty, disclosure and accessibility

| Item | Status | Evidence |
| --- | --- | --- |
| Gauges labelled uncalibrated heuristics | DONE | `ui/templates/gauges.html:34`; gauge aria strings `ui/pipeline.py:283`–`343` |
| LIMITATIONS always reachable | DONE | sidebar expander `app.py:172`–`183`; footer disclosure `app.py:663`–`668` |
| Method-disagreement message names the actual methods | DONE | `app.py:586`–`598`; SSIM indication is serialized at `ui/pipeline.py:466`; real states: shifted = Lab no / Canny no / SSIM yes |
| Forbidden verdict wording guard | DONE | `tests/test_progress_and_report.py:224`–`247` |
| Keyboard focus rings | DONE | `ui/theme.py:151`–`163`; in-widget rules `ui/templates/register.html:18`, `ui/templates/swipe.html:40` |
| SVG/gauge aria labels | DONE | reticle `ui/templates/hero.html:101`, gauges `ui/templates/gauges.html:54`, residuals `ui/templates/register.html:374`, regions `ui/templates/regions.html:65` and `:109` |
| Keyboard-operable region boxes | DONE | `ui/templates/regions.html:107` (`tabindex`), `:111` (Enter/Space) |
| Keyboard step navigation | DONE | `ui/templates/register.html:633`–`636` (arrow keys) |
| Pause hidden-tab registration playback / flicker | DONE | `ui/templates/register.html:637`–`639` |
| OpenCV logo not generated/hotlinked | DONE | `ui/pipeline.py:67`–`78` returns only a user-supplied local asset; fallback badge `ui/templates/hero.html:153`–`159`; one warning `app.py:44`–`48` |

## Verification performed

- `python -m unittest discover -s tests -q` → **43 tests, OK**.
- AppTest clicked every demo button; all five demonstrated cases rendered three
  tabs, five iframes and zero exceptions. Results: shifted → locked/translation,
  0.00%, 0 regions; rotated 15° → locked/similarity, 0.08%, 4 regions; rotated
  180° → locked/orientation, 0.00%, 0 regions; brighter → not needed /
  photometric, 4.77%, 30 regions; periodic grid → **ambiguous/none**, 0.00%,
  0 regions.
- Headless Chrome via CDP at 1440×900 and 390×844: 0 px horizontal overflow;
  REGISTER, ANALYZE and VERIFY each size their iframes when selected (desktop
  REGISTER 980×631, ANALYZE region 980×150 and swipe 980×806; mobile REGISTER
  368×908, ANALYZE region 368×195 and swipe 368×806). Hidden tabs stay 0×0 by
  design.
- The shifted demo shows the real disagreement text: “Lab ΔE no change
  indicated; Canny edges no change indicated; SSIM indicates change.”
- No browser exception was recorded. The only console entries were Chrome
  warnings about unrecognised iframe features and Streamlit's own sandbox note.
- Screenshots were captured at `/tmp/opencode/ui_tab_register.png`,
  `ui_tab_analyze.png`, `ui_tab_verify.png`, `ui_mobile_register.png` and
  `ui_mobile_analyze.png`. Automated evidence above came from the DOM, not from
  hand-inspecting the PNGs.
