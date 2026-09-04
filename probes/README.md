# Probes

**Rule: no code enters `src/` until a probe script proves the feature standalone.**

A probe is a single file that asks one question, answers it by running, and is
then left in the repo forever. It records *how* each piece was validated, which
is a different claim from *that* it is tested — the pytest suite locks the
behaviour after the fact; the probe is the evidence the behaviour was ever
looked at with human eyes.

Every probe carries this header:

```python
"""
probes/p04_tiling.py

QUESTION:  Does the tiler produce 512x512 tiles with 64px overlap and
           correct global coordinates for a 1920x1080 frame?
PASS IF:   Reassembling the tiles reproduces the input exactly, and every
           tile's stored (x, y) maps back to its true origin.
ANSWERED:  yes - see the status table below.
LIVES IN:  linesight/preprocess/tiling.py, locked by tests/test_tiling.py.
"""
```

Workflow: run it, look at the output, record the answer, move the function into
`src/`, write the pytest that locks the behaviour. `LIVES IN` is the pointer
from a question to the code that answers it, which is the fastest way into an
unfamiliar layer.

Run any probe with the project venv:

```bash
~/.venvs/linesight/Scripts/python.exe probes/p02_memory_bank.py
```

## Status table

A row is only ever raised by running the thing. Where a probe's question has
been answered on **synthetic** fabric but not yet on real data, it says so: a
green tick against data nobody has looked at would be exactly the kind of claim
this project is built not to make.

| Probe | Question | Status | Promoted to |
|---|---|---|---|
| `p01_feature_extract.py` | Does the frozen backbone give a (20, 20, 384) grid from one AITEX crop? | **ANSWERED on synthetic** 2026-08-31 -- got exactly `(1, 20, 20, 384)`, 400 patches/tile, finite, spatially varying. Rerun on a real AITEX crop. | `detect/backbone.py` |
| `p02_memory_bank.py` | Fit on 20 defect-free crops of one fabric code, score a defective crop of the same code -- does the defect glow, and does it overlap the `_mask.png`? | **ANSWERED on synthetic** 2026-08-31 -- fit 2.1 s / 20 tiles, coreset 8000->800, peak 17.4 on the defect vs 1.72 on held-out clean (10.1x), pixel AUROC 0.993, every GT pixel above the clean tile's max. Confirmed on real fabric by `p12`. | `detect/patchcore.py` |
| `p03_coreset.py` | Does greedy k-center on 12,000 embeddings finish in reasonable time, and how much AUROC does random subsampling cost? | PARTIAL -- greedy is **quadratic in fit-set size** (k scales with N, so O(N^2 x frac)): 30 tiles 2.8 s, 60 tiles 9.0 s, and 300 tiles did not finish in 10 minutes. Random handles 150 tiles in 4.1 s at equal separation here. Above ~50 tiles, random is the practical choice; the accuracy cost is priced by `p11`'s third configuration -- greedy beats random by 3.6 points at an identical 1200-point bank. | `detect/patchcore.py` |
| `p04_tiling.py` | 512x512 tiles, 64 px overlap, correct global coords on a 1920x1080 frame? | **PASSED** 2026-08-31 -- exact round-trip and coordinate mapping locked by `tests/test_tiling.py` (17 tests). | `preprocess/tiling.py` |
| `p05_flatfield.py` | Does flat-field correction flatten the illumination without eating a 10 px defect? | IMPLEMENTED, and **measured as a no-op on AITEX**: separation 1.35x without vs 1.31x with, across 4 defects and three fit-set sizes. AITEX is evenly-lit scanned imagery, so the uneven-illumination failure this corrects for does not arise there. Still expected to matter on the rig, where it is untested. | `preprocess/flatfield.py` |
| `p06_aruco.py` | Does one `cv2.aruco` call give absolute ID *and* a mm/px scale within 2% of a ruler measurement? | **PASSED on synthetic markers** 2026-08-31 -- ID and sub-pixel scale recovered from generated markers; gap warning fires on a long dropout and clears when a marker returns (`tests/test_geometry.py`, 11 tests). Not yet checked against a physical ruler. | `geometry/aruco.py` |
| `p07_threshold.py` | On held-out clean tiles, does a stated budget of 1 FA/100 m actually produce ~1 FA/100 m? | **PASSED on synthetic scores** 2026-08-31 -- realised rate matches the budget, and the sample-size guard refuses a budget the sample cannot resolve (`tests/test_threshold.py`, 9 tests). Not yet run on real fabric scores. | `calibrate/threshold.py` |
| `p08_events.py` | Do eight consecutive fragments of one warp line become one event of the right length? | **PASSED** 2026-08-31 -- one event, 8 frames, 285 mm, and it survives a one-frame dropout (`tests/test_events.py`, 15 tests). | `events/track.py` |
| `p09_mark_defects.py` | **The standalone defect marker.** Take one defective image, draw a box around the defect, save it next to the ground-truth mask. | **PASSED on real AITEX** 2026-09-01 -- fabric 02 image `0011_006_02`: three detections, all on the defect, zero false alarms, peak inside the ground-truth mask. Boxes visibly track the mask contour. Harder fabrics (04) are mixed: `0043_019_04` hits, `0044_019_04` misses. | -- (verification only) |
| `p10_report.py` | Does a RollReport render to a PDF a mill would file -- verdict, calibration provenance, defect table, positional map, image evidence, limitations -- and does the text fallback carry the same facts? | **PASSED** 2026-09-02 -- a synthetic roll carrying an asserted, a continuous, an uncertain and an operator-rejected event renders 5 pages; points match the ASTM scorer (1+3+8+0+0); an uncalibrated roll is REFUSED (`tests/test_report.py`, 14 tests). | `report/pdf.py`, `report/defect_map.py` |
| `p11_mvtec_reproduction.py` | Is the hand-written PatchCore (ADR-001) actually correct -- does it land near the published image AUROC on MVTec's texture classes? | **PASSED on 2 of 3** 2026-09-02 -- carpet **0.9615**, leather **0.9840** in the shipped 30-image / 10%-greedy configuration; grid fails at **0.5522** and recovers to 0.8062 at backbone input 512, which localises it as a resolution limit rather than a bug. Greedy beats random by 3.6 pts at an identical 1200-point bank. The detector is sound on textured surfaces. | -- (verification only) |
| `p12_aitex_generalisation.py` | On real fabric: does a per-fabric bank detect that fabric's own defects, and does a bank transfer to a construction it never saw? | **PASSED** 2026-09-02 -- cold start over 6 AITEX fabrics: tile AUROC **0.859**, pixel AUROC **0.971**. Transfer from a different fabric: 0.629 / 0.739. The **+0.230** in-distribution advantage is the measured argument for refitting per SKU. | -- (verification only) |
| `p13_phone_stream.py` | Does the camera deliver a usable stream -- steady frame rate, sharp fabric, locked exposure, and a decodable ArUco tape? | **PASSED against a synthetic MJPEG source** 2026-09-02 -- 10.6 fps at 1280x720, markers decoded in 107/107 frames, and **mm/px recovered as 0.5011 against a true 0.5000 (0.2% error)**, which validates the scale chain end to end. Not yet run against a physical phone. | -- (rig bring-up) |
| `p14_position_trace.py` | Does one physical defect report the same roll position in every frame that sees it? | DIAGNOSTIC -- prints marker ID, mm/px and the marker-relative position per frame, so a position fault localises to one of its two inputs rather than to "the geometry". | -- (rig diagnosis) |

`p09` is deliberately not promoted anywhere. It exists as a single-file,
top-to-bottom proof that the whole chain -- backbone, bank, distance, upsample,
threshold, connected components -- works on a real image, with an output a human
can assess in one second.

`p13` and `p14` are rig probes: they answer questions about the camera and the
printed tape rather than about a module, so they stay standalone scripts.
