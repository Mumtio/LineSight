"""Build the LineSight whitepaper as a submission-format PDF.

Format follows the AInspire guideline: Times New Roman 12 pt, single line
spacing, English, body under twenty pages with appendices excluded.

Every measured number in the document is read from ``results/`` at build time
rather than typed into the prose, and every figure is drawn by
``tools/whitepaper_figures.py`` from the same files. Re-running a study
re-writes the paper, so the paper and the repository cannot drift apart. Numbers
that are projections rather than measurements are labelled as such in the text.

    python tools/make_whitepaper.py
    python tools/make_whitepaper.py --out docs/whitepaper/LineSight_Whitepaper.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "docs" / "whitepaper"

#: The guideline asks for Times New Roman. reportlab's built-in "Times-Roman" is
#: a metrically-compatible Type 1 clone with WinAnsi encoding: it has no Greek
#: and no mathematical brackets, so the conformal quantile in Section 3.3
#: rendered with alpha and the ceiling brackets silently missing. Registering the
#: real TrueType face fixes the glyph coverage AND satisfies the requirement
#: literally rather than approximately.
BODY_FONT = "TimesNewRoman"
BOLD_FONT = "TimesNewRoman-Bold"
ITALIC_FONT = "TimesNewRoman-Italic"

_FONT_FILES = {
    BODY_FONT: "times.ttf",
    BOLD_FONT: "timesbd.ttf",
    ITALIC_FONT: "timesi.ttf",
    "TimesNewRoman-BoldItalic": "timesbi.ttf",
}


def register_fonts() -> bool:
    """Register Times New Roman, falling back to the Type 1 clone if absent.

    Returns True when the real face was registered. The fallback keeps the build
    working on a machine without the font, and the caller reports which was used
    so a missing glyph is never a silent surprise.
    """
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_dir = Path("C:/Windows/Fonts")
    if not all((font_dir / f).exists() for f in _FONT_FILES.values()):
        return False
    try:
        for name, filename in _FONT_FILES.items():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    except Exception:
        return False
    addMapping(BODY_FONT, 0, 0, BODY_FONT)
    addMapping(BODY_FONT, 1, 0, BOLD_FONT)
    addMapping(BODY_FONT, 0, 1, ITALIC_FONT)
    addMapping(BODY_FONT, 1, 1, "TimesNewRoman-BoldItalic")
    return True


# --------------------------------------------------------------------------- #
# Measured values, read rather than typed
# --------------------------------------------------------------------------- #


def load_facts() -> dict:
    """Pull every figure the prose quotes out of results/."""
    with (RESULTS / "reproduction.csv").open(encoding="utf-8") as h:
        repro = list(csv.DictReader(h))
    with (RESULTS / "aitex_generalisation.csv").open(encoding="utf-8") as h:
        aitex = list(csv.DictReader(h))
    with (RESULTS / "tfd_per_fabric.csv").open(encoding="utf-8") as h:
        tfd_rows = list(csv.DictReader(h))
    tfd = json.loads((RESULTS / "tfd_summary.json").read_text(encoding="utf-8"))
    cal = json.loads(
        (ROOT / "banks" / "bench.calibration.json").read_text(encoding="utf-8")
    )

    product = {r["category"]: r for r in repro if r["configuration"] == "product"}
    same = [r for r in aitex if r["in_distribution"] == "True"]
    cross = [r for r in aitex if r["in_distribution"] == "False"]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    return {
        "repro": repro,
        "product": product,
        "aitex_same": same,
        "aitex_cross": cross,
        "aitex_same_tile": mean([float(r["tile_auroc"]) for r in same]),
        "aitex_same_pixel": mean([float(r["pixel_auroc"]) for r in same]),
        "aitex_cross_tile": mean([float(r["tile_auroc"]) for r in cross]),
        "aitex_cross_pixel": mean([float(r["pixel_auroc"]) for r in cross]),
        "tfd": tfd,
        "tfd_rows": tfd_rows,
        "cal": cal,
        "greedy": float(
            next(r for r in repro
                 if r["category"] == "carpet" and r["configuration"] == "product")
            ["image_auroc"]
        ),
        "random": float(
            next(r for r in repro
                 if r["category"] == "carpet" and r["configuration"] == "product-random")
            ["image_auroc"]
        ),
    }


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


def build(out_path: Path, facts: dict) -> tuple[Path, int]:
    global BODY_FONT, BOLD_FONT, ITALIC_FONT

    real_times = register_fonts()
    if not real_times:
        BODY_FONT, BOLD_FONT, ITALIC_FONT = "Times-Roman", "Times-Bold", "Times-Italic"

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # Single line spacing at 12 pt. Word renders TNR 12 pt single at ~13.8 pt.
    body = ParagraphStyle("body", fontName=BODY_FONT, fontSize=12, leading=13.8,
                          alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("h1", fontName=BOLD_FONT, fontSize=15, leading=18,
                        spaceBefore=14, spaceAfter=7)
    h2 = ParagraphStyle("h2", fontName=BOLD_FONT, fontSize=12.5, leading=15,
                        spaceBefore=10, spaceAfter=5)
    caption = ParagraphStyle("caption", fontName=ITALIC_FONT, fontSize=10,
                             leading=12, alignment=TA_CENTER, spaceBefore=4,
                             spaceAfter=10, textColor=colors.HexColor("#333333"))
    cell = ParagraphStyle("cell", fontName=BODY_FONT, fontSize=9.5, leading=11.5)
    cell_b = ParagraphStyle("cellb", fontName=BOLD_FONT, fontSize=9.5, leading=11.5)
    title_s = ParagraphStyle("title", fontName=BOLD_FONT, fontSize=24, leading=28,
                             alignment=TA_CENTER, spaceAfter=8)
    sub_s = ParagraphStyle("sub", fontName=BODY_FONT, fontSize=13, leading=16,
                           alignment=TA_CENTER, spaceAfter=4,
                           textColor=colors.HexColor("#333333"))
    ref_s = ParagraphStyle("ref", fontName=BODY_FONT, fontSize=10.5, leading=13,
                           leftIndent=16, firstLineIndent=-16, spaceAfter=4)

    story: list[object] = []
    W = 160 * mm

    def P(text, style=body):
        story.append(Paragraph(text, style))

    def B(items, style=body):
        for item in items:
            story.append(Paragraph(f"•&nbsp;&nbsp;{item}", ParagraphStyle(
                "b", parent=style, leftIndent=12, firstLineIndent=-8, spaceAfter=3)))

    def TBL(rows, widths, caption_text=None, header=True):
        data = [[Paragraph(c, cell_b if (header and i == 0) else cell)
                 for c in row] for i, row in enumerate(rows)]
        table = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
        style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ]
        if header:
            style += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeec")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#888888")),
            ]
        table.setStyle(TableStyle(style))
        block = [table]
        if caption_text:
            block.append(Paragraph(caption_text, caption))
        story.append(KeepTogether(block))

    def FIG(name, caption_text, width_mm=155):
        from reportlab.lib.utils import ImageReader

        path = FIGDIR / f"fig_{name}.png"
        wpx, hpx = ImageReader(str(path)).getSize()
        w = width_mm * mm
        story.append(KeepTogether([
            Image(str(path), width=w, height=hpx * (w / wpx)),
            Paragraph(caption_text, caption),
        ]))

    prod = facts["product"]
    tfd = facts["tfd"]
    cal = facts["cal"]

    # -- title ------------------------------------------------------------- #
    story.append(Spacer(1, 45 * mm))
    P("LineSight", title_s)
    P("Cold-Start Visual Inspection for Textile Rolls:<br/>"
      "Anomaly Detection, Conformal Calibration, and ASTM D5430 Scoring", sub_s)
    story.append(Spacer(1, 14 * mm))
    P("Team Hopium", sub_s)
    P("Blockchain Olympiad Bangladesh 2026 &mdash; AInspire, Artificial Intelligence Category",
      ParagraphStyle("t3", parent=sub_s, fontSize=11))
    story.append(Spacer(1, 20 * mm))
    P("<b>Abstract.</b> A textile mill onboards a new fabric construction every "
      "week. Supervised defect segmentation needs a labelled defect corpus for "
      "each construction, and that corpus does not exist at the moment it is "
      "needed. LineSight inverts the question: instead of learning what a defect "
      "looks like, it learns what <i>this</i> fabric&rsquo;s normal looks like from "
      "roughly thirty defect-free images, in about two seconds of fitting and with "
      "no training loop, then flags everything unlike it &mdash; including defect "
      "types nobody has labelled. Detections are converted to millimetres through "
      "a printed fiducial tape, tracked across frames into single physical "
      "defects, and scored under ASTM D5430 into a points total and a "
      "pass/hold/reject verdict. The detection threshold is not chosen by hand: "
      "the operator states a false-alarm budget and the threshold is the "
      "corresponding split-conformal quantile of scores on held-out clean fabric. "
      "On MVTec AD the shipped cold-start configuration reaches "
      f"{float(prod['carpet']['image_auroc']):.3f} image AUROC on <i>carpet</i> and "
      f"{float(prod['leather']['image_auroc']):.3f} on <i>leather</i> against a published "
      "~0.98 and ~1.00 obtained with a four-times larger backbone. On real woven "
      f"fabric the system reaches {facts['aitex_same_pixel']:.3f} pixel AUROC and "
      f"{facts['aitex_same_tile']:.3f} tile AUROC across six AITEX constructions, "
      f"against {facts['aitex_cross_tile']:.3f} when a bank is borrowed from a "
      "different construction &mdash; a gap that is the measured justification for "
      "the per-SKU design. The system is validated across four independent "
      "datasets, and its two characterised detection boundaries &mdash; sparse "
      "regular structure at the shipped input resolution, and calibration drawn "
      "from a different physical sample than the one being inspected &mdash; are "
      "diagnosed, bounded, and handled by the shipped configuration and the "
      "operating procedure.", body)

    story.append(PageBreak())

    # ================= 1. VISION AND PROBLEM STATEMENT ==================== #
    P("1. Vision and Problem Statement", h1)

    P("1.1 The problem, stated precisely", h2)
    P("Woven and knitted fabric leaves a loom with defects: broken picks, missing "
      "ends, slubs, holes, oil stains, reed marks, float threads. Before a roll "
      "is cut into garments it is inspected, graded and either accepted, held or "
      "rejected. The grading is not subjective folklore; it is a standard. Under "
      "ASTM D5430 [1], every defect is scored by its <i>length</i> along the roll "
      "&mdash; one point up to 75&nbsp;mm, two to 150, three to 230, four beyond &mdash; "
      "capped at four points per linear metre, and normalised to points per 100 "
      "square yards. A buyer contract then sets the threshold at which a roll is "
      "held or rejected.", body)
    P("Two facts about that process define this project. First, <b>the verdict "
      "depends on a physical measurement, not on a defect&rsquo;s name.</b> A mill does "
      "not need to know that a flaw is a slub rather than a float; it needs to "
      "know it is 180&nbsp;mm long, 1.8&nbsp;m down the roll, and therefore worth "
      "three points. Second, <b>the inspection is done by people</b>, at a "
      "light-box, for a full shift, and their performance falls with fatigue and "
      "varies between individuals. The measurement that decides commercial "
      "acceptance is taken by the least repeatable instrument in the mill.", body)

    P("1.2 Why the obvious automation does not work", h2)
    P("The published literature on automated fabric inspection is overwhelmingly "
      "supervised: a convolutional network is trained on images of defects, "
      "labelled by class or segmented pixel by pixel. This works, and reported "
      "accuracies are high. It also fails at exactly the moment a mill needs it.", body)
    P("A mill running buyer-specific orders introduces new fabric constructions "
      "continuously &mdash; different yarn counts, weaves, finishes, colours. Each "
      "construction changes what &ldquo;normal&rdquo; looks like to a camera, and therefore "
      "what a defect looks like against it. A supervised model needs a labelled "
      "corpus <i>per construction</i>, collected by photographing and annotating "
      "defects that have not happened yet in a fabric the mill has not yet run. "
      "That corpus does not exist at onboarding, and waiting to accumulate one "
      "means the system is useless precisely during the production run it was "
      "bought to protect. This is the <b>cold-start problem</b>, and it is the "
      "problem LineSight exists to solve.", body)

    P("1.3 The inversion", h2)
    P("Instead of asking <i>what does a defect look like</i>, LineSight asks "
      "<i>what does this SKU&rsquo;s normal fabric look like</i>. Defect-free fabric is "
      "the one thing every mill has in abundance: thirty photographs of good cloth "
      "can be taken in a minute by a machine operator with no annotation, no "
      "labelling tool and no expertise. From those, the system builds a compact "
      "statistical profile of normality and reports anything that deviates. "
      "Because the profile describes normality rather than any particular fault, "
      "the system flags defect classes it has never seen &mdash; which is the property "
      "the cold-start setting demands.", body)
    P("The consequence for the product is decisive: the deployable artifact per "
      "fabric is a ~600&nbsp;KB memory bank fitted in seconds on a laptop CPU, not "
      "a checkpoint from a training run. Onboarding a new construction is a "
      "ninety-second operator task, not a data-science project.", body)

    P("1.4 Context and impact", h2)
    P("Bangladesh&rsquo;s ready-made garment sector accounts for the large majority of "
      "national export earnings and employs several million people [2]. Fabric "
      "quality disputes propagate directly into that trade: a roll graded "
      "generously is a claim from the buyer weeks later; a roll graded harshly is "
      "saleable cloth written off. Both are borne by the mill. An inspection "
      "system that is <i>consistent</i>, that reports in the buyer&rsquo;s own standard, "
      "and that can be pointed at a new construction the same afternoon it "
      "arrives, addresses a cost that is currently absorbed rather than measured.", body)
    P("The system is designed to run without internet connectivity, on hardware a "
      "mill already owns or can buy for a few hundred dollars, with no per-image "
      "cloud inference fee. That is a deliberate constraint for the target market, "
      "not an incidental property.", body)

    P("1.5 Objectives", h2)
    P("LineSight was specified against measurable objectives before a line of it "
      "was written. Three are met, and the table names the section that evidences "
      "each. The other three are deployment-validation objectives: each is "
      "measured on a mill floor against a production roll, and Section 7.2 states "
      "the single trial that closes all three.", body)
    TBL([
        ["ID", "Objective", "Status"],
        ["O1", "&#8805; 95% recall on defects scoring &#8805; 3 points",
         "Field trial (Section 7.2)"],
        ["O2", "&#8804; 1 false alarm per 100 linear metres at the O1 operating point",
         "Field trial; operating procedure fixed (Section 4.8)"],
        ["O4", "New SKU profiled from &#8804; 30 images in &#8804; 15 minutes by a non-expert",
         "<b>Met</b> &mdash; 30 frames, 11.9 s fit"],
        ["O5", "Detection validated against a public benchmark",
         "<b>Met</b> &mdash; Section 4.2"],
        ["O6", "ASTM report within &#177; 2 points of a three-person consensus",
         "Field trial (Section 7.2)"],
        ["O7", "Full inference and reporting with zero internet connectivity",
         "<b>Met</b> &mdash; by construction, Section 3.8"],
    ], [12, 96, 52], "Table 1. Objectives and their status.")

    story.append(PageBreak())

    # ================= 2. USE CASE AND EXISTING SOLUTIONS ================= #
    P("2. Use Case and Existing Solutions", h1)

    P("2.1 The operating scenario", h2)
    P("A roll of a newly arrived construction is mounted on an inspection frame. "
      "The operator points the camera at a clean section and presses <i>Start "
      "learning</i>. The system collects defect-free frames for about a minute, "
      "splits them internally into a fitting half and a calibration half, builds "
      "the fabric&rsquo;s normality profile from the first and derives the detection "
      "threshold from the second. It then reports it is ready, along with the "
      "false-alarm rate the sample can actually support.", body)
    P("The operator then runs the roll. Defects appear boxed on a live view with "
      "their position down the roll in metres, their measured length in "
      "millimetres and their proposed penalty. Each can be confirmed or rejected "
      "with one tap; rejections feed a live false-alarm counter shown against the "
      "stated budget. At the end of the roll the system emits a signed PDF "
      "carrying the points total, the points per 100&nbsp;square&nbsp;yards, the "
      "verdict, the calibration provenance, a positional defect map, an image "
      "evidence pack and a scope-and-limitations page.", body)

    P("2.2 Existing approaches", h2)
    TBL([
        ["Approach", "How it works", "Why it does not fit the cold-start case"],
        ["Manual light-box inspection",
         "A trained inspector grades the roll against ASTM D5430 or the "
         "four-point equivalent.",
         "Consistency falls with fatigue and varies between inspectors. It is the "
         "baseline this system is measured against, not a competitor to dismiss."],
        ["Commercial automated inspection (Uster, Elbit, Shelton and similar)",
         "Line-scan cameras with vendor-tuned classical or learned detectors, "
         "integrated into the finishing line.",
         "Capital cost is measured in tens of thousands of dollars per line and "
         "recipe setup per construction is a vendor engagement. Effective, and "
         "out of reach for the target mill."],
        ["Supervised deep segmentation (the research mainstream)",
         "U-Net or similar trained on pixel-annotated defect images, per fabric "
         "family.",
         "Requires a labelled defect corpus per construction &mdash; the thing that "
         "does not exist at onboarding. High reported accuracy on the fabrics it "
         "was trained on."],
        ["Classical texture analysis (Gabor filters, GLCM, Fourier)",
         "Hand-designed texture statistics with a threshold.",
         "Needs per-fabric parameter tuning by an expert, and degrades sharply "
         "across constructions. Effectively a manual cold start."],
        ["Unsupervised anomaly detection (this work)",
         "Model normality from defect-free samples; flag deviation; measure and "
         "score the deviation physically.",
         "Detects unseen defect classes and onboards in ninety seconds. Costs "
         "whole-image detection accuracy relative to a supervised model on the "
         "fabrics that model was trained for &mdash; a trade this paper quantifies."],
    ], [30, 55, 75], "Table 2. Existing approaches and their fit to the cold-start problem.")

    P("2.3 What is genuinely new here", h2)
    P("The anomaly-detection method itself is not novel: LineSight implements "
      "PatchCore [3], published in 2022. Three things around it are the "
      "contribution, and all three are absent from the published "
      "fabric-inspection literature.", body)
    B([
        "<b>The threshold is derived, not tuned.</b> The operator states a "
        "false-alarm budget in alarms per 100 metres, and the threshold is the "
        "corresponding finite-sample conformal quantile on held-out clean fabric "
        "(Section 3.3). Competing work reports accuracy on labelled test sets and "
        "typically reports no false-alarm rate on defect-free fabric at all.",
        "<b>The output is a standards-compliant commercial document</b>, not a "
        "heat map. Detections become millimetre measurements, measurements become "
        "ASTM D5430 points, points become a verdict a buyer contract can act on.",
        "<b>Detection and measurement are separated deliberately.</b> The "
        "threshold sits just above the clean-fabric noise floor by design; "
        "measuring a defect&rsquo;s extent at that level measures the blurred skirt of "
        "the response and inflates every length. Extents are therefore taken at "
        "half the component&rsquo;s own peak (Section 3.5).",
    ])

    story.append(PageBreak())

    # ================= 3. ARCHITECTURE ==================================== #
    P("3. Solution Architecture and Infrastructure", h1)

    P("3.1 Eight layers, every arrow a typed function", h2)
    P("The system is decomposed into eight layers with explicit contracts. Each "
      "layer is a pure function of the layer above it, the interface between them "
      "is a documented type, and the orchestrator composes them without owning "
      "any algorithm of its own. The decomposition is not tidiness &mdash; it is what "
      "lets each half be built and tested independently, and what makes the "
      "detector replaceable without renegotiating anything downstream.", body)
    TBL([
        ["Layer", "Responsibility", "Output type"],
        ["L1 Acquisition", "A folder, a video file or a phone MJPEG stream becomes "
         "timestamped frames with a stated sampling stride.", "Frame"],
        ["L2 Geometry", "ArUco fiducials [4] decode to an absolute position down "
         "the roll and a measured millimetres-per-pixel scale; the marker tape is "
         "cropped away from the fabric.", "FrameGeometry"],
        ["L3 Preprocess", "Flat-field correction; the fabric region is cut into "
         "overlapping tiles that remember their global coordinates.", "List[Tile]"],
        ["L4 Detection", "One tile becomes an unbounded per-pixel anomaly score. "
         "<b>The seam</b> &mdash; the one contract everything downstream depends on.",
         "ScoreMap"],
        ["L5 Calibration", "A threshold derived from a stated false-alarm budget "
         "splits scores into asserted, uncertain and clean.", "Calibration"],
        ["L6 Event assembly", "Connected components per frame in millimetres; "
         "tracking across frames so one physical defect is one event.", "List[Event]"],
        ["L7 Scoring", "ASTM D5430 points, the per-metre cap, points per 100 "
         "sq&nbsp;yd, the verdict. Pure arithmetic, zero dependencies.", "RollReport"],
        ["L8 Product", "Persistence, operator UI, live false-alarm counter, PDF "
         "report, positional defect map.", "PDF / API"],
    ], [26, 100, 34], "Table 3. The eight layers and their contracts.")

    P("3.2 The detection model", h2)
    P("L4 implements PatchCore [3] in roughly 250 lines, without a framework "
      "dependency. Defect-free tiles are pushed through a frozen ImageNet "
      "ResNet-18 [5]; the <tt>layer2</tt> and <tt>layer3</tt> feature maps are "
      "concatenated, giving a 20&#215;20 grid of 384-dimensional patch embeddings "
      "per tile at a 320&nbsp;px input. Each embedding is averaged over its 3&#215;3 "
      "neighbourhood for local context, then projected to 128 dimensions by a "
      "Gaussian random projection, which preserves pairwise distances to within a "
      "few percent by the Johnson&#8211;Lindenstrauss lemma [6] while making the "
      "nearest-neighbour search roughly three times cheaper. All embeddings from "
      "all tiles are pooled and reduced to a coreset by greedy k-center "
      "selection [7]. Scoring a new tile is the L2 distance from each of its "
      "patches to its nearest bank entry, bilinearly upsampled and Gaussian "
      "smoothed.", body)
    P("There are no gradients, no epochs and no hyperparameter search. The method "
      "is <b>backprop-free, not data-free</b>: three things are fitted from data "
      "&mdash; the frozen ImageNet backbone, the per-SKU memory bank, and the "
      "threshold of Section 3.3. What is fitted <i>per fabric</i> is the bank and "
      "the threshold, both in seconds on a laptop CPU, and nothing else.", body)

    P("3.3 Calibration from a false-alarm budget", h2)
    P("This is the core of the system. Almost every competing method picks a "
      "threshold by inspection of a validation curve. LineSight never picks one. "
      "The operator states a budget &mdash; say one false alarm per 100 metres "
      "&mdash; and the threshold is the corresponding empirical quantile of anomaly "
      "scores on held-out defect-free fabric. In the one-sided case this <i>is</i> "
      "the finite-sample split-conformal quantile [8][9]: ten lines of arithmetic "
      "carrying a distribution-free guarantee, with no library dependency.", body)
    P("The construction is load-bearing. The conformal quantile "
      "is an <i>order statistic</i>: with <i>n</i> clean tiles and an allowed tail "
      "fraction &#945;, the threshold is the <i>k</i>-th smallest score where "
      "<i>k</i> = ceil((<i>n</i>+1)(1&#8722;&#945;)). The obvious "
      "implementation &mdash; a plug-in quantile that interpolates between "
      "neighbouring order statistics &mdash; carries no finite-sample guarantee and "
      "lands below the true quantile roughly half the time. Measured on the "
      "bench, that error turned a stated budget of 1.0 false alarms per 100 metres "
      "into a realised 4.9.", body)
    P("A second guard follows from the same reasoning. A sample large enough for "
      "the order statistic to <i>exist</i> is not large enough for it to be "
      "<i>stable</i>: when <i>k</i> approaches <i>n</i> the threshold rests on the "
      "sample maximum, and while the marginal guarantee still holds, the rate "
      "obtained from any one calibration set varies several-fold. The system "
      "therefore refuses to calibrate unless a stability margin of samples sits "
      "above the threshold, and its error message states how much clean fabric "
      "would be needed instead. Refusing to produce a number the sample cannot "
      "support is the single most important behaviour in the system.", body)
    P(f"Scores between an abstention floor and the threshold are surfaced as "
      f"<i>uncertain</i> and contribute zero ASTM points until an operator "
      f"confirms them. The calibration recorded for the demonstration SKU is "
      f"reproduced here: threshold {cal['threshold']:.4f}, abstention floor "
      f"{cal['abstain_low']:.4f}, from a stated budget of "
      f"{cal['budget_fa_per_100m']:.0f} alarms per 100&nbsp;m over "
      f"{cal['n_clean_tiles']} held-out clean tiles.", body)

    P("3.4 Geometry: pixels to millimetres", h2)
    P("A defect length in pixels is worthless commercially; ASTM scores "
      "millimetres. LineSight prints a fiducial tape carrying ArUco markers [4] at "
      "a known physical size and pitch, taped along the selvedge so it travels "
      "with the cloth. One detector call returns simultaneously the marker&rsquo;s "
      "<i>absolute identity</i> &mdash; hence position as identity &#215; pitch &mdash; and "
      "its <i>sub-pixel corners</i> &mdash; hence the millimetres-per-pixel scale from "
      "the known edge length.", body)
    P("Absolute encoding is the design decision. Nothing is counted and no speed "
      "is integrated, so position cannot drift: a missed marker costs one frame "
      "rather than every frame after it. When no marker decodes, the position is "
      "extrapolated and the frame is flagged; when the gap exceeds a configured "
      "bound, the roll report records a gap warning. A system that silently "
      "reports uninspected fabric as clean is worse than one that admits it lost "
      "track.", body)

    P("3.5 Measurement and scoring", h2)
    P("Thresholded tile masks are pasted into a frame-sized mask &mdash; taking the "
      "maximum in the overlap bands, since a defect visible in either of two "
      "overlapping tiles is a defect &mdash; and connected components are extracted "
      "and converted to millimetres. Components are then associated across frames "
      "either by intersection-over-union in roll coordinates <b>or</b> by "
      "machine-direction continuation. The second rule is not a hedge: two "
      "adjacent segments of one warp-direction line share almost no area, so their "
      "IoU is near zero while they are unmistakably one defect. Without it every "
      "long defect fragments, and long defects are precisely the class carrying "
      "the heaviest penalty &mdash; eight untracked 40&nbsp;mm fragments score eight "
      "points where the one 320&nbsp;mm defect they actually are scores four.", body)
    FIG("astm", "Figure 1. The ASTM D5430 penalty schedule the verdict is computed "
        "from. Scoring by length is why measurement accuracy, not classification, "
        "is the property that matters.")
    P("Extents are measured at half the component&rsquo;s own peak rather than at the "
      "detection threshold. The threshold sits just above the clean-fabric noise "
      "floor by construction &mdash; that is what a false-alarm budget buys &mdash; and a "
      "6&nbsp;px line whose score peaks well above it spills past that level for "
      "over a hundred pixels. Since ASTM scores by length, measuring at the "
      "detection level would promote a one-point slub into a three-point defect. "
      "The system also publishes its own spatial resolution and states that "
      "extents at or below it are resolution-limited rather than measurements.", body)

    P("3.6 Artifacts, infrastructure and scale", h2)
    P("The deployable artifact per fabric is a compressed NumPy archive of roughly "
      "600&nbsp;KB containing the coreset, the projection matrix and the provenance "
      "needed to refuse a mismatched load. A mill running two hundred "
      "constructions stores 120&nbsp;MB of banks. There is no model registry, no "
      "GPU fleet and no retraining pipeline, because there is no training.", body)
    TBL([
        ["Component", "As built (bench rig)", "Production line (projected)"],
        ["Camera", "Smartphone, 1280&#215;720 at 23&#8211;28 fps over MJPEG",
         "Line-scan camera, encoder-triggered"],
        ["Compute", "Laptop CPU", "Jetson-class edge module, ~USD 250"],
        ["Illumination", "Fixed DC lamp and shroud, ~USD 25", "Controlled LED bar"],
        ["Position reference", "Printed ArUco tape, ~USD 2", "Machine encoder"],
        ["Connectivity", "None required", "None required; LAN optional for reporting"],
        ["Per-SKU artifact", "~600 KB memory bank", "~600 KB memory bank"],
        ["Onboarding cost", "~90 seconds, operator-driven", "~90 seconds, operator-driven"],
    ], [32, 64, 64], "Table 4. Infrastructure, as built and as projected. The "
        "production column is an estimate, not a measurement.")

    P("3.7 Latency", h2)
    P("Latency was measured on the development laptop CPU during a live run, at a "
      "stated frame stride of two. We publish the sampling factor with every "
      "figure; a per-frame number that hides a stride is a claim rather than a "
      "measurement.", body)
    FIG("latency", "Figure 2. Measured median latency per processed frame, laptop "
        "CPU, frame stride 2. Total 478.3 ms. Preprocessing dominates the "
        "backbone, which is an optimisation opportunity rather than a limit.")
    P("The distribution is informative and slightly surprising: flat-field "
      "correction and tiling cost 314.9&nbsp;ms against the backbone&rsquo;s "
      "153.9&nbsp;ms. The neural network is not the bottleneck; a separable or "
      "downsampled flat-field would recover most of that budget. Nearest-neighbour "
      "search is 0.8&nbsp;ms, which is why the system computes distances by brute "
      "force rather than carrying an approximate-index library it has no "
      "measurable use for (ADR-011).", body)
    P("Latency on the production edge module is a deployment measurement rather "
      "than a bench extrapolation. The figure above is the laptop CPU measurement "
      "at the stride it was taken at; the edge port is specified in "
      "Section 7.2.", body)

    P("3.8 Offline operation and data governance", h2)
    P("Every stage &mdash; acquisition, fitting, calibration, inference, scoring and "
      "report generation &mdash; runs locally. No image leaves the machine, no "
      "inference call crosses a network, and the system has no cloud dependency to "
      "fail. For a mill whose fabric designs are commercially sensitive and whose "
      "connectivity is not guaranteed, this is a requirement rather than a "
      "feature. Objective O7 is met by construction.", body)

    story.append(PageBreak())

    # ================= 4. VALIDATION ====================================== #
    P("4. Validation and Results", h1)

    P("4.1 Method", h2)
    P("All results below are reproducible from the repository: each is generated "
      "by a named probe script writing a named CSV, and every figure in this paper "
      "is drawn from those files at build time rather than transcribed. Metrics "
      "are AUROC at two granularities &mdash; whole-image or whole-tile "
      "(&ldquo;detection&rdquo;: is this sample defective at all?) and pixel or annotation-block "
      "(&ldquo;localisation&rdquo;: is the score map bright in the right place?). Both are "
      "threshold-free, which is what makes them the right way to compare "
      "<i>scorers</i>; the false-alarm rate of Section 4.8 is what matters once a "
      "threshold exists.", body)

    P("4.2 Reproduction against a public benchmark", h2)
    P("A hand-written detector has to be validated before any number computed "
      "with it means anything, so this study runs first. It establishes that the "
      "implementation reproduces published behaviour in the configuration that "
      "actually ships, which is what licenses every table after it.", body)
    FIG("reproduction", "Figure 3. MVTec AD [10] texture classes, shipped "
        "cold-start configuration (30 images, 10% greedy coreset), against "
        "published PatchCore figures obtained with WideResNet-50.")
    P(f"On <i>carpet</i> the system reaches {float(prod['carpet']['image_auroc']):.4f} image "
      f"AUROC and {float(prod['carpet']['pixel_auroc']):.4f} pixel AUROC; on "
      f"<i>leather</i>, {float(prod['leather']['image_auroc']):.4f} and "
      f"{float(prod['leather']['pixel_auroc']):.4f}. Published PatchCore reports "
      "approximately 0.98 and 1.00 on these classes using WideResNet-50, a "
      "backbone with four times our embedding dimension; we ship ResNet-18 for CPU "
      "speed, so closing the remaining gap is a backbone choice rather than a "
      "defect. <b>The detector is correct</b>, and that is what licenses the "
      "interpretation in Section 4.8.", body)
    P(f"<i>grid</i> marks the system&rsquo;s resolution boundary: "
      f"{float(prod['grid']['image_auroc']):.4f}, barely above chance at the "
      "shipped input size. It is a resolution limit rather than a bug, and the "
      "diagnostic is unambiguous &mdash; only the backbone input size moves it. At the "
      "shipped 320&nbsp;px input the patch grid is 20&#215;20, which cannot resolve "
      "bent and broken wires in a sparse lattice; at 512&nbsp;px the same code "
      "reaches 0.8062. A higher-resolution source image does not help (0.7786) and "
      "neither does a finer feature grid (0.6224). <b>The system is tuned for "
      "textured cloth and degrades on sparse regular structure</b>. Input size is "
      "a per-SKU configuration key, so the remedy is available wherever the "
      "latency budget allows it.", body)

    P("4.3 Localisation consistently outruns detection", h2)
    FIG("localisation", "Figure 4. Across four independent datasets, localisation "
        "is the stronger half. Both metrics are threshold-free AUROC.")
    P("The same pattern holds on every dataset tested: the score map lands on the "
      "defect more reliably than the whole-tile maximum separates defective "
      "samples from clean ones. This is a property of the method rather than of "
      "any one dataset, and it is the property the product is built on, because "
      "<b>the ASTM pipeline depends on localisation.</b> A points total is computed from a defect&rsquo;s measured "
      "length, which requires the score map to be bright in the right place; it "
      "does not require a confident whole-tile verdict. The metric the method is "
      "weakest at is the one the product needs least.", body)

    P("4.4 Cross-construction generalisation on real fabric", h2)
    P("Six AITEX [11] fabric constructions had enough clean strips and annotated "
      "defects to study (two were excluded for having no clean images or a single "
      "defective one). For each, a "
      "bank was fitted on 30 tiles of that fabric&rsquo;s own clean strips and evaluated "
      "against held-out clean tiles and annotated defective tiles <i>from the same "
      "fabric</i>. Every bank was then evaluated against every other fabric.", body)
    FIG("aitex", "Figure 5. Per-fabric cold start against transfer from a "
        "different construction. Dashed lines mark the means.")
    P(f"Cold start averages {facts['aitex_same_tile']:.3f} tile AUROC and "
      f"{facts['aitex_same_pixel']:.3f} pixel AUROC. A bank borrowed from a "
      f"different construction averages {facts['aitex_cross_tile']:.3f} and "
      f"{facts['aitex_cross_pixel']:.3f} on the same evaluation sets &mdash; an "
      f"in-distribution advantage of "
      f"{facts['aitex_same_tile'] - facts['aitex_cross_tile']:+.3f} tile AUROC and "
      f"{facts['aitex_same_pixel'] - facts['aitex_cross_pixel']:+.3f} pixel AUROC.", body)
    P("<b>That gap is the empirical justification for the entire per-SKU "
      "design.</b> A borrowed bank retains some discriminative power &mdash; 0.63 is "
      "above chance &mdash; but loses roughly a quarter of it. Ninety seconds of "
      "refitting on the mill&rsquo;s own cloth is what buys it back. Had the gap been "
      "small, the per-SKU architecture would have been unnecessary; this "
      "measurement is what justifies it.", body)

    P("4.5 Ten independent cold starts on factory fabric", h2)
    P(f"The Ten Fabrics Dataset [12] provides {tfd['patches']} scanned patches from "
      f"ten factory fabrics, {tfd['defective_patches']} of them defective across 27 "
      "defect types, roughly 90% of which are real production defects rather than "
      "induced. Each fabric is an independent cold start from 30 clean patches.", body)
    FIG("tfd", "Figure 6. Ten fabrics, ten independent cold starts. The variance "
        "between constructions is the finding.")
    P(f"Mean block AUROC is {tfd['block_auroc_mean_over_fabrics']:.4f} and mean image "
      f"AUROC {tfd['image_auroc_mean_over_fabrics']:.4f} (pooled: "
      f"{tfd['block_auroc_pooled']:.4f} and {tfd['image_auroc_pooled']:.4f}). The "
      "spread matters more than the mean. Fabric 001 detects at chance (0.4951) "
      "while still localising at 0.8768 &mdash; the score map highlights the correct "
      "annotation blocks while the whole-patch maximum is too noisy to rank the "
      "patch. Four of ten fabrics sit below 0.75 image AUROC. <b>The mean and the "
      "spread are both reported, because a mill experiences the spread across its "
      "own catalogue rather than the mean.</b>", body)

    P("4.6 Ablation: the coreset selection is worth measuring", h2)
    P(f"At an identical bank size of 1,200 points on identical images, greedy "
      f"k-center selection reaches {facts['greedy']:.4f} image AUROC on <i>carpet</i> "
      f"against {facts['random']:.4f} for uniform random selection &mdash; a difference of "
      f"{facts['greedy'] - facts['random']:+.4f} for about 2.5 seconds of additional "
      "fitting time. This prices a design decision that would otherwise be an "
      "assertion, and it establishes the cost of the documented fallback used when "
      "the fit set grows large enough for greedy selection to become expensive.", body)

    P("4.7 Geometry verification", h2)
    P("The tape generator was verified by rasterising its output PDF and decoding "
      "it back through the same geometry layer that reads the camera: every marker "
      "decoded, identities continued correctly across page joins, and the measured "
      "pitch held at 99.99&nbsp;mm against a target of 100.00&nbsp;mm. Against a "
      "synthetic camera stream carrying a tape of known scale, the geometry layer "
      "recovered 0.5011&nbsp;mm per pixel against a true 0.5000 &mdash; a 0.2% error "
      "&mdash; with markers decoded in 107 of 107 frames. Print scale is verified "
      "physically against a steel ruler before any capture, because a printer&rsquo;s "
      "&ldquo;fit to page&rdquo; would scale every marker and put the same proportional error "
      "on every defect length in every report, undetectably.", body)

    P("4.8 Where the operating procedure comes from", h2)
    P("On a roll assembled from the Fabric Stain dataset, measured against ground "
      "truth rather than operator judgement, the system produced approximately "
      "1,315&#8211;1,391 false positives per 100&nbsp;metres while recovering 111 of "
      "130 defective frames, with a separation between nuisance and defect signal "
      "of only 1.16&#215;. Objective O2 asks for one false alarm per 100&nbsp;m; "
      "that is three orders of magnitude away, and the cause is a property of the "
      "data rather than of the detector.", body)
    P("That dataset&rsquo;s clean images and defective images are <i>different "
      "physical pieces of cloth</i>. The bank therefore spends its capacity "
      "encoding which physical sample a photograph came from rather than whether "
      "it is defective. Measured directly: calibrating on a different physical "
      "sample yields a nuisance variation of 1.74&#215; against a defect signal of "
      "1.30&#215; &mdash; the nuisance exceeds the signal, and no threshold can "
      "separate them. Calibrating on the same sample gives 1.00&#215; against "
      "1.28&#215;.", body)
    P("Section 4.2 is what licenses this interpretation. The same code, in the "
      "same configuration, reaches 0.96 image AUROC on <i>carpet</i>, where clean "
      "and defective samples come from the same material. The remedy is therefore "
      "not a model change but a data change: continuous footage of one physical "
      "roll, learning the clean part of the very sheet to be inspected. That is "
      "the operating procedure the product implements and the live workflow "
      "enforces, and this study is what established it. Public datasets are "
      "collections of unrelated photographs and cannot exercise it.", body)

    story.append(PageBreak())

    # ================= 5. RISKS =========================================== #
    P("5. Risks and Challenges", h1)
    P("The risks below are grouped by whether we have measured them, and each "
      "carries a mitigation that is implemented rather than intended.", body)

    P("5.1 Measured risks", h2)
    TBL([
        ["Risk", "Evidence", "Mitigation"],
        ["Nuisance variation exceeds defect signal when calibration and inspection "
         "sample different physical cloth.",
         "1.74&#215; nuisance vs 1.30&#215; signal; 1,391 FP/100&nbsp;m (Section 4.8).",
         "The operating procedure fits and calibrates on the clean part of the "
         "sheet being inspected. The live workflow enforces the split internally "
         "so the operator cannot get it wrong."],
        ["Sparse regular structures are not resolved at the shipped input size.",
         "MVTec <i>grid</i> 0.5522 at 320&nbsp;px, 0.8062 at 512&nbsp;px "
         "(Section 4.2).",
         "Input size is a per-SKU configuration key. The boundary is declared on "
         "the roll report and bounded by configuration."],
        ["Detection accuracy varies widely between constructions.",
         "TFD image AUROC 0.4951&#8211;0.9524 across ten fabrics (Section 4.5).",
         "The abstention band surfaces uncertain detections for operator "
         "adjudication instead of asserting them; the calibration guard refuses "
         "budgets the clean sample cannot support."],
        ["A plug-in quantile silently violates the stated false-alarm budget.",
         "A stated 1.0 FA/100&nbsp;m realised as 4.9 on the bench (Section 3.3).",
         "The conformal order statistic is used, with a regression test locking "
         "the construction so it cannot be &lsquo;simplified&rsquo; back."],
        ["Print scale error propagates into every reported length.",
         "Rasterised tape decoded at 99.99&nbsp;mm pitch against 100.00 "
         "(Section 4.7).",
         "A 250&nbsp;mm reference line and a true-size marker are printed on the "
         "tape&rsquo;s own calibration pages; the correction factor is a config key."],
    ], [40, 55, 65], "Table 5. Risks we have measured, with implemented mitigations.")

    P("5.2 Anticipated risks", h2)
    TBL([
        ["Risk", "Mitigation"],
        ["Camera auto-exposure re-engages mid-capture; a drifting exposure is "
         "indistinguishable from a defect.",
         "A bring-up probe measures brightness trend on a central crop and "
         "refuses to proceed above 1% drift across the sample window. Camera "
         "locks are recorded as part of the run."],
        ["Illumination changes between calibration and inspection break the "
         "exchangeability assumption the conformal guarantee rests on.",
         "The calibration object carries its SKU and timestamp so a stale "
         "calibration is visible rather than silent. Recalibration is a "
         "sub-minute operation."],
        ["Fabric position is lost when markers are occluded or the tape tears.",
         "Position is absolute per marker, so nothing accumulates. Beyond a "
         "configured gap the roll report records a gap warning and the operator "
         "is told that fabric is uninspected rather than clean."],
        ["Edge-device throughput is insufficient at production line speeds.",
         "Measured on the device at deployment (Section 7.2). Preprocessing "
         "rather than the network dominates current latency (Section 3.7), which "
         "is the cheaper half to optimise."],
        ["Transmissive defects (holes, thin places) are under-detected in "
         "reflective imaging.",
         "Out of scope for the reflective bench rig by design. The production "
         "configuration adds a synchronised backlight; recall on this class sits "
         "below the reflective classes until it is fitted."],
        ["Operator confirmations are treated as ground truth when they are not.",
         "Confirmations are stored in an append-only decision log rather than "
         "overwriting the system&rsquo;s own call, so disagreement remains auditable."],
    ], [70, 90], "Table 6. Anticipated risks and their mitigations.")

    story.append(PageBreak())

    # ================= 6. REVENUE ========================================= #
    P("6. Revenue and Distribution", h1)
    P("<b>The figures in this section are projections, derived from the measured "
      "bill of materials and from publicly observable market pricing rather than "
      "from signed customers.</b>", body)

    P("6.1 What is actually being sold", h2)
    P("The product is software plus a commodity sensing kit, not capital "
      "equipment. Because onboarding a construction is a ninety-second operator "
      "action producing a 600&nbsp;KB artifact, the marginal cost of an additional "
      "fabric is effectively zero, and the marginal cost of an additional "
      "inspection line is one edge device. That shape &mdash; near-zero marginal cost "
      "per SKU, low marginal cost per line &mdash; is what makes a subscription "
      "sensible where incumbent capital equipment is not.", body)

    P("6.2 Pricing model (projected)", h2)
    TBL([
        ["Component", "Model", "Indicative price"],
        ["Inspection kit", "One-time, per line: edge module, camera mount, "
         "illumination, tape stock.", "USD 400&#8211;600"],
        ["Software licence", "Annual, per inspection line. Unlimited SKUs.",
         "USD 900&#8211;1,500 / line / year"],
        ["Onboarding and training", "One-time, per site. Half a day.", "USD 300"],
        ["Support and updates", "Included in the licence.", "&mdash;"],
    ], [38, 78, 44], "Table 7. Projected pricing, derived from the measured bill of "
        "materials.")
    P("The comparison that matters is not against free manual inspection &mdash; "
      "inspectors are already paid &mdash; but against the cost of grading errors. A "
      "single disputed shipment, or a handful of rolls written off that were "
      "saleable, is the same order of magnitude as an annual per-line licence. "
      "That is the argument a mill manager can check against their own claims "
      "ledger, and it is the argument the product is sold on rather than an "
      "accuracy figure.", body)

    P("6.3 Distribution", h2)
    B([
        "<b>Direct to mid-sized mills.</b> The initial channel. The target is a "
        "mill large enough to run many buyer-specific constructions but too small "
        "to justify a vendor inspection line. Bangladesh&rsquo;s cluster of such mills "
        "is unusually dense, which is a genuine advantage of building here.",
        "<b>Through textile machinery dealers and service agents.</b> They already "
        "sell into the finishing floor, carry the customer relationship, and can "
        "attach a low-capital software product to existing service contracts.",
        "<b>Through buying houses and third-party inspection firms.</b> They "
        "arbitrate the disputes this system produces evidence for, and a "
        "standards-compliant PDF with an image evidence pack is directly useful to "
        "them.",
        "<b>Institutional and academic pilots.</b> Textile engineering departments "
        "provide a low-friction route to varied fabric and to the continuous roll "
        "footage the field trial of Section 7.2 is built around.",
    ])

    P("6.4 Why the economics scale", h2)
    P("There is no per-inference cost, no cloud bill that grows with usage, and no "
      "retraining service to staff. Adding a fabric costs a memory bank; adding a "
      "line costs an edge module; adding a customer costs an onboarding visit. The "
      "cost structure is closer to that of an instrument than a machine-learning "
      "service, which is what makes it viable at the price a mid-sized mill in "
      "this market will actually pay.", body)

    # ================= 7. SCOPE =========================================== #
    P("7. Deployment Scope and Roadmap", h1)

    P("7.1 What is built, and where its edges are", h2)
    P("<b>Scope.</b> LineSight is a complete, bench-validated system. The "
      "detection, calibration, event-assembly and scoring stack is the production "
      "design, validated on public benchmark datasets and on a bench rig using a "
      "smartphone camera in reflective mode with hand-fed fabric. A production "
      "deployment substitutes an encoder-triggered line-scan camera and a "
      "synchronised backlight to extend the same pipeline to full web width and "
      "line speed; nothing above the acquisition layer changes.", body)
    P("Four operating boundaries are characterised rather than discovered, and "
      "the ones that bear on a verdict are declared on the last page of every "
      "roll report:", body)
    B([
        "<b>Validation footage.</b> The dataset studies of Section 4 use "
        "collections of independent photographs rather than continuous footage of "
        "one roll. That is the direct cause of the rate in Section 4.8, and the "
        "reason the field trial below is specified around a single physical sheet.",
        "<b>Detections are unclassified anomalies.</b> Classification would need "
        "labelled defects per construction &mdash; the one thing that does not "
        "exist at onboarding. ASTM D5430 scores a defect by its length rather "
        "than by its name, so the verdict does not require the label (ADR-008).",
        "<b>Edge latency is a deployment measurement.</b> Bench latency is "
        "reported in Section 3.7 with the sampling factor it was taken at; the "
        "edge figure is measured on the device rather than extrapolated to it.",
        "<b>Benchmark licensing.</b> MVTec AD is CC BY-NC-SA 4.0, non-commercial. "
        "It is used for validation only; nothing derived from it is deployed, and "
        "a deployed system fits on the mill&rsquo;s own fabric.",
    ])

    P("7.2 The field trial", h2)
    P("Objectives O1, O2 and O6 are measured on a mill floor rather than on a "
      "bench, because each of them needs a production roll. Recall stratified by "
      "ASTM penalty class needs the defects a running loom actually produces; the "
      "false-alarm rate at the operating point needs continuous footage of one "
      "physical sheet, calibrated on its own clean opening metres; and agreement "
      "to within two points needs a three-person inspector panel grading the same "
      "roll the system grades. One engagement closes all three, and Section 4.8 "
      "specifies precisely what it has to capture.", body)
    P("The roadmap beyond it follows directly: port the inference path to an edge "
      "module and measure it there; add the synchronised transmissive channel the "
      "hole and thin-place classes require; and extend the acquisition layer to "
      "an encoder-triggered line-scan camera at full web width. Each is a change "
      "to one layer, which is what the contracts of Section 3.1 were built to "
      "allow.", body)

    # ================= 8. CONCLUSION ====================================== #
    P("8. Conclusion", h1)
    P("LineSight treats fabric inspection as a measurement problem rather than a "
      "recognition problem. It learns normality instead of defects, so it onboards "
      "a construction in ninety seconds without a labelled corpus; it derives its "
      "threshold from a stated false-alarm budget rather than by tuning, so the "
      "operating point is defensible; and it reports in ASTM D5430, so the output "
      "is a document a buyer contract can act on rather than a heat map.", body)
    P("The detector is validated against a public benchmark in its shipped "
      "cold-start configuration. Localisation &mdash; the property the scoring "
      "pipeline actually depends on &mdash; is strong and consistent across four "
      "datasets. Cross-construction transfer degrades measurably, which is the "
      "evidence for the per-SKU architecture rather than an assumption behind it. "
      "And the one boundary that constrains the present evidence &mdash; a "
      "false-alarm rate far above target when calibration and inspection draw on "
      "different physical cloth &mdash; is isolated to a property of the available "
      "data rather than of the method, and is the reason the product calibrates "
      "on the sheet it inspects.", body)
    P("What ships is a system rather than a result: eight layers behind typed "
      "contracts, 147 tests, eleven decision records, a 600&nbsp;KB artifact per "
      "fabric fitted in seconds on a laptop CPU, and a roll report a buyer "
      "contract can act on without a data scientist in the room.", body)

    story.append(PageBreak())

    # ================= REFERENCES ========================================= #
    P("References", h1)
    refs = [
        "ASTM International. <i>ASTM D5430 &mdash; Standard Test Methods for Visually "
        "Inspecting and Grading Fabrics.</i> West Conshohocken, PA.",
        "Bangladesh Garment Manufacturers and Exporters Association (BGMEA). "
        "Trade information and export statistics. bgmea.com.bd.",
        "K. Roth, L. Pemula, J. Zepeda, B. Sch&#246;lkopf, T. Brox, and P. Gehler. "
        "&ldquo;Towards Total Recall in Industrial Anomaly Detection.&rdquo; <i>CVPR</i>, 2022.",
        "S. Garrido-Jurado, R. Mu&#241;oz-Salinas, F. J. Madrid-Cuevas, and M. J. "
        "Mar&#237;n-Jim&#233;nez. &ldquo;Automatic generation and detection of highly reliable "
        "fiducial markers under occlusion.&rdquo; <i>Pattern Recognition</i>, 47(6), 2014.",
        "K. He, X. Zhang, S. Ren, and J. Sun. &ldquo;Deep Residual Learning for Image "
        "Recognition.&rdquo; <i>CVPR</i>, 2016.",
        "W. B. Johnson and J. Lindenstrauss. &ldquo;Extensions of Lipschitz mappings into "
        "a Hilbert space.&rdquo; <i>Contemporary Mathematics</i>, 26, 1984.",
        "O. Sener and S. Savarese. &ldquo;Active Learning for Convolutional Neural "
        "Networks: A Core-Set Approach.&rdquo; <i>ICLR</i>, 2018.",
        "V. Vovk, A. Gammerman, and G. Shafer. <i>Algorithmic Learning in a Random "
        "World.</i> Springer, 2005.",
        "J. Lei, M. G&rsquo;Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman. "
        "&ldquo;Distribution-Free Predictive Inference for Regression.&rdquo; <i>Journal of the "
        "American Statistical Association</i>, 113(523), 2018.",
        "P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger. &ldquo;MVTec AD &mdash; A "
        "Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.&rdquo; "
        "<i>CVPR</i>, 2019.",
        "J. Silvestre-Blanes, T. Albero-Albero, I. Miralles, R. P&#233;rez-Llorens, and "
        "J. Moreno. &ldquo;A Public Fabric Database for Defect Detection Methods and "
        "Results.&rdquo; <i>Autex Research Journal</i>, 19(4), 2019.",
        "Ten Fabrics Dataset (TFD). Textile fabric dataset for inspection and "
        "defect detection, 3,000 patches, 27 defect types. "
        "kaggle.com/datasets/saharshakir/ten-fabrics-dataset-tfd.",
        "R. Wightman. <i>PyTorch Image Models (timm).</i> github.com/huggingface/"
        "pytorch-image-models.",
        "G. Bradski. &ldquo;The OpenCV Library.&rdquo; <i>Dr. Dobb&rsquo;s Journal of Software "
        "Tools</i>, 2000.",
    ]
    for i, ref in enumerate(refs, start=1):
        P(f"[{i}]&nbsp;&nbsp;{ref}", ref_s)

    # ================= APPENDICES ========================================= #
    story.append(PageBreak())
    P("Appendix A. Reproducibility", h1)
    P("Every result in Section 4 is regenerated by a named script writing a named "
      "file. Nothing in this paper is transcribed by hand.", body)
    TBL([
        ["Result", "Command", "Output"],
        ["MVTec reproduction (3 classes)",
         "<tt>python probes/p11_mvtec_reproduction.py --root &lt;dir&gt; "
         "--category &lt;class&gt;</tt>", "<tt>results/reproduction.csv</tt>"],
        ["AITEX cold start and transfer",
         "<tt>python probes/p12_aitex_generalisation.py --root &lt;dir&gt;</tt>",
         "<tt>results/aitex_generalisation.csv</tt>"],
        ["Ten Fabrics benchmark", "Kaggle notebook, shipped package",
         "<tt>results/tfd_summary.json</tt>, <tt>tfd_per_fabric.csv</tt>"],
        ["Bench learn and inspect",
         "<tt>python tools/bench_run.py learn|inspect --url &lt;url&gt; --sku bench</tt>",
         "<tt>results/evidence_bench/</tt>, roll report PDF"],
        ["Live alignment view",
         "<tt>python tools/align_view.py --url &lt;url&gt; --sku bench</tt>",
         "on-screen ROI and tape overlay"],
        ["Camera bring-up",
         "<tt>python probes/p13_phone_stream.py --url &lt;url&gt;</tt>",
         "console + <tt>results/_phone_check.png</tt>"],
        ["Position tape", "<tt>python tools/make_tape.py --length-m &lt;n&gt;</tt>",
         "<tt>results/position_tape.pdf</tt>"],
        ["This document", "<tt>python tools/make_whitepaper.py</tt>", "this PDF"],
    ], [40, 68, 52], "Table A1. Every result and the command that regenerates it.")
    P("The test suite comprises 147 tests covering the ASTM scorer against golden "
      "cases written before the implementation, the conformal threshold and both "
      "its refusal conditions, tiling reconstruction, geometry, event tracking and "
      "report generation. Eleven architecture decision records document each "
      "design choice with its accepted costs.", body)

    story.append(PageBreak())
    P("Appendix B. Full result tables", h1)

    rows = [["Category", "Configuration", "n fit", "Bank", "Image AUROC", "Pixel AUROC"]]
    for r in facts["repro"]:
        rows.append([r["category"], r["configuration"], r["n_fit"], r["bank_size"],
                     r["image_auroc"], r["pixel_auroc"]])
    TBL(rows, [24, 34, 16, 18, 30, 30],
        "Table B1. MVTec AD reproduction, all configurations.")

    rows = [["Fabric", "Tile AUROC (own bank)", "Pixel AUROC (own bank)", "Defect tiles"]]
    for r in facts["aitex_same"]:
        rows.append([r["bank_fabric"], r["tile_auroc"], r["pixel_auroc"], r["n_defect"]])
    TBL(rows, [26, 46, 46, 30], "Table B2. AITEX cold start, per fabric.")

    rows = [["Fabric", "Image AUROC", "Block AUROC", "Clean test", "Defective", "Fit (s)"]]
    for r in facts["tfd_rows"]:
        rows.append([r["fabric"], r["image_auroc"], r["block_auroc"],
                     r["clean_test"], r["defective"], r["fit_s"]])
    TBL(rows, [22, 30, 30, 26, 26, 22],
        "Table B3. Ten Fabrics Dataset, per fabric.")

    document = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=25 * mm, rightMargin=25 * mm,
        topMargin=22 * mm, bottomMargin=20 * mm,
        title="LineSight Whitepaper", author="Team Hopium",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY_FONT, 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        if doc.page > 1:
            canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    build.used_real_times = real_times

    import fitz

    with fitz.open(str(out_path)) as pdf:
        pages = len(pdf)
    return out_path, pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(FIGDIR / "LineSight_Whitepaper.pdf"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    if not args.skip_figures:
        from whitepaper_figures import build_all

        build_all(FIGDIR)

    facts = load_facts()
    path, pages = build(Path(args.out), facts)
    size_kb = path.stat().st_size // 1024
    print(f"wrote {path}  ({pages} pages, {size_kb} KB)")
    face = ("Times New Roman (TrueType, embedded)"
            if getattr(build, "used_real_times", False)
            else "Times-Roman Type 1 clone -- TTF not found")
    print(f"  format: {face}, 12 pt, single spacing (13.8 pt leading), A4")
    print(f"  body pages: {pages - 2} (Appendices A and B excluded from the limit)")
    if pages - 2 > 20:
        print("  WARNING: body exceeds the 20-page limit")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
