"""L8 PRODUCT - FastAPI service and the live operator page.

The page is the product surface: fabric moving past the lens with detections
boxed on it as they are found, the roll position read off the ArUco tape, the
running ASTM total, and a live false-alarm counter beside the budget the
operator stated. That last pairing is the point of the page - it shows the
realised false-alarm rate against the promised one while the roll is running,
so the calibration is a measurement the operator can watch rather than a claim
in a report.

``LiveInspector`` is the whole of the logic: it drives one stream through the
four phases below and holds the latest annotated frame; the routes at the
bottom of the file only read its state.

Endpoints:

    GET  /                      the operator page
    GET  /stream.mjpg           annotated frames, as MJPEG
    GET  /api/stats             live counters (position, points, FA rate, latency)
    GET  /api/events            events found so far
    POST /api/events/{id}/confirm
    POST /api/events/{id}/reject    <- counts as a false alarm
    GET  /api/rolls                 stored rolls
    GET  /api/rolls/{id}

Inspection runs on a background thread while the page polls ``/api/stats`` and
pulls frames from ``/stream.mjpg``. Polling rather than a WebSocket: the video
is already a long-lived HTTP stream, and a poll that misses a beat recovers by
itself on the next tick with no reconnection logic to maintain.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..calibrate.threshold import calibrate
from ..config import Config
from ..types import Assertion, EventStatus, Frame

__all__ = ["LiveInspector", "create_app", "run_server"]


class LiveInspector:
    """Runs the operator's procedure against one stream, with visible phases.

    LEARNING -> CALIBRATING -> READY -> INSPECTING. The phase is drawn onto the
    video as a banner and published in ``/api/stats``, for two reasons that are
    both about not misreading what you are watching:

      * Learning and inspecting look identical on a video feed. The banner is
        what distinguishes "this is clean fabric being learned" from "this is
        the cloth under test".
      * The run holds at READY until the operator presses Start, instead of
        inspecting the moment calibration finishes. Otherwise fabric that
        passes the lens while the bank is still fitting is never inspected -
        and it would be reported as clean roll, not as skipped roll.
    """

    LEARNING = "learning"
    CALIBRATING = "calibrating"
    READY = "ready"
    INSPECTING = "inspecting"

    def __init__(
        self,
        config: Config,
        pipeline: Any,
        n_fit_frames: int = 30,
        n_calibration_frames: int = 120,
        camera_control: str | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.roll_id = f"{config.sku}-{int(time.time())}"
        self.n_fit_frames = n_fit_frames
        self.n_calibration_frames = n_calibration_frames
        self.camera_control = camera_control

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._go = threading.Event()

        self._phase = self.LEARNING
        self._jpeg: bytes | None = None
        self._events: dict[int, Any] = {}
        self._fit_images: list = []
        self._fit_geoms: list = []
        self._cal_images: list = []
        self._cal_geoms: list = []
        self._along_mm = 0.0
        self._frames_seen = 0
        self._frames_inspected = 0
        self._gap_warnings = 0
        self._latency_ms = 0.0
        self._error: str | None = None
        self._note = "waiting for the camera"
        self._started_inspecting: float | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, source_uri: str | None = None) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(source_uri,), name="inspector", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._go.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def begin_inspection(self) -> bool:
        """Operator pressed Start. Advance the camera and open the gate."""
        with self._lock:
            if self._phase != self.READY:
                return False
        self._advance_camera()
        self._go.set()
        return True

    def _advance_camera(self) -> None:
        """Tell the simulator to bring the sheet under test into view."""
        if not self.camera_control:
            return
        try:
            import urllib.request

            urllib.request.urlopen(
                urllib.request.Request(self.camera_control, method="POST"), timeout=3
            ).read()
        except Exception as exc:  # a run continues on the fabric in front of it
            with self._lock:
                self._note = f"camera did not advance: {exc}"

    # -- the loop ----------------------------------------------------------- #

    def _run(self, source_uri: str | None) -> None:
        try:
            source = self.pipeline._open_source(source_uri)
            self.pipeline.geometry.reset()
            self.pipeline.tracker.reset()
            with source:
                for frame in source:
                    if self._stop.is_set():
                        return
                    self._handle(frame)
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    def _handle(self, frame: Frame) -> None:
        try:
            geometry = self.pipeline.geometry.read(frame.image, frame.timestamp)
        except RuntimeError:
            geometry = None

        with self._lock:
            phase = self._phase
            self._frames_seen += 1
            if geometry is not None:
                self._along_mm = geometry.along_mm
                self._gap_warnings = self.pipeline._gap_warnings

        if phase == self.LEARNING:
            self._collect_fit(frame, geometry)
        elif phase == self.CALIBRATING:
            self._collect_cal(frame, geometry)
        elif phase == self.READY:
            self._show(frame, geometry, "READY - press Start Inspection")
            self._go.wait(timeout=0.2)
            if self._go.is_set():
                with self._lock:
                    self._phase = self.INSPECTING
                    self._started_inspecting = time.monotonic()
                    self._note = "inspecting the sheet under test"
        else:
            self._inspect(frame, geometry)

    def _collect_fit(self, frame: Frame, geometry: Any) -> None:
        self._fit_images.append(frame.image)
        self._fit_geoms.append(geometry)
        n = len(self._fit_images)
        self._show(frame, geometry, f"LEARNING clean fabric  {n}/{self.n_fit_frames}")
        if n < self.n_fit_frames:
            return
        with self._lock:
            self._note = "fitting the memory bank ..."
        self.pipeline._fit_from_frames(self._fit_images, self._fit_geoms)
        with self._lock:
            self._phase = self.CALIBRATING
            self._note = "bank fitted; measuring clean fabric for the threshold"

    def _collect_cal(self, frame: Frame, geometry: Any) -> None:
        self._cal_images.append(frame.image)
        self._cal_geoms.append(geometry)
        n = len(self._cal_images)
        self._show(
            frame, geometry, f"CALIBRATING on clean fabric  {n}/{self.n_calibration_frames}"
        )
        if n < self.n_calibration_frames:
            return
        try:
            scores = self.pipeline._maxima_from_images(self._cal_images, self._cal_geoms)
            self.pipeline.calibration = calibrate(
                scores,
                self.config.calibration.budget_fa_per_100m,
                self.pipeline.metres_per_tile,
                self.config.calibration.abstain_multiplier,
                self.config.calibration.stability_margin,
                sku=self.config.sku,
            )
            with self._lock:
                self._phase = self.READY
                self._note = (
                    f"threshold {self.pipeline.calibration.threshold:.2f} from "
                    f"{self.pipeline.calibration.n_clean_tiles} held-out clean tiles"
                )
        except ValueError as exc:
            # The clean sample cannot yet resolve the stated budget, so watch
            # more fabric rather than failing the run. Every frame already
            # collected is kept: discarding them would make the target recede
            # faster than the sample grows towards it.
            self.n_calibration_frames = n + 20
            with self._lock:
                self._note = f"need more clean fabric: {str(exc).split('.')[0]}"

    def _inspect(self, frame: Frame, geometry: Any) -> None:
        result = self.pipeline._process_with_geometry(frame, geometry)
        closed = self.pipeline.tracker.update(result.detections)
        with self._lock:
            for event in closed:
                self._events[event.event_id] = event
            for track in getattr(self.pipeline.tracker, "_open", []):
                self._events[track.event.event_id] = track.event
            self._frames_inspected += 1
            self._latency_ms = result.latency.total_ms
        self._publish(self._draw(result, None))

    # -- drawing ------------------------------------------------------------ #

    def _show(self, frame: Frame, geometry: Any, banner: str) -> None:
        """A frame with a phase banner but no detections drawn."""
        from ..pipeline import _crop_roi

        image = (
            _crop_roi(frame.image, geometry.roi) if geometry is not None else frame.image
        )
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._banner(canvas, banner, geometry, (245, 190, 60))
        self._publish(canvas)

    def _draw(self, result: Any, _unused: Any) -> np.ndarray:
        from ..pipeline import _crop_roi
        from ..scoring.astm_d5430 import points_for_length

        image = _crop_roi(result.frame.image, result.geometry.roi)
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for det in result.detections:
            x, y, w, h = det.bbox_px
            asserted = det.assertion is Assertion.ASSERTED
            colour = (60, 60, 235) if asserted else (40, 190, 245)
            cv2.rectangle(canvas, (x - 3, y - 3), (x + w + 3, y + h + 3), colour, 2)
            label = f"{det.length_mm:.0f}mm {points_for_length(det.length_mm)}pt"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(th + 6, y - 3)
            cv2.rectangle(canvas, (x - 3, ly - th - 6), (x + tw + 3, ly), (25, 25, 25), -1)
            cv2.putText(canvas, label, (x, ly - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)

        n = len(result.detections)
        self._banner(
            canvas,
            f"INSPECTING   {n} defect{'' if n == 1 else 's'} in view",
            result.geometry,
            (120, 235, 140),
        )
        return canvas

    def _banner(self, canvas: np.ndarray, text: str, geometry: Any, colour) -> None:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (18, 18, 18), -1)
        cv2.putText(canvas, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, colour, 2, cv2.LINE_AA)
        if geometry is not None:
            right = f"{geometry.along_mm / 1000:6.3f} m   {geometry.mm_per_px:.4f} mm/px"
            (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(canvas, right, (canvas.shape[1] - tw - 12, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    def _publish(self, canvas: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    # -- readers ------------------------------------------------------------ #

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stats(self) -> dict:
        from ..scoring.astm_d5430 import points_for_event, points_per_100yd2

        with self._lock:
            phase, note, error = self._phase, self._note, self._error
            events = list(self._events.values())
            along_mm, frames = self._along_mm, self._frames_inspected
            gap, latency = self._gap_warnings, self._latency_ms
            fit_n, cal_n = len(self._fit_images), len(self._cal_images)
            since = self._started_inspecting

        progress = 0.0
        if phase == self.LEARNING:
            progress = fit_n / max(1, self.n_fit_frames)
        elif phase == self.CALIBRATING:
            progress = cal_n / max(1, self.n_calibration_frames)
        elif phase in (self.READY, self.INSPECTING):
            progress = 1.0

        rejected = sum(1 for e in events if e.status is EventStatus.REJECTED)
        counted = [e for e in events if e.counts_toward_score]
        points = sum(points_for_event(e) for e in counted)
        inspected_m = max((time.monotonic() - since) if since else 0.0, 0.0) * 0.04
        metres = max(inspected_m, 1e-6)
        cal = self.pipeline.calibration

        return {
            "phase": phase,
            "note": note,
            "progress": round(min(1.0, progress), 3),
            "roll_id": self.roll_id,
            "sku": self.config.sku,
            "along_m": round(along_mm / 1000.0, 3),
            "frames_inspected": frames,
            "frame_stride": self.config.acquisition.frame_stride,
            "latency_ms": round(latency, 1),
            "events": len(events),
            "asserted": sum(1 for e in events if e.assertion is Assertion.ASSERTED),
            "uncertain": sum(1 for e in events if e.assertion is Assertion.UNCERTAIN),
            "confirmed": sum(1 for e in events if e.status is EventStatus.CONFIRMED),
            "rejected": rejected,
            "total_points": points,
            "points_per_100yd2": round(
                points_per_100yd2(points, metres, self.config.scoring.roll_width_m), 1
            ) if metres > 1e-3 else 0.0,
            "fa_per_100m": round(rejected * 100.0 / metres, 2) if metres > 1e-3 else 0.0,
            "budget_fa_per_100m": cal.budget_fa_per_100m if cal else None,
            "threshold": round(cal.threshold, 3) if cal else None,
            "n_clean_tiles": cal.n_clean_tiles if cal else 0,
            "spatial_resolution_mm": round(self.pipeline.spatial_resolution_mm, 1),
            "gap_warnings": gap,
            "error": error,
        }

    def events(self) -> list[dict]:
        from ..scoring.astm_d5430 import points_for_event

        with self._lock:
            events = sorted(self._events.values(), key=lambda e: -e.along_start_mm)
        return [
            {
                "id": e.event_id,
                "along_m": round(e.along_start_mm / 1000.0, 3),
                "across_mm": round(e.across_start_mm, 1),
                "length_mm": round(e.length_mm, 1),
                "width_mm": round(e.width_mm, 1),
                "score": round(e.max_score, 2),
                "assertion": e.assertion.value,
                "status": e.status.value,
                "points": points_for_event(e),
                "n_frames": e.n_frames,
            }
            for e in events[:60]
        ]

    def set_status(self, event_id: int, status: EventStatus) -> bool:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                return False
            event.status = status
        return True


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

OPERATOR_PAGE: str = """<!doctype html>
<html><head><meta charset="utf-8"><title>LineSight — live inspection</title>
<style>
 :root{--bg:#0e1013;--panel:#171a1f;--line:#262b33;--ink:#e6e9ee;--dim:#8b93a1;
       --red:#eb3c3c;--amber:#f5be28;--green:#5ac87a;--blue:#5aa9e6}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;
   align-items:center;gap:14px;flex-wrap:wrap}
 h1{font-size:16px;margin:0;font-weight:650}
 .sku{color:var(--dim);font-size:13px}
 .steps{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
 .step{font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid var(--line);
   color:var(--dim);white-space:nowrap}
 .step.on{background:#1d2b3a;border-color:var(--blue);color:#cfe6fb;font-weight:600}
 .step.done{border-color:#2c4a35;color:var(--green)}
 .wrap{display:grid;grid-template-columns:1fr 400px;gap:16px;padding:16px;align-items:start}
 @media(max-width:1150px){.wrap{grid-template-columns:1fr}}
 .video{background:#000;border:1px solid var(--line);border-radius:8px;
   overflow:hidden;line-height:0}
 .video img{width:100%;display:block}
 .bar{height:6px;background:#22262e;border-radius:4px;overflow:hidden;margin-top:10px}
 .bar i{display:block;height:100%;background:var(--blue);transition:width .3s}
 .note{color:var(--dim);font-size:12.5px;margin-top:8px}
 .cta{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 .big{background:var(--green);color:#06210f;border:0;border-radius:7px;padding:11px 22px;
   font-size:15px;font-weight:700;cursor:pointer}
 .big:disabled{background:#2b3038;color:#6d7683;cursor:not-allowed}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
   padding:14px;margin-bottom:14px}
 .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
   margin:0 0 10px;font-weight:600}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 12px}
 .stat .v{font-size:21px;font-weight:640;font-variant-numeric:tabular-nums}
 .stat .k{font-size:11px;color:var(--dim)}
 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th{text-align:left;font-size:10px;text-transform:uppercase;color:var(--dim);
   font-weight:600;padding:5px 4px;border-bottom:1px solid var(--line)}
 td{padding:5px 4px;border-bottom:1px solid #1e222a;font-size:12.5px}
 tr.new td{animation:flash 1.4s ease-out}
 @keyframes flash{from{background:#2a1f10}to{background:transparent}}
 .asserted{color:var(--red);font-weight:600} .uncertain{color:var(--amber)}
 button.sm{background:#232831;color:var(--ink);border:1px solid var(--line);
   border-radius:5px;padding:2px 8px;font-size:11px;cursor:pointer}
 button.sm:hover{background:#2c323d}
 .warn{background:#3d1616;border-color:#5e2020;color:#ffc9c9}
</style></head><body>
<header>
  <h1>LineSight</h1><span class="sku" id="hdr">connecting…</span>
  <div class="steps">
    <span class="step" id="s1">1 · learn clean fabric</span>
    <span class="step" id="s2">2 · calibrate</span>
    <span class="step" id="s3">3 · ready</span>
    <span class="step" id="s4">4 · inspecting</span>
  </div>
</header>
<div class="wrap">
  <div>
    <div class="video"><img id="feed" src="/stream.mjpg" alt="live inspection"></div>
    <div class="bar"><i id="prog" style="width:0%"></i></div>
    <div class="note" id="note">…</div>
    <div class="cta">
      <button class="big" id="go" disabled onclick="startInspection()">Start inspection</button>
      <span class="note" id="cta-hint">the model is still learning the clean fabric</span>
    </div>
    <div class="note" style="margin-top:14px">
      Red box = asserted defect · Amber = uncertain (inside the abstention band,
      scores zero points until confirmed) · Banner shows the phase, the roll
      position read from the ArUco tape, and the measured mm/px scale.
    </div>
  </div>
  <div>
    <div class="card" id="errcard" style="display:none"></div>
    <div class="card">
      <h2>Detections</h2>
      <table><thead><tr><th>#</th><th>at (m)</th><th>len</th><th>score</th>
        <th>state</th><th>pts</th><th></th></tr></thead>
        <tbody id="rows"><tr><td colspan="7" style="color:#8b93a1">none yet</td></tr></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Roll</h2>
      <div class="grid">
        <div class="stat"><div class="v" id="ev">—</div><div class="k">defects found</div></div>
        <div class="stat"><div class="v" id="pts">—</div><div class="k">ASTM points</div></div>
        <div class="stat"><div class="v" id="pos">—</div><div class="k">position (m)</div></div>
        <div class="stat"><div class="v" id="fr">—</div><div class="k">frames inspected</div></div>
      </div>
    </div>
    <div class="card">
      <h2>Threshold</h2>
      <div class="grid">
        <div class="stat"><div class="v" id="th">—</div><div class="k">score threshold</div></div>
        <div class="stat"><div class="v" id="bud">—</div><div class="k">budget FA/100 m</div></div>
      </div>
      <div class="note" id="calnote">not calibrated yet</div>
    </div>
    <div class="card">
      <h2>Honesty panel</h2>
      <div class="grid">
        <div class="stat"><div class="v" id="lat">—</div><div class="k">ms / frame</div></div>
        <div class="stat"><div class="v" id="strd">—</div><div class="k">frame stride</div></div>
        <div class="stat"><div class="v" id="res">—</div><div class="k">resolution (mm)</div></div>
        <div class="stat"><div class="v" id="gaps">—</div><div class="k">gap warnings</div></div>
      </div>
      <div class="note">
        Every detection is an <b>unclassified anomaly</b> — the system reports
        that a region is unlike defect-free fabric, not what kind of defect it
        is. Extents at or below the resolution figure are resolution-limited,
        not measurements. No defect was ever labelled to train this.
      </div>
    </div>
  </div>
</div>
<script>
const $ = i => document.getElementById(i);
let seen = new Set();

async function startInspection(){
  $('go').disabled = true;
  $('cta-hint').textContent = 'bringing the sheet under test into view…';
  await fetch('/api/start', {method:'POST'});
}
async function act(id, what){
  await fetch(`/api/events/${id}/${what}`, {method:'POST'}); tick();
}

const ORDER = ['learning','calibrating','ready','inspecting'];
function paintSteps(phase){
  const at = ORDER.indexOf(phase);
  ORDER.forEach((p,i) => {
    const el = $('s'+(i+1));
    el.className = 'step' + (i < at ? ' done' : i === at ? ' on' : '');
  });
}

async function tick(){
  try{
    const s = await (await fetch('/api/stats')).json();
    paintSteps(s.phase);
    $('hdr').textContent = `${s.sku} · roll ${s.roll_id}`;
    $('note').textContent = s.note;
    $('prog').style.width = (s.progress*100).toFixed(0)+'%';
    $('prog').style.background = s.phase==='inspecting' ? 'var(--green)' : 'var(--blue)';

    $('ev').textContent  = s.events;
    $('pts').textContent = s.total_points;
    $('pos').textContent = s.along_m.toFixed(2);
    $('fr').textContent  = s.frames_inspected;
    $('th').textContent  = s.threshold == null ? '—' : s.threshold.toFixed(2);
    $('bud').textContent = s.budget_fa_per_100m == null ? '—' : s.budget_fa_per_100m;
    $('lat').textContent = s.latency_ms.toFixed(0);
    $('strd').textContent= s.frame_stride;
    $('res').textContent = s.spatial_resolution_mm;
    $('gaps').textContent= s.gap_warnings;
    $('calnote').textContent = s.threshold == null ? 'not calibrated yet'
      : `from ${s.n_clean_tiles} held-out clean tiles. We never picked a `
        + 'threshold; we picked a false-alarm budget.';

    const ready = s.phase === 'ready';
    $('go').disabled = !ready;
    $('cta-hint').textContent =
      s.phase==='learning'    ? 'watching clean fabric to build the bank' :
      s.phase==='calibrating' ? 'measuring clean fabric to set the threshold' :
      s.phase==='ready'       ? 'model is ready — press to bring in the test sheet' :
                                'inspecting the sheet under test';

    const ec = $('errcard');
    if(s.error){ ec.style.display='block'; ec.className='card warn';
                 ec.textContent = 'stream error — ' + s.error; }
    else { ec.style.display='none'; }

    const evs = await (await fetch('/api/events')).json();
    $('rows').innerHTML = evs.length ? evs.map(e => {
        const isNew = !seen.has(e.id); seen.add(e.id);
        return `<tr class="${isNew?'new':''}">
          <td>${e.id}</td>
          <td>${e.along_m.toFixed(3)}</td>
          <td>${e.length_mm.toFixed(0)}mm</td>
          <td>${e.score.toFixed(1)}</td>
          <td class="${e.assertion}">${e.status==='proposed'?e.assertion:e.status}</td>
          <td>${e.points}</td>
          <td style="white-space:nowrap">
            <button class="sm" onclick="act(${e.id},'confirm')">✓</button>
            <button class="sm" onclick="act(${e.id},'reject')">✕</button></td>
        </tr>`;
      }).join('')
      : '<tr><td colspan="7" style="color:#8b93a1">none yet</td></tr>';
  }catch(err){ /* server restarting; the next tick catches up */ }
}
tick(); setInterval(tick, 800);
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def create_app(
    config: Config | None = None,
    source_uri: str | None = None,
    n_fit_frames: int = 30,
    n_calibration_frames: int = 120,
) -> Any:
    """Build the FastAPI application with its inspector and routes attached.

    Imported lazily by the CLI so that ``linesight run`` never pays for
    FastAPI, and so a base install without the ``api`` extra can still fit,
    calibrate, inspect and report from the command line.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    from ..pipeline import Pipeline

    config = config or Config()
    pipeline = Pipeline(config)
    inspector = LiveInspector(
        config,
        pipeline,
        n_fit_frames=n_fit_frames,
        n_calibration_frames=n_calibration_frames,
        camera_control=_control_url(source_uri or config.acquisition.uri),
    )

    app = FastAPI(title="LineSight", version="0.1.0")
    app.state.inspector = inspector

    @app.on_event("startup")
    def _startup() -> None:
        inspector.start(source_uri)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        inspector.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return OPERATOR_PAGE

    @app.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg(inspector),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/start")
    def start_inspection() -> dict:
        """Operator pressed Start: bring the sheet under test into view."""
        return {"ok": inspector.begin_inspection(), "phase": inspector.stats()["phase"]}

    @app.get("/api/stats")
    def stats() -> JSONResponse:
        return JSONResponse(inspector.stats())

    @app.get("/api/events")
    def events() -> JSONResponse:
        return JSONResponse(inspector.events())

    @app.post("/api/events/{event_id}/confirm")
    def confirm(event_id: int) -> dict:
        if not inspector.set_status(event_id, EventStatus.CONFIRMED):
            raise HTTPException(404, f"no event {event_id}")
        return {"ok": True}

    @app.post("/api/events/{event_id}/reject")
    def reject(event_id: int) -> dict:
        """Operator says false alarm. This is what the FA counter counts."""
        if not inspector.set_status(event_id, EventStatus.REJECTED):
            raise HTTPException(404, f"no event {event_id}")
        return {"ok": True}

    @app.get("/api/rolls")
    def rolls() -> JSONResponse:
        from .store import Store

        with Store(config.api.db_path) as store:
            return JSONResponse(store.list_rolls())

    @app.get("/api/rolls/{roll_id}")
    def roll(roll_id: str) -> JSONResponse:
        from .store import Store

        with Store(config.api.db_path) as store:
            report = store.get_roll(roll_id)
        if report is None:
            raise HTTPException(404, f"no roll {roll_id}")
        from ..cli import _report_dict

        return JSONResponse(_report_dict(report))

    return app


def _mjpeg(inspector: LiveInspector):
    """Yield the latest annotated frame forever, as a multipart stream."""
    blank = _placeholder()
    while True:
        jpeg = inspector.latest_jpeg() or blank
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.05)


def _placeholder() -> bytes:
    """Shown until the first inspected frame arrives."""
    canvas = np.full((360, 640, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, "waiting for the camera...", (140, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", canvas)[1].tobytes()


def _control_url(source_uri: str) -> str | None:
    """The simulator's /start-inspection endpoint, derived from its video URL.

    Only meaningful for the simulator. A real camera has no phase to advance -
    the operator physically swaps the cloth - so this stays None and the Start
    button simply opens the inspection gate.
    """
    if not source_uri.startswith("http"):
        return None
    base = source_uri.rsplit("/", 1)[0]
    return f"{base}/start-inspection"


def _load_calibration(config: Config) -> Any:
    """Restore this SKU's saved calibration, if one exists."""
    from ..types import Calibration

    path = Path(config.detect.bank_dir) / f"{config.sku}.calibration.json"
    if not path.exists():
        return None
    return Calibration(**json.loads(path.read_text(encoding="utf-8")))


def run_server(
    config: Config,
    host: str | None = None,
    port: int | None = None,
    source_uri: str | None = None,
    n_fit_frames: int = 30,
    n_calibration_frames: int = 120,
) -> None:
    """Serve with uvicorn. Blocks."""
    import uvicorn

    host = host or config.api.host
    port = port or config.api.port
    print(f"\n  operator page   http://{host}:{port}/")
    print(f"  camera feed     {source_uri or config.acquisition.uri}\n")
    uvicorn.run(
        create_app(config, source_uri, n_fit_frames, n_calibration_frames),
        host=host,
        port=port,
        log_level="warning",
    )
