# ADR-009 — No Jetson. Measure on local hardware and state the extrapolation

**Status:** Accepted · **Date:** 2026-08-31

## Context

The deployment target for this system is an edge device beside a fabric
inspection frame, and a Jetson is the obvious candidate. Measuring on one would
be the strongest possible latency evidence.

No Jetson is available here, and the alternatives are all worse than they look:
quoting someone else's Jetson benchmark for a different model, quoting a
theoretical FLOP count, or saying nothing about latency at all.

## Decision

Measure honestly on the hardware that is available. Publish the per-stage
table — frame
decode and geometry, flat-field and tiling, backbone forward, nearest-neighbour
search, event assembly and scoring — with real numbers from a real run, and state
the extrapolation to edge hardware as an extrapolation.

Publish the **frame sampling factor** with every latency figure. At a 25 mm/s
hand pull with 64 px tile overlap there is enormous temporal redundancy, so
processing every second or third frame is a legitimate engineering choice — but a
per-frame latency that hides a stride of 3 is not a measurement, it is a claim.

## Consequences

The latency table is defensible line by line. The optimisations it reflects are
also the honest ones available without special hardware: batching every tile of a
frame into a single forward pass, dropping the backbone input from 320 to 256
when needed, and increasing the stride.

`LatencyRecord` carries per-stage timings for every frame and `Pipeline.latency_table`
reports medians, so the numbers come from instrumentation rather than from a
stopwatch and a memory.

TensorRT and INT8 quantisation are explicitly not attempted. Both are edge-device
optimisations for hardware we cannot test on, and an unverified quantised model
is a worse claim than an unquantised measured one.

"Measured on a laptop CPU" is a weaker headline than "runs on a Jetson at
30 fps", and that cost is accepted: it is a considerably stronger position than
a number nobody can trace back to a run.
