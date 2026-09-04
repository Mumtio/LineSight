# ADR-003 — MVTec AD stays on Kaggle; local development runs on AITEX

**Status:** Accepted · **Date:** 2026-08-31

## Context

MVTec AD is the reproduction anchor: it is what published PatchCore numbers are
quoted against, and landing near one of them is what licenses our other results
to be believed. It is also about 5 GB. Downloading, unzipping and storing it on
a development laptop costs disk, bandwidth, and an unpredictable slice of setup
time, none of which buys anything a read-only mount does not.

It mounts read-only at `/kaggle/input/` with one click and never touches local
disk. Meanwhile AITEX is about 170 MB, has pixel-precise masks, encodes fabric
structure in the filename, and is a *more* relevant benchmark for this problem
than carpet and leather: it is actual woven fabric, and its seven fabric codes
give us leave-one-fabric-out for free.

## Decision

MVTec AD is used **only** from a Kaggle session, attached as a notebook input
as the first action of that session, before any code is written. It is never
fetched locally, and `scripts/fetch_data.sh` deliberately does not fetch it —
with a comment saying so, because its absence would otherwise look like an
oversight. Local development runs on AITEX plus Fabric Stain, together under
250 MB.

## Consequences

Setup does not begin with a 5 GB download. Local iteration is fast, and the main
path runs on the dataset whose defects most resemble the target domain.
Fabric Stain covers the one thing AITEX cannot: 1920×1080 phone-shaped frames,
which is what proves the tiler and the frame loop before the rig exists.

`datasets/mvtec.py` exists but is unreachable from the product path, and its
`kaggle_root()` raises with an explicit "attach the dataset as a notebook input"
message, because that is always the actual cause.

The licensing consequence has to be stated explicitly. MVTec AD is
CC BY-NC-SA 4.0 — **non-commercial** — while this is intended as a commercial
product. It is used for validation only; nothing derived from it ships; the
deployed system fits on the mill's own fabric. The separation between "the
dataset that never leaves Kaggle" and "the artifact that ships" is enforced by
what the product code can reach, not by a promise.
