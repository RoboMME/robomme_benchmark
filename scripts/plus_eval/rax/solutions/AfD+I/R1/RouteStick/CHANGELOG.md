# routestick controller changelog

## v1 (2026-11-16) — offline-only development (R1 budget: the single start is the rehearsal)
- Video route parser (red-disk arrivals + trail-appearance side rule); validated on 3 demos, 22/22 moves.
- Depth localization of disks, table, pillars (pixel_to_position tool, own back-projection fallback).
- Tip-offset calibration from early stall / first contact; rho-relative stall detection.
- Half-ellipse arcs around pillars on the parsed side; adaptive speed under step budget.
- Touch verification by red bullseye (front window / wrist) or stall at disk-top height; 2-offset search.
- Mock matrix 24/24 (experiments/matrix_164112.txt).
- v1 final edits before registration: image-label hints only from short non-JSON text blocks;
  follow() empty-row guard counts toward call limit. Regression: experiments/matrix2_*.txt.
