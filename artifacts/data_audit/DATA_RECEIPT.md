# REBOOT Pilot Data Receipt

- Dataset: `REBOOT26/sample_recovery-demonstration`
- Immutable revision: `0633573d0438be1185bddebdf1a2c8f5505f7b2a`
- Last modified: `2026-05-12T09:21:37.000Z`
- Retrieved through: `https://hf-mirror.com`
- Receipt generated (UTC): `2026-09-29T22:00:38.654605+00:00`
- Episodes observed: **60**
- Episodes usable: **53**
- Episodes quarantined: **7**
- Frames declared by `meta/info.json`: **53886**
- Sum of annotation terminal frame indices: **53820**
- Expected rows under inclusive terminal-index semantics: **53880**
- FPS: **30**
- RGB cameras: `observation.images.cam_high, observation.images.cam_left_wrist, observation.images.cam_low, observation.images.cam_right_wrist`
- Depth present in this pilot snapshot: **False**
- Robot state/action shape: `[14]` / `[14]`

## Integrity notes

- `episode_index` is `int64` in the frame-table schema but encoded as a zero-padded string in `meta/phase.json`; consumers normalize it explicitly.
- Official REBOOT phase numbering is one-based (1–5). Out-of-range values are quarantined rather than remapped.
- The annotation-duration sum does not equal `total_frames`; the mismatch is retained as evidence and is not silently repaired.
- Full frame tables were audited against the annotations; RGB video decoding and model execution are separate evidence.

## Source SHA-256

- `api.json`: `3670d463b9fcb76a67fcb467ec2b50e8e6423d47032a1b5b5b13d0bd3c36738d`
- `info.json`: `b0890cbbccd2b4db1e6b897d51ce038e2e40eec6cfd643698b01555b9abc099d`
- `phase.json`: `1d61ff591ff9f52628c49e71a0435ab83a62c87a4d607ae91a50478dbaf5813b`
