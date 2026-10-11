# Neural Derby Implementation Plan

STATUS: OUTLINE ONLY. The task map below is complete and ordered; the per-task steps (failing test, run, implement, run, commit) are not yet written. Next session: invoke superpowers:writing-plans against the spec and expand each task below into full steps, keeping these names, files, and interfaces.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live, projector-and-phones show where seven small CNN/transformer "horses" built from the course's Module 3 blocks race to learn photos the room just took, with betting, spoken commentary, measured verdicts, and on-phone deployment of the champion.

**Architecture:** One Python process (FastAPI + uvicorn) serves a stadium page, a phone page, and a control page over plain HTTP (port 8000) plus an HTTPS listener (port 8443) for the team phone. A lock-step trainer on the GPU yields lap events; pure-function scoring, caller, and verdict modules turn events into standings, lines, and explanations; a websocket hub streams them to the pages. The champion is exported to ONNX and run on phones with onnxruntime-web.

**Tech Stack:** Python 3.12 venv, PyTorch (CUDA) + torchvision, FastAPI, uvicorn, onnx, onnxruntime, Pillow, numpy, qrcode, pytest. Frontend: plain HTML/CSS/JS, Press Start 2P font vendored, onnxruntime-web vendored, WebAudio for sound, browser speech synthesis for the caller.

Spec: `docs/superpowers/specs/2026-10-12-neural-derby-design.md` (approved 2026-10-12).

## Global Constraints

- Python 3.12; project lives in `neural-derby/` under the repo root `C:\dev\study\Edge-Ai`.
- Image size 96x96 for all horses; 6 classes: bottle, pen, book, phone, backpack, shoe.
- 20 laps, batch 64, Adam for all horses, same shuffle order per lap, pacing floor 4 s per lap.
- Upload caps: 3 per device, 120 total, 5 MB, JPEG/PNG only, EXIF rotation applied, stored at 160 px long side.
- Live split by device, half/half; fewer than 12 live photos means all go to test. Backup split 85/15 by seed.
- Nothing on the pages may need internet: fonts, scripts, runtime all served locally.
- Commentary and verdict text must be generated only from measured numbers; no scripted outcomes.
- No attribution lines in commits (user rule). Commit messages in plain words.
- Every horse is one of ma'am's blocks: plain conv, residual, channel attention, spatial attention, inception, depthwise separable, transformer encoder.

---

## File map (from spec section 11)

See spec section 11 for the full tree. Each task below names exactly which of those files it creates.

## Task map (ordered by dependency)

### Task 1: Scaffold, environment, config
Files: `neural-derby/pyproject.toml` or `requirements.txt`, `neural-derby/derby/__init__.py`, `neural-derby/derby/config.py`, `neural-derby/tests/test_config.py`, `neural-derby/docs/setup.md`.
Produces: `config.CLASSES: list[str]`, `config.LAPS=20`, `config.BATCH=64`, `config.IMG=96`, `config.CAPS = {"per_device":3,"total":120,"max_bytes":5_000_000}`, `config.PORTS={"http":8000,"https":8443}`, `config.LAP_FLOOR_S=4.0`, `config.HORSES_ENABLED: dict[str,bool]` (INFLUENCER default True).
Verify: venv created, `torch.cuda.is_available()` True on the RTX 4050, pytest runs one passing test.

### Task 2: Ma'am's blocks
Files: `derby/horses/blocks.py`, `tests/test_blocks.py`.
Produces: `ConvBlock(cin,cout)`, `ResidualBlock(c)`, `ChannelAttention(c, r=16)` (fully conv, avg+max pooled, sigmoid scale), `SpatialAttention(k=7)`, `InceptionBlock(cin, branch)`, `DepthwiseSeparable(cin,cout)`, `PatchEmbed(patch=8, dim=64)`, `TransformerEncoderBlock(dim=64, heads=4)`. All take and return `(B,C,H,W)` except the transformer pieces which use `(B,T,D)`.
Tests: output shapes for a 2x3x96x96 input; attention output equals input times a value in [0,1].

### Task 3: The stable and the stats
Files: `derby/horses/stable.py`, `derby/horses/stats.py`, `tests/test_stable.py`, `tests/test_stats.py`.
Produces: `Horse` dataclass `(key, name, tagline, colour, block_name, build: Callable[[], nn.Module])`; `STABLE: list[Horse]` with keys `textbook, stairs, elevator, featherweight, spotlight, trifocals, influencer`; `stats.count_params(m)`, `stats.count_macs(m, img=96)` via forward hooks on Conv2d/Linear, `stats.layer_table(m, img=96) -> list[dict(layer, out_shape, params)]`.
Tests: every enabled horse maps 1x3x96x96 to 1x6; `count_params` equals `sum(p.numel())`; MACs for a known 3x3 conv on 8x8 equals hand value; depthwise MACs strictly less than textbook's.

### Task 4: Backup dataset tool and loader
Files: `tools/build_backup_dataset.py`, `derby/data/backup.py`, `tests/test_backup.py`.
Produces: tool writes `data/backup/<class>/*.jpg` from LVIS train JSON + COCO image URLs (mapping and filters per spec section 5; `--per-class 400 --min-side 64 --max-per-image 2`); `backup.load(root) -> (uint8 tensor N,3,96,96, labels N, source_ids)`, `backup.split(seed, test_frac=0.15)`.
Tests: loader on a tiny fixture folder of 12 images; split is deterministic and disjoint.

### Task 5: Live uploads
Files: `derby/data/live.py`, `tests/test_live.py`.
Produces: `LiveStore(root, caps)` with `add(device_id, class_name, bytes) -> Photo | LiveError`, `count()`, `by_device()`, `split_by_device(seed) -> (train_ids, test_ids)` applying the fewer-than-12 rule, `tensors()`.
Tests: per-device cap, global cap, size cap, bad bytes rejected, EXIF-rotated fixture comes out upright, device split never leaks a device.

### Task 6: GPU augmentation and dataset assembly
Files: `derby/data/augment.py`, `derby/data/dataset.py`, `tests/test_augment.py`.
Produces: `augment.batch(x_uint8, gen) -> float tensor` applying resized crop, flip, brightness/contrast, rotation, noise, erasing; `augment.preview_pairs(x, n=48) -> list[(orig_png_b64, aug_png_b64)]`; `dataset.assemble(backup, live, seed) -> RaceData(train_x, train_y, test_x, test_y, test_is_live mask)` on the chosen device.
Tests: output shape and range; preview returns n pairs; assemble keeps live photos on one side only.

### Task 7: Lock-step trainer
Files: `derby/race/trainer.py`, `tests/test_trainer.py`.
Produces: `LapEvent(horse, lap, train_acc, test_acc, test_acc_internet, test_acc_live, loss, ms)`; `run_race(stable, data, laps, batch, lap_floor_s, seed) -> Iterator[LapEvent]` training one epoch per horse per lap in lock-step with a shared shuffle order and the pacing floor.
Tests: synthetic 2-horse, 2-lap race on CPU yields 4 events in lock-step order; pacing floor honoured within tolerance when set to 0.1 s.

### Task 8: Scoring and recording
Files: `derby/race/scoring.py`, `derby/race/recorder.py`, `tests/test_scoring.py`, `tests/test_recorder.py`.
Produces: `race_standings(events)`, `sprint_standings(ms_by_horse)`, `champion(race_rank, sprint_rank, final_acc)`, `settle_bets(bets, champion, race_winner)`; `Recorder.write(path, events, standings, verdicts)`, `Recorder.replay(path) -> Iterator[LapEvent]` with timing, `form_guide(race_files) -> {horse: wins, races}`.
Tests: ties resolved per spec; rank sums; replay reproduces events; form guide counts.

### Task 9: Caller and verdicts
Files: `derby/commentary/caller.py`, `derby/commentary/verdict.py`, `tests/test_caller.py`, `tests/test_verdict.py`.
Produces: `Caller.on_event(event, standings) -> list[str]` implementing the nine rules; `facts(horse, events, stats, sprint_ms) -> Facts`; `verdict(facts_by_horse) -> {horse: list[str]}` with the guarded rules from spec section 8.
Tests: each rule fires exactly on its synthetic stream; no slow-start claim below the lap-gap threshold; no memorising claim below 20 points.

### Task 10: ONNX export and laptop bench
Files: `derby/export/onnx_export.py`, `derby/export/bench.py`, `tests/test_export.py`.
Produces: `export(model, path, img=96)` opset 17 with a torch-vs-onnxruntime check within 1e-4; `bench_cpu(path, n=30) -> median_ms`.
Tests: every enabled horse exports and matches.

### Task 11: Show state machine and server
Files: `derby/server/show.py`, `derby/server/app.py`, `tests/test_show.py`, `tests/test_app.py`.
Produces: acts `LOBBY, PADDOCK, MULTIPLY, HORSES, DERBY, SPRINT, VERDICT, SHIP`; `Show.advance()` with gates; routes `POST /api/photo`, `POST /api/bet`, `GET /api/state`, `POST /api/sprint-result`, `GET /models/champion.onnx`, `WS /ws`; static mounts for `web/`.
Tests: illegal transitions rejected; DERBY blocked until data loaded; upload route enforces caps; websocket broadcasts a lap event.

### Task 12: Stadium page (arcade UI)
Files: `web/stadium/index.html`, `stadium.css`, `stadium.js`, `sprites.js`, `audio.js`, `web/vendor/press-start-2p.woff2`.
Produces: renders every act from websocket messages only; lanes, tote board, ticker, speech synthesis, cards, verdict flip, podium; forgotten states per spec section 10.
Verify: open in the built-in browser with a replayed race; screenshot each act at 1920x1080; check legibility.

### Task 13: Phone and control pages
Files: `web/phone/index.html`, `phone.css`, `phone.js`, `web/control/index.html`, `web/vendor/onnxruntime-web/*`.
Produces: join with nickname + stored device token, class buttons, camera file input, thumbnail + retake, bet screen, live mini standings, result screen, snapshot inference with onnxruntime-web, Sprint timing routine posting `median_ms`; control page shows GPU, phones, photos, and fallback buttons.
Verify: phone preset in the built-in browser at 360 px; no network calls leave the laptop.

### Task 14: Practice races, balancing, form guide, Veteran
Files: `tools/practice_races.py`, `derby/horses/veteran.py`, `tests/test_veteran.py`.
Produces: headless N-race runner writing `data/races/practice-*.json` and `data/form_guide.json`; width/depth knobs in `stable.py` adjusted with evidence until no horse wins more than about a third; `veteran.build()` (MobileNetV3-small, new head) and its one-lap epilogue event.
Verify: paste the 20-race win table into `docs/learning/02-the-stable.md`.

### Task 15: HTTPS team-phone listener and Sprint round-trip
Files: `tools/make_cert.py`, changes to `derby/server/app.py`, `web/phone/phone.js` (live video mode on HTTPS).
Produces: second uvicorn listener on 8443; live video classification on the team phone; Sprint results flow phone to server to stadium; fallback to `bench_cpu` labelled on screen.
Verify: real Android phone over the chosen network.

### Task 16: Learning docs, demo script, rehearsal
Files: `docs/learning/00-overview.md`, `01-race-engine.md`, `02-the-stable.md`, `03-the-grandstand.md`, `qa-drill.md`, `docs/demo-script.md`.
Produces: a 15-year-old-readable chapter per role, the module map, the likely-questions drill with answers, and a minute-by-minute script naming who speaks.
Verify: full rehearsal with three phones; each act timed; prep checklist (spec section 15) ticked.
