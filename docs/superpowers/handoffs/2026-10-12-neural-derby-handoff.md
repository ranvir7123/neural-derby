# Neural Derby build: handoff for the next session

Written 2026-10-12 at the end of a long build session. Read this first, then the ledger in the appendix.

## Where things stand

The plan `docs/superpowers/plans/2026-10-12-neural-derby.md` (16 tasks) is being executed with the
`superpowers:subagent-driven-development` skill: one fresh implementer subagent per task, a task review
after each, fix rounds until the review is clean, one commit per task (plus fix commits), push after each task.

- Tasks 1 to 11 are complete and reviewed clean. Every commit is pushed.
- Task 12 (stadium page) is implemented and committed at `b27fe1f` and pushed, but its **task review has not
  been done yet**. That is the first thing the next session does.
- Tasks 13, 14, 15, 16 remain.
- Tests: `107 passed`, no warnings (`pytest -q` from `neural-derby/`).

Branch `claude/elastic-tesla-5be637`, remote `origin` = https://github.com/ranvir7123/neural-derby (private).
Work happens in the git worktree `C:\dev\study\Edge-Ai\.claude\worktrees\elastic-tesla-5be637` (not the main
checkout). The venv, the built dataset, and the ledger all live in that worktree, so **resume in that worktree**.

## How to resume (exact steps)

1. Open a session with the working directory `C:\dev\study\Edge-Ai\.claude\worktrees\elastic-tesla-5be637`.
2. Invoke `superpowers:subagent-driven-development`. Its workspace for this plan is
   `.superpowers/sdd/2026-10-12-neural-derby/` (git-ignored). It holds `progress.md` (the ledger, also copied
   in the appendix below), `task-N-brief.md`, `task-N-report.md`, and `review-<base>..<head>.diff` files.
   The skill's ledger check will see Tasks 1 to 11 marked complete and resume at Task 12.
3. **Task 12 review first.** Inputs: brief `task-12-brief.md`, report `task-12-report.md`, diff package
   `review-3faca44..b27fe1f.diff` (regenerate it with the skill's `review-package` script if missing, but exclude
   `neural-derby/web/vendor/` from the diff; the vendored runtime is 11 MB of generated code nobody should read).
   Things the reviewer must check are listed under "Task 12 carry-overs" below. Then fix rounds as usual,
   then `Task 12: complete` in the ledger, then push.
4. Tasks 13 to 16 in order, each with the carry-overs listed below folded into the implementer's dispatch.
5. Final whole-branch review (most capable model), one fix wave, then `superpowers:finishing-a-development-branch`.

Rules that bind every dispatch (from the user's global instructions): commits in plain words with **no attribution
or co-author lines**; evidence rules (never claim something works without the command and its pasted output; say
what was not verified); plain language for a student; delegate test runs and searches to subagents; push after
each task's commit; stop only for decisions that are genuinely the user's or steps that need a human with a phone.

## Environment facts

- Python 3.12.10 via `py -3.12`; venv at `neural-derby\.venv` (call `neural-derby\.venv\Scripts\python.exe`
  by full path; never activate). torch `2.11.0+cu128`; `torch.cuda.is_available()` prints `True` on the
  `NVIDIA GeForce RTX 4050 Laptop GPU` (driver 591.74). fastapi 0.143, starlette 1.7, httpx 0.28 + httpx2 2.13
  (installed so Starlette's TestClient stops warning), onnxruntime 1.31, Pillow 12.3, pytest 9.1.
- Backup dataset built under `neural-derby/data/backup/` (git-ignored): bottle 400, pen 244, book 400, phone 400,
  backpack 400, shoe 382; zero duplicate files. Caches under `data/cache/` (LVIS zip 350 MB, COCO and Open Images
  photos). Rebuild is `python tools/build_backup_dataset.py` (idempotent, stable file names).
- Vendored under `neural-derby/web/vendor/` and committed: Press Start 2P font + OFL licence; onnxruntime-web
  1.20.1 (`ort.min.js`, `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm` 11.2 MB, LICENSE).
- `.claude/launch.json` at the worktree root: config `derby` runs uvicorn on port 8000 with this worktree's venv
  and `cwd`. Start it with the browser pane's `preview_start` (name `derby`); the server loads the 2,226 backup
  photos at startup, so wait for `GET /api/state` before opening pages.
- GPU lap times (3-lap smoke race at current widths): textbook 567, stairs 2113, elevator 2240, featherweight 537,
  spotlight 903, trifocals 658, influencer 496 ms. One lap for all seven is about 7.5 s, so a 20-lap race takes
  about 150 s (the 4 s floor never bites). Accuracies were weak early (0.20 to 0.37 at lap 9 in one live race).

## Decisions and deviations from the plan made so far (all reviewed)

- Stats count the transformer's two attention matrix products (`2*T*T*dim` per block); the plan said "not counted".
- Taglines with numbers are self-checking: STAIRS "11 floors deep", FEATHERWEIGHT "4.5 times fewer multiplications
  than TEXTBOOK"; `tests/test_stable.py::test_numbered_taglines_match_measured_stats` pins them to measured stats.
  Task 14 must update the words when widths change (the test fails until it does).
- `stats.layer_table` lists each weight layer once (ChannelAttention's shared MLP runs twice per forward).
- Dataset tool: stable file names `coco_<id>_<k>.jpg` / `oi_<split>_<id>_<k>.jpg`; per-class minimum side
  (`pen` and `shoe` 48 px, others 64); every box padded to a square and slid inside the photo; Open Images pen
  top-up from the test then validation split (spec section 5 names this fallback); md5 duplicate report; near-copy
  skip (overlap > 0.7 within a photo). Never add rotate-before-crop for Open Images (boxes match stored pixels).
- `backup.Loaded.groups` + `backup.split_by_group`: the backup set is split by source photo (twin crops from one
  photo never straddle train and test). `dataset.assemble` uses it when groups are present.
- Live store: accepts MPO (multi-picture JPEG from phone cameras), a lock around `add()`, every decode failure
  becomes a `LiveError` ("too big" for decompression bombs).
- Trainer: every horse sees identical augmented pixels in a lap (augmentation seeded per lap, not per horse).
- Caller/verdict: thresholds use an `EPS` tolerance so exact 8/20/2/10-point boundaries behave; STAIRS/ELEVATOR
  sentences handle both directions; Sprint times print one decimal under 10 ms; INFLUENCER "not enough data" only
  when it finished below TEXTBOOK; start line says "{n} brains"; the clear-finish line states the margin.
- ONNX export: classic exporter (`dynamo=False`) on torch 2.11, IR 8, opset 17, names `image`/`logits`,
  dynamic batch; the legacy exporter is deprecated (hidden by `pytest.ini`); a future torch may force the dynamo
  exporter.
- Server (`derby/server/app.py`): `lifespan` instead of `on_event`; worker-thread failures are logged and sent as
  `{"type":"error","where":"load"|"race","message"}` to stadium/control only; `race_running` flag (Space out of
  DERBY and Replay are refused while the race thread runs); `sprint_done` lock (late phone times cannot overwrite
  a decided Sprint); one-off messages (`previews`, `race_done`, `sprint_request`, `sprint`, `verdict`, `ship`,
  `veteran`) are remembered and re-sent to sockets that connect later, by act and role; `previews` go to
  stadium/control only; lap JSON is NaN-safe; bad `ms` values give 400.
- Security hardening after an automated review: `/api/replay` takes a bare `{"name": "<file>.json"}` inside
  `data/races` only (no path echo); operator endpoints (`advance`, `gate/close`, `sprint/laptop`, `replay`,
  `team-url`) are laptop-only (`Settings.local_only_ops`, client host 127.0.0.1/::1/localhost) so the stadium and
  control pages must be opened at `http://localhost:8000/...`; the websocket `team` role and `/api/sprint-result`
  need `Settings.team_token` (random per run unless given), shown by `GET /api/team-url` as
  `https://<ip>:8443/phone/?team=<token>`; bets stay keyed by the phone's random device id (spec design).
- `state` payload carries `laps`; stadium reads `LAPS_TOTAL` from it. `/api/join-url` exists.
- Stadium JS: `if (e.repeat) return;` on keydown; `esc()` for phone-typed nicknames and winner lists; `error`
  handler with a long toast; many visual fixes from the 1920x1080 walk (see task-12-report.md).

## Task 12 carry-overs (what the pending review must check)

- Every server message type has an idempotent handler (reconnect re-sends can arrive twice); `error`, `veteran`.
- Every field the JS reads exists in the server payloads; keys all seven bindings; `e.repeat` guard; 400/403 JSON
  errors surface as toasts.
- `esc()` on `nick` and winners; no other user-typed string reaches `innerHTML`.
- No internet: grep the three web files for `http`; font from `/vendor/`.
- Sprites two frames in code; audio no files; forgotten states (empty wall, gate closed, horse with no Sprint time,
  long nickname).
- Launch config paths point at the worktree venv/folder; `vendor_assets.py` filter excludes `.jsep` and pins the
  LICENSE to the v1.20.1 release; `app.py` conditional mounts still serve all folders when present.
- Implementer's own concerns: THE INFLUENCER verdict card has ~30 px to spare; sound, the real error path, the
  "LOADING PHOTOS" banner and the winner lists were not verified.

## Task 13 carry-overs (phone + control pages)

- Phone page must read `?team=<token>` from the URL, connect with `ws?role=team&token=<token>`, and send
  `token` in the `/api/sprint-result` body. `TEAM` is "token present", not `team=1`.
- Control page must send `{"name": ...}` (or `{}`) to `/api/replay`, show `/api/team-url`, show `error`
  messages, and have a "Run the Veteran" button (`POST /api/veteran`, Task 14).
- Render `nick`, `class`, `device` as text (`textContent`), never raw `innerHTML`.
- Phones receive `race_done`, `sprint`, `ship` on reconnect; handlers must redraw, not append.
- Browser walk at the mobile preset (375 wide) and at 1024 wide; check the ONNX runtime loads from `/vendor/`
  and that THE INFLUENCER's operators (LayerNormalization, Erf, Softmax) run in onnxruntime-web 1.20.1.
- `/api/photo`, `/api/bet`, `/api/join` are not operator-only, so a phone at the LAN address works.

## Task 14 carry-overs (practice races, balancing, Veteran)

- `POST /api/veteran` must be operator-only (reuse `operator_allowed`) and its broadcast must go through
  `send_and_remember` so late pages get it; the stadium already handles `veteran`.
- Changing widths will break `tests/test_stats.py` hard-coded TEXTBOOK expectations (`[24, 96, 96]`, `3*24*9`)
  and the tagline test; update the expected values and the tagline words to the new measured numbers.
- Budget the lap time as the SUM over horses (~570 ms per horse for a 4 s lap) if the 90 s race target matters;
  STAIRS/ELEVATOR (436 M MACs, ~2.2 s per lap each) are the main cost. Changing `DEEP_STAGES` changes the
  STAIRS floor count in its tagline.
- The form guide must come from practice files only (`practice-*.json`); `app.py` reads `data/form_guide.json`.
- Balancing rule (spec): no horse wins the Race more than about 7 of 20; Sprint order is a real speed order
  (FEATHERWEIGHT fastest). If no setting satisfies it, that decision is the user's: stop and ask.
- The MobileNetV3-small weights download needs internet once.

## Task 15 carry-overs (HTTPS, Sprint round-trip)

- The plan's `test_sprint_results...` test must send the team token; its `winners` assertion is tautological by
  the plan's own note (deferred minor; make it meaningful if cheap: assert against the champion in the message).
- `main.py` runs HTTP 8000 and HTTPS 8443 from one app; `tools/make_cert.py` writes `data/cert.pem`/`key.pem`.
- **Step 4 (real Android phone on the chosen network) needs the user in person.** Do Steps 1 to 3, commit, and
  hand Step 4 to the user with exact instructions; Task 16's phone millisecond table waits for it.

## Task 16 carry-overs (docs)

- Every number in the docs comes from real runs: Task 3's stats, Task 14's practice table, Task 15's phone ms
  (label laptop numbers as laptop if the phone run has not happened).
- Explain: train accuracy is measured on augmented batches during training (can sit below test early); GPU runs
  are not bit-repeatable; operator pages must be opened at localhost; the team phone URL carries a token; after a
  failed data load the server must be restarted.
- Step 6 (read-aloud check and rehearsal) needs the team in person.

## Deferred minors (for the final review to triage)

They are all in the ledger below, one line per task, prefixed `minor (deferred)`.

## Appendix: the ledger (`.superpowers/sdd/2026-10-12-neural-derby/progress.md`, verbatim)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-12-neural-derby.md

Worktree: C:/dev/study/Edge-Ai/.claude/worktrees/elastic-tesla-5be637 (branch claude/elastic-tesla-5be637)
Started: 2026-10-12

Pre-flight notes (not blocking): (a) Task 14 balancing example changes TEXTBOOK widths, which would break tests/test_stats.py hard-coded [24,96,96] and 3*24*9 expectations; resolution: Task 14 implementer updates those expected values to the new widths (same mechanism tested). (b) Task 15 test's `winners` assertion is tautological by the plan's own explanation; treat as deferred minor, surface in final report. (c) Task 15 Step 4 and Task 16 Step 6 need a human with a phone / an audience; do all other steps, then hand those to the user.

Task 1: dispatched (BASE 9e22020)
Task 1: minor (deferred): test_dirs_exist only checks names, not existence (plan-mandated)
Task 1: minor (deferred): pytest.ini ignores all DeprecationWarnings globally (plan-mandated); -q doubled with `pytest -q`
Task 1: minor (deferred): requirements.txt has lower bounds only, no lock; resolved: torch 2.11.0+cu128, numpy 2.5.2, onnxruntime 1.31.0, pytest 9.1.1, fastapi 0.143.0
Task 1: controller verified: venv Python 3.12.10; GPU check `2.11.0+cu128 True NVIDIA GeForce RTX 4050 Laptop GPU`; .venv and .pytest_cache ignored
Task 1: complete (commits 9e22020..137e94b, review clean)
Task 2: dispatched (BASE 137e94b)
Task 2: review found 1 Important (plan-mandated): TransformerEncoderBlock docstring overstates what the Linear-hook counter sees (attention matmuls q@k^T and att@v are ~36% of the block's MACs). Controller ruling: reword docstring now (fix round 1); Task 3 stats.py must add 2*T*T*dim per transformer block so the card number is honest (deviation from plan's "not counted" text, in favour of the spec's honesty rule; surface in final report).
Task 2: minor (deferred): no test locks PlainPair/ResidualBlock parameter parity (reviewer measured 10464 == 10464); attention/transformer/depthwise tests are shape-only; ChannelAttention hidden clamp max(c//r,4) undocumented; shared-MLP bias added twice (harmless)
Task 2: fix round 1 dispatched (resume implementer af8e991fbbf0380d1; FIX_BASE dfef29a)
Task 2: fix round 1/5 (1 addressed, 0 open — docstring now accurate; commits dfef29a..c25a04a)
Task 2: complete (commits 137e94b..c25a04a, review clean after 1 fix round)
Task 3: dispatched (BASE c25a04a)
Task 3: carry to Task 14: FEATHERWEIGHT tagline says 'Eight times fewer multiplications' but measured TEXTBOOK/FEATHERWEIGHT MAC ratio is 53,748,288/12,027,456 = 4.47x at current widths; fix the number once Task 14 widths are final (same rule as the STAIRS floor count)
Task 3: review found 2 Important (both plan-mandated): (1) FEATHERWEIGHT tagline "Eight times fewer" false (measured 4.47x); (2) layer_table repeats rows for layers called twice (ChannelAttention.score), SPOTLIGHT table 14 rows vs depth 10. Controller ruling (user asleep, honesty rule governs): taglines become "4.5 times fewer multiplications than TEXTBOOK" and "11 floors deep" with a test that pins both to measured stats; layer_table records each named layer once.
Task 3: minor (deferred): stats hooks not in try/finally; probe tensor is CPU-only (would fail for a GPU-resident model); no test for STAIRS==ELEVATOR MACs; stats imports Horse only for a type hint; INFLUENCER head differs slightly (mean+LayerNorm+Linear) from the conv horses' head (plan-mandated); attention hook assumes positional input; table rows exclude BatchNorm/LayerNorm/pos params so rows never sum to count_params (UI must not show a row total)
Task 3: fix round 1 dispatched (resume implementer a1842a4d152c9b46e; FIX_BASE cba646b)
Task 3: fix round 1/5 (2 addressed, 0 open — taglines self-checking, layer_table one row per layer; commits cba646b..53118cd)
Task 3: complete (commits c25a04a..53118cd, review clean after 1 fix round). Stats at current widths: textbook 53,406p/53.7M MACs/d4; stairs=elevator 438,078p/436.0M/d12; featherweight 8,118p/12.0M/d6; spotlight 55,294p/54.0M/d10; trifocals 68,766p/67.9M/d8; influencer 122,502p/23.9M/d20
Task 4: dispatched (BASE 53118cd)
Task 4: first implementer cut off by API rate limit after writing files and building data (no report, no commit). Counts after LVIS + min-side-48 top-up: bottle 400, pen 94, book 400, phone 400, backpack 400, shoe 382. Pen is far short; spec section 5 says top up pen from Open Images "Pen" test split. Controller ruling: fresh implementer (opus) adds that top-up to tools/build_backup_dataset.py (plan deviation mandated by the spec), finishes the task.
Task 4: re-dispatched fresh implementer (BASE 53118cd)
Task 4: implementer DONE_WITH_CONCERNS at 08c6bb6: pen 209 (Open Images test split had 156 usable); index-named crops + plan's min-side re-run created duplicates (23 pen, 89 shoe) that can straddle the split; thin pens lose shape under centre-crop; OI pens are clean product shots. Controller ruling: pre-review fix round — stable names, per-class MIN_SIDE map, square padding, OI test+validation splits, md5 dupe check, wipe+rebuild from cache.
Task 4: pre-review fix round dispatched (resume a66a025615f2f1b74; from 08c6bb6)
Task 4: pre-review fix landed e35036a: stable names, per-class MIN_SIDE, square padding, OI test+validation, md5 check, near-copy skip (implementer's addition after seeing 3 dupes from double-labelled boxes). Final counts bottle 400, pen 244 (94 LVIS + 111 OI test + 39 OI val), book 400, phone 400, backpack 400, shoe 382, duplicates: 0.
Task 4: task review dispatched (BASE 53118cd, HEAD e35036a)
Task 4: review (opus) verified data: 0 md5 dupes, names/filters hold, OI parsing right. 1 Important: square-padded twin crops from one photo (36 pairs at overlap 0.5-0.7) can straddle the per-file backup split -> inflated test score. Ruling: add Loaded.groups + split_by_group (by source photo) in backup.py now; Task 6 assemble must use it when groups present.
Task 4: controller rulings on ⚠️ items: per-(photo,class) 2-crop cap accepted (brief's code; by-photo split removes the leak); 160 px is an upper bound only (thumbnail never enlarges; loader resizes) accepted; pen 244 / shoe 382 accepted (spec allows pen shortfall; counts are what the public sources hold); OI Rotation field check delegated to fix round 2.
Task 4: minor (deferred): LVIS stage has no bad-photo guard (one dead COCO photo blocks every run); save_crop writes straight to final name (no .part); md5 check misses same-picture pairs from different COCO ids (book coco_102834_0/coco_519398_0, backpack coco_188044_0/coco_558794_0); square_box/overlap/open_images_boxes/numbering have no repo tests; main() LVIS pick inline vs OI in a function; `dupes += digest in seen` readability; re-runs recount OI near-copies
Task 4: fix round 1/5 (post-review) dispatched (resume a66a025615f2f1b74; FIX_BASE e35036a)
Task 4: fix round 1/5 (2 addressed, 0 open — by-photo split added; OI Rotation checked: 5 photos non-zero but boxes match stored pixels, no rotate; commits e35036a..f93cc68)
Task 4: minor (deferred): split_by_group can leave train empty / overshoot test frac for tiny or lopsided inputs (not reachable on the real set: 1,717 photos); group_key is name-only so old-style names collide across class folders (only makes groups coarser)
Task 4: NOTE for Task 6: dataset.assemble must call backup.split_by_group(backup.groups, seed) when groups is not None, else split_indices. Real set: 2,226 rows, per-file split straddles 129 photos; by-photo split straddles 0 (1892 train / 334 test). split_by_group returns ascending indices (trainer shuffles anyway).
Task 4: NOTE: never add rotate-before-crop for Open Images; boxes match stored pixels.
Task 4: complete (commits 53118cd..f93cc68, review clean after 1 pre-review fix and 1 fix round). Data: bottle 400, pen 244, book 400, phone 400, backpack 400, shoe 382; duplicates 0.
Task 5: dispatched (BASE f93cc68)
Task 5: review found 3 Important (all plan-mandated, all reproduced): MPO phone JPEGs rejected; add() unsafe under overlapping calls (file overwrite, caps exceeded); DecompressionBombError/ValueError escape as raw exceptions. Ruling: fix all three (accept MPO, lock, catch bombs -> "too big").
Task 5: controller rulings on ⚠️: odd device count -> extra device goes to train (len//2 to test), accepted; LiveStore is in-memory only so a server restart forgets photos and could overwrite dev_x_0.jpg -> deferred minor (demo restart is a failure case anyway; note in docs)
Task 5: minor (deferred): alpha discarded by convert("RGB") (transparent pixels show their hidden colour); case-insensitive FS could merge ABCD_x and abcd_x file names; Photo.label uses config.CLASSES not the store's classes arg; non-string device gives TypeError; test gaps (PNG accept, rejects leave no file, split boundary/odd count, thumbnail orientation)
Task 5: fix round 1/5 dispatched (resume a50eef3159eec25be; FIX_BASE f0bd66c)
Task 5: fix round 1/5 (3 addressed, 0 open — MPO accepted, lock around add(), bomb/ValueError/struct.error -> LiveError; commits f0bd66c..41bef2e)
Task 5: minor (deferred): ValueError/struct.error clauses untested in repo; readers of _photos take no lock (single list ops, harmless); 1x-2x MAX_IMAGE_PIXELS images only warn
Task 5: complete (commits f93cc68..41bef2e, review clean after 1 fix round)
Task 6: dispatched (BASE 41bef2e)
Task 6: review approved; reviewer verified zoom direction, mirror (no shear), erase prob 0.289/area 0.04-0.23, CUDA determinism, preview_pairs == augment (0 px diff). Controller ruling on ⚠️: crop "scale 0.6-1.0" is side length (area 0.36-1.0), accepted after viewing the sheet.
Task 6: minor (deferred): comment should say scale is side length and that rotation can sample slightly outside (reflection pads); brightness is additive (+-0.3) and stacks with contrast so dark photos can go near-black (first knob if dark phone photos do badly); to_float div_ mutates a float input (outside uint8 contract); test gaps (one-sided live tuples, empty live, device arg, live label positions); row-order contract not in docstrings; no dtype check on live tensors; "Module 4" jargon in docstring
Task 6: complete (commits 41bef2e..6ade1a0, review clean)
Task 7: dispatched (BASE 6ade1a0)
Task 7: NOTE for Task 14: GPU lap times (3-lap smoke, lap 3): textbook 567, stairs 2113, elevator 2240, featherweight 537, spotlight 903, trifocals 658, influencer 496 ms -> one full lap ~7.5 s > 4 s floor, so a 20-lap race is ~2.5 min; accuracies at lap 3: 0.20-0.35 (test set 335 rows, phone 67 so majority guess = 0.20). Consider trimming STAIRS/ELEVATOR widths in balancing if the race rule allows.
Task 7: review (opus) approved, 0 Critical/Important. Controller rulings on ⚠️: (1) fairness -> every horse gets identical augmented pixels (drop `+ i`), fix round 1; (2) 4 s floor covers the whole lap of all horses, sum ~7.5 s on this GPU -> race ~150 s; Task 14 must budget the sum (~570 ms/horse) if the 90 s target matters; (3) GPU runs not bit-repeatable (CPU is) -> accepted, docs say so; (4) run() blocks during the floor -> Task 11 runs it in a thread (plan does); (5) train_acc is on augmented batches in train mode, can sit below test early -> docs explain; caller/verdict thresholds unaffected.
Task 7: minor (deferred): pacing test only lower-bounds; no test for shared shuffle order or uneven last batch; device mismatch fails late; Race build reseeds the global RNG; first horse pays GPU warm-up in lap 1 ms (TEXTBOOK 1491 vs 571 ms); run() twice would double events
Task 7: fix round 1/5 dispatched (resume a735826bc2252d07f; FIX_BASE 9860529)
Task 7: fix round 1/5 (1 addressed, 0 open — per-lap augmentation seed, docstrings, position-independence test; commits 9860529..d3c59a7)
Task 7: complete (commits 6ade1a0..d3c59a7, review clean after 1 fix round)
Task 8: dispatched (BASE d3c59a7)
Task 8: review found 2 Important (plan-mandated test gaps): champion accuracy tie-break untested; replay speed division untested. Ruling: add both tests (no production change).
Task 8: controller rulings on ⚠️: form_guide must be fed practice files only -> Task 14 writes practice-*.json and app.py reads data/form_guide.json written by the tool (not a glob), accepted; same device in both winner lists when champion == race winner -> UI shows both lines as the spec says, accepted.
Task 8: minor (deferred): race_winner([]) raises bare IndexError (Task 11 must guard); write_race not atomic; replay(speed<=0) would hang (max(speed,1e-6)); replay drifts by consumer time; LapEvent(**e) brittle to schema change; form_guide iterates a set (dict order varies run to run); two races in one second share a file name; lap_reached uses ">= final" (overshoot-then-settle gets the earlier lap; defensible reading, comment slightly imprecise)
Task 8: fix round 1/5 dispatched (resume aa88ad3cf7cd378ae; FIX_BASE 5c99660)
Task 8: fix round 1/5 (2 addressed, 0 open — tie-break and replay-speed tests, both shown to bite; commits 5c99660..22d2242)
Task 8: minor (deferred): replay speed test bound 0.05 s is tight if Windows falls back to the 15.6 ms timer (loosen to 0.1 if it ever flakes)
Task 8: complete (commits d3c59a7..22d2242, review clean after 1 fix round)
Task 9: dispatched (BASE 22d2242)
Task 9: review (opus) found 5 Important (all plan-mandated): float-boundary misfires at exactly 8/20/2/10 points; STAIRS/ELEVATOR sentences false when ELEVATOR slower; sprint ms rounded to whole ms ("1 ms against 1 ms"); INFLUENCER "not enough data" unconditional; "Seven brains" hard-coded. Ruling (spec honesty rule governs): fix all five + CLEAR line states the margin.
Task 9: controller rulings on ⚠️: (b) trainer always emits one event per horse per lap (lock-step), so lap-wide lines fire; (c) verdicts before the Sprint show "8th in the Sprint (no time)" -> app.py only enters VERDICT after SPRINT; the L-key laptop fallback always fills sprint_ms; acceptable, note in ops docs; (d) FEATHERWEIGHT MACs (12.0M) < TEXTBOOK (53.7M) at current widths; Task 14 tagline test re-pins after balancing.
Task 9: minor (deferred): repeated "wall" lines every 3 laps on a flat horse (~13/race in synthetic runs); "0 points" on near-ties and dead heats unreported; memorised verdict says "percent" for a points gap; SPOTLIGHT "-0 points"/"1 points"; FEATHERWEIGHT claim does not check macs < textbook macs; missing stats render "0 parameters"; pct and the 0.20 threshold duplicated across caller/verdict; "lead" keyword not unique (HALFWAY/FINAL_LAP contain "leads"); thin boundary tests; no docstrings on Caller/compute_facts/verdicts
Task 9: fix round 1/5 dispatched (resume a23a88aae0bccb8ae; FIX_BASE 82aa8a7)
Task 9: fix round 1/5 (5+1 addressed, 0 open; commits 82aa8a7..4607903; 12 tests added)
Task 9: minor (deferred): INFLUENCER "kept up with the convolution horses" compares only to TEXTBOOK (say "kept up with TEXTBOOK"); one-horse race prints "1 brains"; points() rounds so a 1.6-point photo finish says "by just 2 points" while exactly 2 says "2 points back"; ms_text equal-looking at shown precision (10.2 vs 10.4)
Task 9: complete (commits 22d2242..4607903, review clean after 1 fix round)
Task 10: dispatched (BASE 4607903)
Task 10: implementer DONE_WITH_CONCERNS at 4ec2b34: classic exporter (dynamo=False) works on torch 2.11 with hidden deprecation warnings; IR 8, opset 17, max diff 2.09e-7; review dispatched
Task 10: review found 1 Important (plan-mandated test gap): ExportMismatch path untested. Ruling: add the test (no production change).
Task 10: controller rulings on ⚠️: onnxruntime-web operator coverage (LayerNormalization/Erf/Softmax for INFLUENCER) is checked in Task 13's browser walk; torch is unpinned (requirements.txt deliberately omits it) -> final report notes the legacy exporter deprecation; server maps model_paths only on export success, so a bad file left on disk is harmless (Minor).
Task 10: minor (deferred): bad .onnx stays on disk after ExportMismatch; `except TypeError` fallback wider than its comment (dead on torch 2.11); test name promises device check but asserts only mode; no function docstrings; bench_cpu(n<1) raises StatisticsError; absolute tolerance with unseeded input
Task 10: fix round 1/5 dispatched (resume aea31264b28135e2c; FIX_BASE 4ec2b34)
Task 10: fix round 1/5 (1 addressed, 0 open — ExportMismatch test, shown to bite; commits 4ec2b34..9ae0d06)
Task 10: complete (commits 4607903..9ae0d06, review clean after 1 fix round). Export: classic exporter, IR 8, opset 17, names image/logits, dynamic batch.
Task 11: dispatched (BASE 9ae0d06)
Task 11: implementer DONE_WITH_CONCERNS at 076e3e6 (lifespan + Sprint guards applied). Concerns: worker threads (load_data, run_events) swallow nothing and send no `error` message -> show stalls silently; StarletteDeprecationWarning (httpx2 missing) on every run. Ruling: pre-review fix round for both (try/except + broadcast error; install httpx2 or narrow filter).
Task 11: pre-review fix round dispatched (resume a0efdec4d59e5fcbb; from 076e3e6)
Task 11: automated security review of 076e3e6 flagged: HIGH path traversal in /api/replay (body path used as-is, echoed back); MEDIUM operator endpoints and websocket roles have no protection (any LAN phone can advance the show, claim role=team, post fake sprint results); MEDIUM bets keyed by client-supplied device id. Controller ruling: harden in fix round 2 (after the pending thread-error fix): replay accepts a bare .json name inside data/races only; operator endpoints (advance, gate/close, sprint/laptop, replay, later veteran) are localhost-only via Settings.local_only_ops (tests pass False or a local client tuple); team role + /api/sprint-result need Settings.team_token (random per run), exposed via local-only GET /api/team-url; bets stay device-keyed (random token, no personal data; spec design) -> acknowledged, no change. Tasks 12/13 must carry the token: phone.js sends it, control page shows the team URL.
Task 11: pre-review fix landed a6d431e: load_data/run_events wrapped, `error` message broadcast, send_previews guarded for empty batch, httpx2 installed (2.13.1) and added to requirements; 101 passed, no warnings. Open (deferred): after a load failure the show cannot recover without restart and Space shows "still loading"; full-run test would hang under -W error (torch exporter deprecation).
Task 11: fix round 2 (security hardening) dispatched + addendum (error broadcasts to stadium/control only) (resume a0efdec4d59e5fcbb; from a6d431e)
Task 11: fix round 2 landed 3085da3 (replay by name only; local_only_ops; team_token; /api/team-url; error msgs to stadium/control). Deferred: operator pages must be opened via localhost (LAN address would 403; ::ffff:127.0.0.1 not in set); any phone may claim role=stadium/control (receives operator messages, no actions); load failure needs restart. Task review dispatched (BASE 9ae0d06, HEAD 3085da3).
Task 11: review (opus) spec-compliant; 3 Important (plan-mandated live-show gaps): no race-running guard (Space during export / double Replay); Sprint results accepted after decided (late phone times overwrite laptop fallback, champion flips after verdict); reconnecting sockets miss one-off messages (phones miss bet results, team phone misses sprint_request). Ruling: fix all three + NaN-safe lap JSON + bad `ms` -> 400.
Task 11: controller rulings on ⚠️: pages must render nick/class/device as text (carry to Tasks 12/13); previews now stadium/control only; NaN handled in round 3; Space auto-repeat guard + open operator pages at localhost + phone token + replay {"name"} carried to Tasks 12/13; LiveStore validates device/class before file names (Task 5 reviewed); data/form_guide.json is written by Task 14's tool.
Task 11: minor (deferred): finish_verdict writes a race file even with no events (replay-latest could play an empty race; replayed races saved again -> form guide double counts); ship advertises champion.onnx even when none; partial export leaves model_paths partly filled; advance changes act before act work (a 500 there skips the act); broadcast_threadsafe discards the future (lost exceptions, un-awaited coroutine after loop close; state_payload built on worker thread); Hub.broadcast serial (one stuck phone delays all); operator_allowed passes when request.client is None; bet_results leaks all device ids to all phones; "still loading"/"Run the race first." misleading after failures; /api/team-url ignores public_url; test wait loops unbounded; no SPRINT/VERDICT/SHIP test (Task 15 adds); 403 test covers advance+team-url only; unused LapEvent import; create_app ~300 lines
Task 11: fix round 3 dispatched (resume a0efdec4d59e5fcbb; FIX_BASE 3085da3)
Task 11: fix round 3/5 (3+2 addressed, 0 open; commits 3085da3..3faca44)
Task 11: minor (deferred): /api/sprint/laptop has no act check (L in DERBY decides the Sprint early, then phone times refused); replay forgets sprint_request and has no act check (replay in HORSES then Space -> two race threads); finish_sprint locks before scoring (a raise would leave it locked); error messages not remembered for reconnects; forget_race keeps old champion/sprint_standings in /api/state after a replay; microsecond window between clearing race_running and storing race_done
Task 11: complete (commits 9ae0d06..3faca44, 4 commits, review clean after pre-review fix + 2 hardening/fix rounds + 1 review fix round). Tests: 107 passed, no warnings.
Task 12: dispatched (BASE 3faca44)
Task 12: implementer DONE_WITH_CONCERNS at b27fe1f; browser walk done at 1920x1080 with visual fixes; controller viewed DERBY and HORSES screenshots: legible. Review dispatched (BASE 3faca44).
Task 12: HANDOFF — session ended before the task review ran. Next session: run the Task 12 task review first (see docs/superpowers/handoffs/2026-10-12-neural-derby-handoff.md), then Tasks 13-16.
```
