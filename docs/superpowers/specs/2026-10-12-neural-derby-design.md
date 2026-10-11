# Neural Derby: design spec

Date: 2026-10-12
Course: UCS668 Edge AI and Robotics, Thapar. Group Project 1 (10 marks). Demo 12 to 16 October 2026 in the lab slot.
Team: Ranvir, Tejas, Harshit.
Brief from the course: "Choose any concept in this course and demonstrate it in a gamified or creative manner."

## 1. The one-paragraph pitch

Everyone in the room photographs an object on their desk with their phone. Seven small neural networks, each built from a different block ma'am taught in Module 3, race to learn those photos on the laptop GPU while the room bets on a horse from their phones. One lap is one training epoch. A race-caller voice reacts to real events. The second trophy is thinking speed on a real Android phone. Every horse's card then explains, from measured numbers, why it won or lost. The champion is exported and sent back to every phone, where it recognises the room's objects on-device. The whole course pipeline (task, data, augmentation, network design, training, evaluation, speed-versus-accuracy, edge deployment) runs live in about ten minutes.

How you tell a 15-year-old: "Everyone takes photos of their stuff. Seven tiny AI brains race to learn what's in the photos. You bet on one. The winner gets sent to your phone and can see your stuff."

## 2. Who uses it and what they see

Three kinds of people touch the system.

**The room (about 30 people, each with a phone).** They scan a QR code, pick an object type, take a photo, place a bet, watch the race on the projector, see whether they won, and finally run the champion on their own phone. They never install anything. Every action is one or two taps. If their phone cannot reach the laptop, they still see the show and bet by raising a hand.

**The operator (Ranvir, at the laptop).** Drives the show with one key (space advances to the next act), watches a small control page for health (GPU, connected phones, photos received), and can trigger fallbacks (replay a recorded race, skip the Sprint, use laptop timing).

**Ma'am and the graders.** They see the projector. They will ask "why did that horse win", "how many parameters does it have", and "what happens if you remove the shortcut". Every card and verdict is built to answer those with numbers.

Where people get stuck, and the fix: the QR URL is long, so the page shows a 4-character room code too; photos arrive sideways from phones, so EXIF rotation is applied on upload; people photograph the wrong thing, so uploads show a thumbnail with the chosen label and a "retake" button; the race is too fast to follow, so laps are paced to about four seconds each.

## 3. The run of show (six acts)

Each act is a state in a state machine. Space advances. Every act has a fallback that keeps the show moving.

| Act | Name | Projector | Phone | Time |
|---|---|---|---|---|
| 1 | FILL THE PADDOCK | QR code, room code, a wall of photo tiles growing live, counter "47 / 120 photos" | 6 big object buttons, camera button, thumbnail, retake | 2 min |
| 2 | MULTIPLY | The wall animates: real augmented versions of real uploaded photos spin, darken, blur, flip, and multiply. Caption: "600 photos just became 6,000." | "Augmenting your photos..." | 30 s |
| 3 | MEET THE HORSES | Seven fighter-select cards: pixel sprite, name, tagline, block diagram, parameters, multiplies, depth, stat bars, odds from the form guide. Phones bet. | Bet screen: seven cards, tap one, lock in | 2 min |
| 4 | THE DERBY | Seven lanes, 20 laps. Position equals accuracy on held-out photos. Tote board with live standings. Commentary ticker plus voice. Checkered flag, photo finish, podium. | Live mini standings, "your horse is 3rd" | 90 s |
| 5 | THE SPRINT | Each horse timed on the team phone. Bar chart of milliseconds per photo. Champion announced from combined rank. Winning bettors scroll. | "You won!" or "Better luck next race" | 45 s |
| 6 | WHY, THEN SHIP IT | Cards flip to verdicts built from measured facts. Then the champion is pushed to phones. Epilogue: THE VETERAN (pretrained MobileNetV3-small) runs one lap, beats everyone's accuracy, then is shown to be far slower on the phone. | "Point at your desk" camera button, top-3 prediction on-device | 3 min |

Acts 1 and 3 have a hard gate the operator closes. Act 4 cannot start until the trainer reports the dataset is loaded on the GPU.

## 4. The horses

All horses share the same jockey (the Adam optimiser with the same learning-rate schedule), the same food (the same training photos in the same order per lap), the same image size (96 by 96), and the same head (global average pool, then one linear layer to 6 classes). Only the body differs. That is the point: the race isolates architecture.

| Horse | Ma'am's block | Body (initial widths, tuned in prep) | Tagline on the card | Teaches |
|---|---|---|---|---|
| TEXTBOOK | plain conv, maxpool, ReLU | 3 blocks of conv 3x3, ReLU, maxpool; widths 24, 48, 96 | "Conv, pool, repeat. The recipe every network starts from." | the basic recipe, parameter counting |
| STAIRS | deep conv, no shortcuts | 10 conv 3x3 layers with BatchNorm and ReLU, widths 24 to 96, pool every 2 or 3 layers | "Ten floors deep and the learning signal has to climb every step. Tires early, strong finish." | depth without skip connections learns slowly |
| ELEVATOR | residual block | identical to STAIRS plus a skip connection every two layers | "Same building, but with shortcuts. The signal rides straight through." | why skip connections exist |
| FEATHERWEIGHT | depthwise separable conv | TEXTBOOK's shape, but each conv after the stem is depthwise 3x3 then pointwise 1x1 | "One cheap step per colour channel, then mix. Eight times fewer multiplications." | why edge models use depthwise conv |
| SPOTLIGHT | channel attention, spatial attention | TEXTBOOK plus her fully-convolutional channel attention and a 7x7 spatial attention after blocks 2 and 3 | "Turns up the volume on useful features and points a light at where the object is." | attention as a volume knob and a where-mask |
| TRIFOCALS | inception block | stem conv, then two inception blocks (1x1, 3x3, 5x5 branches, concatenated), pool between | "Three lens sizes at once. Sees the whole shoe and the shoelace." | multiscale features |
| THE INFLUENCER (switchable) | transformer encoder block | 8x8 patches (144 tokens), width 64, 3 encoder layers, 4 heads, mean pool | "The architecture behind ChatGPT. Famous. Needs a million photos to perform." | transformers are data-hungry; a predictable but funny loser |

STAIRS versus ELEVATOR is the contrast the race is built around. Their depth is a single tunable; prep picks the depth where ELEVATOR visibly converges earlier but STAIRS still finishes respectably. If practice races show no gap at the chosen depth, the verdict says so honestly and prep increases the depth.

The Veteran is not a horse. It is an epilogue: a pretrained torchvision MobileNetV3-small with a new head, fine-tuned for one lap. It exists to show transfer learning (Module 2) and the edge trade-off (Module 7) in thirty seconds.

**Balancing rule.** During prep, run at least 20 practice races on the backup set. Adjust widths (never the block type) until no horse wins the Race more than about a third of the time and the Sprint order is still a real speed order. Record every practice race; the win frequencies become the betting odds on the cards.

## 5. Data

**Classes (6):** bottle, pen, book, phone, backpack, shoe. "Book" includes paper notebooks. Chosen because they are on every desk in the room and all six exist in the backup source with at least 1,500 labelled objects each.

**Backup set (always present).** Built once by a prep tool from LVIS v1 annotations over COCO 2017 images. Mapping: bottle = water_bottle + bottle; pen = pen + pencil; book = book + notebook; phone = cellular_telephone; backpack = backpack; shoe = shoe. Keep boxes with shorter side at least 64 px, at most 2 crops per source image, 400 crops per class, saved at 160 px on the long side under `data/backup/<class>/`. Licence: LVIS annotations CC BY 4.0, images under Flickr terms, research use. If pen falls short after filtering, top up from Open Images "Pen" (test split, no account needed).

**Live photos.** Uploaded from phones during Act 1. Rules: JPEG or PNG only, at most 5 MB, EXIF rotation applied, resized to 160 px long side, stored under `data/live/<class>/<device>_<n>.jpg`. Caps: 3 per device, 120 total, configurable. The device is identified by a random token the phone page stores, not by anything personal. No faces are requested; the buttons name objects.

**Train and test split.** Backup: 85 percent train, 15 percent test, fixed by seed. Live: split by device, half of devices to train and half to test, so a person's three photos never leak across the split. If fewer than 12 live photos arrive, all go to test. The race position is accuracy on the combined test set. The verdict reports two numbers separately: accuracy on internet photos and accuracy on the room's photos. That second number is the honest one and usually lower.

**Augmentation (Module 4).** Done on the GPU per batch with torch operations so training stays fast: random resized crop (scale 0.6 to 1.0), horizontal flip, brightness and contrast jitter (plus or minus 30 percent), small rotation (plus or minus 15 degrees), gaussian noise (sigma 0.02), random erasing (one rectangle, probability 0.3). Act 2 shows real outputs of these same functions on real uploaded photos; nothing in the visual is faked.

**Everything is loaded to GPU memory once** as uint8 tensors (about 2,600 images at 96 px is roughly 70 MB), so a lap is pure GPU work.

## 6. Race rules and scoring

- 20 laps. One lap is one full pass over the training set for every horse, in lock-step: lap k for all horses completes before lap k+1 starts. Batch size 64. Same shuffle order for all horses within a lap.
- After each horse's lap: evaluate on the test set, emit a lap event with train accuracy, test accuracy (overall, internet-only, room-only), loss, and milliseconds taken.
- Pacing: the trainer will not start lap k+1 earlier than 4 seconds after lap k started, so the race lasts about 90 seconds even if the GPU is faster. If the GPU is slower than that, laps take their real time.
- **Race standings:** final test accuracy, descending. Tie-break: the horse that first reached its final accuracy.
- **Sprint standings:** median of 30 single-image inferences on the team phone in the browser, in milliseconds, ascending. Fallback: laptop CPU timing with onnxruntime. The display always says which device was used.
- **Champion:** lowest sum of Race rank and Sprint rank. Tie-break: final test accuracy.
- **Bets:** one horse per device, locked when the Act 3 gate closes. Winners are devices that picked the champion. Devices that picked the Race winner get a separate shout-out.
- **Form guide odds:** from recorded practice races, shown as "wins 6 of 20" and a fraction, so no one is misled.
- Every race is recorded to `data/races/<timestamp>.json` with all events, standings, and verdicts. A recorded race can be replayed into the stadium page at the same pacing.

## 7. Commentary (the caller)

Event rules evaluated after every lap event. Each rule has several phrasings picked at random. Lines go to the ticker and to the browser's speech synthesis (offline Windows voices, no internet). Rules:

- start: "And they're off."
- first leader: "X takes the early lead at N percent."
- lead change: "Y overtakes X."
- surge: a horse gains 8 points or more in one lap.
- stall: a horse moves less than 1 point over three laps: "X has hit a wall at N percent."
- memorising: train accuracy exceeds test accuracy by 20 points or more: "X is memorising the photos instead of learning."
- halfway: lap 10 standings summary.
- final lap: "Last lap. X by a nose over Y."
- finish: winner, margin, and whether it was a photo finish (margin under 2 points).

Every line is generated from the actual numbers in the event. There is no scripted outcome.

## 8. Verdicts (the "why")

After the Sprint, each horse gets a verdict card built from facts.

Facts computed per horse: parameters, multiply-accumulates at 96 px, depth (number of weight layers), final test accuracy (overall, internet, room), the lap at which it first reached 90 percent of its own final accuracy (convergence lap), the largest train-minus-test gap, Sprint milliseconds, Race rank, Sprint rank, Champion rank.

Verdict sentences are produced by rules that only state what the numbers support. Examples:

- If STAIRS's convergence lap is later than ELEVATOR's by 3 or more: "STAIRS reached its stride on lap K; ELEVATOR got there on lap J. Same depth, one shortcut." Otherwise: "At this depth the shortcuts did not change the pace. Deeper buildings need elevators more."
- For FEATHERWEIGHT: "M multiplies against TEXTBOOK's N. That is why it ran the Sprint in T ms."
- For any horse with a gap of 20 points or more: "It memorised: A percent on photos it had seen, B percent on new ones."
- For every horse: internet versus room accuracy, with one sentence on the difference.
- For THE INFLUENCER: "A percent with N photos. Transformers have no built-in idea that nearby pixels belong together; they must learn it from data, and there was not enough."

Each card also shows the per-layer table ma'am asks for in exams: layer, output shape, parameters.

## 9. Phones, network, deployment

**Serving.** One Python process serves both pages. Plain HTTP on port 8000 for the room. A second HTTPS listener on port 8443 with a self-signed certificate for the team phone only, because browsers allow live camera access only on HTTPS. Crowd phones never see a certificate warning.

**Camera on crowd phones.** The photo button is an HTML file input with camera capture, which opens the phone's camera app and works over plain HTTP on Android and iPhone. Live video is only on the team phone.

**Network options, tested in advance, in this order.** (1) The college Wi-Fi, if phones can reach the laptop (many campus networks block device-to-device traffic; test by opening the laptop's address from a phone). (2) An Android phone's hotspot (10 to 32 devices depending on the phone) with the laptop joined to it. (3) A borrowed travel router. The Windows built-in hotspot allows only 8 devices and is the last resort. The QR shows the chosen address; nothing on the pages needs internet: fonts, scripts, and the inference runtime are all served from the laptop.

**On-device inference.** The champion is exported to ONNX (opset 17) and served to phones. The phone page runs it with onnxruntime-web (WebAssembly backend) and does the preprocessing in JavaScript (resize to 96, normalise). Crowd phones: snapshot mode (take a photo, classify, show top-3 with bars). Team phone: live video at a few frames per second with the prediction overlaid. The Sprint timing uses the same runtime on the team phone.

## 10. User interface direction

Committed style: **a 1980s arcade cabinet.** Not a dashboard, not a slide.

- Background near-black blue. Seven saturated lane colours. Amber text on the tote board. Subtle CRT scanline overlay at low opacity, never on the phone.
- Pixel font (Press Start 2P, Open Font Licence, vendored). Chunky 4 px borders. Big type: every number on the projector is readable from the back row at 1920 by 1080.
- Horses are 16 by 16 pixel sprites with a two-frame gallop, drawn as pixel maps in code (no image assets to lose). Lanes run left to right with accuracy markers at 0, 25, 50, 75, 100.
- Sounds synthesised with WebAudio: starting gun, lap chime, crowd roar on the finish, a small jingle when a card flips. No audio files.
- Cards are fighter-select screens: portrait, name, stat bars (SPEED, DEPTH, WEIGHT, BRAIN), a block diagram drawn as pixel boxes, tagline, odds.
- Blinking "PLACE YOUR BETS" during Act 3, "GATES CLOSED" when locked.
- The phone page shares the palette and font but is thumb-sized and plain: one column, big buttons, no scanlines, works on a 360 px wide screen.
- Forgotten states are designed: empty wall before photos arrive (a "waiting for the first photo" blink), the gate closing, a horse with no Sprint time (shows "DNS" for did not start, with laptop time as fallback), a phone that lost connection (banner, retry), and long nicknames (truncated at 12 characters).

## 11. Software shape

Python 3.12 virtual environment. PyTorch with CUDA, torchvision (Veteran only), FastAPI, uvicorn, onnx, onnxruntime, Pillow, numpy, qrcode, pytest. Frontend is plain HTML, CSS and JavaScript with no build step.

```
neural-derby/
  derby/
    config.py              classes, horses on/off, laps, caps, ports, pacing
    data/
      backup.py            load data/backup into tensors
      live.py              accept uploads, caps, per-device split
      augment.py           GPU augmentations and preview thumbnails
      dataset.py           assemble train/test tensors on the GPU
    horses/
      blocks.py            ma'am's blocks: residual, channel attention, spatial attention, inception, depthwise separable, transformer encoder
      stable.py            the seven horse definitions (name, tagline, builder, colour)
      stats.py             parameters, multiply-accumulates, per-layer shapes
    race/
      trainer.py           lock-step training loop that emits lap events
      scoring.py           race standings, sprint standings, champion, bets
      recorder.py          race JSON record and replay, form guide
    commentary/
      caller.py            lap events to caller lines
      verdict.py           facts to why-sentences
    export/
      onnx_export.py       export a horse, check it matches torch
      bench.py             laptop CPU timing fallback
    server/
      app.py               FastAPI routes, websocket hub, static files
      show.py              the act state machine
  web/
    stadium/               index.html, stadium.css, stadium.js, sprites.js, audio.js
    phone/                 index.html, phone.css, phone.js
    control/               index.html (operator health and fallbacks)
    vendor/                press-start-2p.woff2, onnxruntime-web files
  tools/
    build_backup_dataset.py   LVIS crops
    practice_races.py         run N races headless, write the form guide
    make_cert.py              self-signed certificate for the team phone
  data/                    backup/, live/, races/, models/   (git-ignored except a small sample)
  docs/
    learning/              00-overview, 01-race-engine, 02-the-stable, 03-the-grandstand, qa-drill
    demo-script.md         minute-by-minute, who speaks
    setup.md               from clone to running show
  tests/
```

**Boundaries that matter.** The trainer knows nothing about the web; it yields events. The server turns events into websocket messages. The stadium page only renders messages; it holds no training logic. Horses are built by functions in `stable.py` and the stats module works on any `nn.Module`, so adding a horse is one entry. Commentary and verdict rules are pure functions of events and facts, so they are unit-testable without a GPU.

**Data flow.** Phone upload goes to `live.py` (validate, store), then to a websocket broadcast (a tile appears on the wall). Operator advances to Act 2; the server asks `augment.py` for preview pairs and broadcasts them. Act 4: the server starts the trainer in a background thread; each lap event goes to `scoring.py` (standings), `caller.py` (lines), `recorder.py` (file), and the websocket. Act 5: the server asks the team phone to time each exported model and collects results. Act 6: `verdict.py` runs on the recorded facts; the champion's ONNX file becomes downloadable; phones switch to inference mode.

## 12. Failure handling

- No GPU detected: trainer runs on CPU with a warning on the control page; laps take their real time; the operator can reduce laps in config.
- Fewer than 12 live photos: all live photos go to the test set; the show says "tested on your photos" truthfully.
- Zero live photos or no network: the race runs on the backup set; Acts 1 and 2 use backup samples; betting by show of hands.
- Trainer crash mid-race: the control page offers "replay last practice race"; the stadium replays it at the same pacing.
- Team phone unreachable for the Sprint: laptop CPU timing is used and labelled as such.
- Upload abuse: caps per device and global, size and type checks, corrupt images rejected with a friendly message.
- Phone loses connection: the page shows a banner and retries the websocket every 3 seconds; state is re-sent on reconnect.

## 13. Testing

Unit tests (no GPU needed, CPU tensors):

- Every horse builds, accepts a 1 by 3 by 96 by 96 input, outputs 1 by 6, and its parameter count equals what `stats.py` reports.
- The multiply counter matches a hand-computed value for a small conv and a small linear layer.
- Scoring: race standings with ties, sprint standings, champion rank sums with ties, bet settlement.
- Caller rules fire on synthetic event streams exactly when they should (lead change, stall, surge, memorising, photo finish).
- Verdict rules never claim a slow start unless the convergence laps differ by the threshold; never claim memorising below the gap threshold.
- Live upload: caps, per-device split, EXIF rotation, rejection of corrupt files.
- ONNX export: outputs match torch within 1e-4 on random input for every horse.
- Show state machine: only legal transitions; Act 4 blocked until the dataset is loaded.

Integration: a headless smoke race with synthetic data, 2 laps, 2 horses, asserts that events, standings, a race file, and a verdict are produced.

Manual, before the demo: full rehearsal with three phones on the chosen network; a projector check for legibility from the back of the room; the 20 practice races and the balancing step; timing of each act.

## 14. Roles

- **Ranvir, race engine and show director.** `race/`, `server/`, `web/stadium/`, the state machine, pacing, recording and replay, the run of show. Presents the architecture of the system and the training loop (Modules 2 and 5).
- **Tejas, the stable.** `horses/`, the seven cards, the per-layer tables, `commentary/verdict.py`, the balancing step and form guide. Presents each block and why each horse won or lost (Module 3), and parameter counting on demand.
- **Harshit, the grandstand.** `web/phone/`, `data/live.py`, the network setup, `export/`, the Sprint timing, on-device inference. Presents data collection, augmentation (Module 4), and edge deployment (Module 7).

Each role has a chapter in `docs/learning/` written so a 15-year-old could read it, followed by a question-and-answer drill of the questions ma'am is most likely to ask that person.

## 15. Prep checklist (before the lab slot)

1. Environment: venv, PyTorch with CUDA, confirm `torch.cuda.is_available()` on the RTX 4050.
2. Build the backup dataset with the LVIS tool; eyeball 20 crops per class; top up pen if short.
3. Smoke race on the GPU; measure seconds per lap per horse; set pacing.
4. Run 20 practice races; balance widths and the STAIRS/ELEVATOR depth; write the form guide.
5. Vendor the font and onnxruntime-web; confirm the phone page works with the laptop's internet off.
6. Test the three network options on campus; pick one; print the QR.
7. Generate the self-signed certificate; accept it once on the team phone; confirm live video.
8. Full rehearsal with three phones; time each act; fix anything that needs explaining.
9. Learning docs read by all three; question drill done.

## 16. Out of scope

Training on faces or people. Internet access during the demo. Native phone apps. Streamlit (the course tool is fine for chapters but cannot animate a race or serve 30 phones well). Any horse that is not built from a block in Module 3.
