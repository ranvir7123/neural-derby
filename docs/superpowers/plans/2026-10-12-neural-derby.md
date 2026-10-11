# Neural Derby Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live, projector-and-phones show where seven small networks built from the course's Module 3 blocks race to learn photos the room just took, with betting, spoken commentary, verdicts built from measured numbers, and on-phone deployment of the champion.

**Architecture:** One Python process (FastAPI + uvicorn) serves a stadium page, a phone page, and a control page over plain HTTP (port 8000), plus an HTTPS listener (port 8443) for the team phone. A lock-step trainer on the GPU yields lap events; pure-function scoring, caller, and verdict modules turn events into standings, lines, and explanations; a websocket hub streams them to the pages. The champion is exported to ONNX and run on phones with onnxruntime-web.

**Tech Stack:** Python 3.12 venv, PyTorch (CUDA) + torchvision, FastAPI, uvicorn, python-multipart, httpx (tests), onnx, onnxruntime, Pillow, numpy, qrcode, cryptography (self-signed cert), pytest. Frontend: plain HTML/CSS/JS, Press Start 2P font vendored, onnxruntime-web vendored, WebAudio for sound, browser speech synthesis for the caller.

Spec: `docs/superpowers/specs/2026-10-12-neural-derby-design.md` (approved 2026-10-12). Read it first.

## Global Constraints

- Python 3.12. Project root for code is `neural-derby/` under the git root `C:\dev\study\Edge-Ai`. All commands below run from `neural-derby/` with the venv active unless stated.
- Image size `96x96` for every horse. Classes, in this fixed order: `bottle, pen, book, phone, backpack, shoe`.
- 20 laps, batch 64, Adam for every horse, same shuffle order for all horses within a lap, pacing floor 4.0 s per lap.
- Upload caps: 3 per device, 120 total, 5,000,000 bytes, JPEG or PNG only, EXIF rotation applied, stored at 160 px long side. Live photos split by device, half and half; fewer than 12 live photos means all go to test. Backup split 85/15 by seed.
- Nothing on the web pages may need internet: fonts, scripts, and the inference runtime are served from the laptop.
- Commentary and verdict text are generated only from measured numbers; no scripted outcomes.
- Every horse is one of ma'am's Module 3 blocks: plain conv, residual, channel attention, spatial attention, inception, depthwise separable, transformer encoder.
- Commits: plain-words messages, no attribution lines, no co-author lines (user rule). Commit from the git root with paths prefixed `neural-derby/`.
- Tests run with `pytest -q` from `neural-derby/`. Tests must pass on CPU; GPU is optional for tests.
- Pixel input normalisation everywhere (training, export check, phone JavaScript): `(x / 255 - 0.5) / 0.5` per channel, RGB order.

## File map

```
neural-derby/
  requirements.txt
  pytest.ini
  derby/__init__.py
  derby/config.py                 constants: classes, laps, caps, ports, enabled horses
  derby/data/__init__.py
  derby/data/images.py            load_image(path) -> uint8 tensor 3x96x96
  derby/data/backup.py            load backup folder, deterministic split
  derby/data/live.py              LiveStore: uploads, caps, per-device split
  derby/data/augment.py           GPU augmentation, normalize, preview pairs
  derby/data/dataset.py           RaceData assembly
  derby/horses/__init__.py
  derby/horses/blocks.py          ma'am's blocks
  derby/horses/stable.py          Horse dataclass, STABLE, builders
  derby/horses/stats.py           params, MACs, depth, layer table
  derby/horses/veteran.py         pretrained MobileNetV3-small epilogue
  derby/race/__init__.py
  derby/race/trainer.py           Race: lock-step training, LapEvent
  derby/race/scoring.py           standings, champion, bets
  derby/race/recorder.py          race JSON write/read/replay, form guide
  derby/commentary/__init__.py
  derby/commentary/caller.py      Caller rules
  derby/commentary/verdict.py     Facts and verdict rules
  derby/export/__init__.py
  derby/export/onnx_export.py     export + match check
  derby/export/bench.py           CPU timing fallback
  derby/server/__init__.py
  derby/server/show.py            Act enum and Show state machine
  derby/server/hub.py             websocket Hub
  derby/server/app.py             FastAPI app factory and routes
  derby/server/main.py            runs HTTP and HTTPS listeners
  web/stadium/index.html, stadium.css, stadium.js, sprites.js, audio.js
  web/phone/index.html, phone.css, phone.js
  web/control/index.html
  web/vendor/PressStart2P-Regular.ttf, onnxruntime-web/ort.min.js (+ wasm files)
  tools/build_backup_dataset.py   LVIS crops
  tools/practice_races.py         N headless races, form guide
  tools/make_cert.py              self-signed certificate
  tools/vendor_assets.py          download font and onnxruntime-web once
  data/backup/<class>/*.jpg       git-ignored
  data/live/<class>/*.jpg         git-ignored
  data/races/*.json               git-ignored
  data/models/*.onnx              git-ignored
  data/form_guide.json            committed after prep
  docs/setup.md, docs/demo-script.md, docs/learning/*.md
  tests/*.py
```

---

### Task 1: Scaffold, environment, config

**Files:**
- Create: `neural-derby/requirements.txt`, `neural-derby/pytest.ini`, `neural-derby/derby/__init__.py`, `neural-derby/derby/config.py`, `neural-derby/tests/__init__.py`, `neural-derby/tests/test_config.py`, `neural-derby/docs/setup.md`

**Interfaces:**
- Produces: `config.CLASSES: list[str]`, `config.IMG = 96`, `config.LAPS = 20`, `config.BATCH = 64`, `config.LAP_FLOOR_S = 4.0`, `config.LR = 2e-3`, `config.CAPS: dict`, `config.MIN_LIVE_FOR_SPLIT = 12`, `config.PORTS: dict`, `config.HORSES_ENABLED: dict[str, bool]`, `config.DATA_DIR: Path`, `config.WEB_DIR: Path`.

- [ ] **Step 1: Create the environment**

From `C:\dev\study\Edge-Ai`:

```powershell
mkdir neural-derby; cd neural-derby
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected last line: a version string, `True`, `NVIDIA GeForce RTX 4050 Laptop GPU`. If the cu128 index has no wheel for this Python, use `cu126`. If `False`, stop and fix the driver before continuing; nothing else in this plan depends on CPU-only speed being acceptable.

- [ ] **Step 2: Write requirements.txt and pytest.ini**

`neural-derby/requirements.txt` (torch is installed separately above and deliberately not listed):

```
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
httpx>=0.27
onnx>=1.16
onnxruntime>=1.19
pillow>=10.4
numpy>=2.0
qrcode[pil]>=7.4
cryptography>=43
pytest>=8.3
```

`neural-derby/pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -q
filterwarnings = ignore::DeprecationWarning
```

Run: `pip install -r requirements.txt`

- [ ] **Step 3: Write the failing config test**

`neural-derby/tests/__init__.py`: empty file.

`neural-derby/tests/test_config.py`:

```python
from derby import config


def test_classes_are_the_six_in_fixed_order():
    assert config.CLASSES == ["bottle", "pen", "book", "phone", "backpack", "shoe"]


def test_race_constants():
    assert config.IMG == 96
    assert config.LAPS == 20
    assert config.BATCH == 64
    assert config.LAP_FLOOR_S == 4.0


def test_caps():
    assert config.CAPS == {"per_device": 3, "total": 120, "max_bytes": 5_000_000}
    assert config.MIN_LIVE_FOR_SPLIT == 12


def test_seven_horses_enabled_by_default():
    assert list(config.HORSES_ENABLED) == [
        "textbook", "stairs", "elevator", "featherweight", "spotlight", "trifocals", "influencer",
    ]
    assert all(config.HORSES_ENABLED.values())


def test_dirs_exist():
    assert config.DATA_DIR.name == "data"
    assert config.WEB_DIR.name == "web"
```

- [ ] **Step 4: Run it to see it fail**

Run: `pytest tests/test_config.py`
Expected: `ModuleNotFoundError: No module named 'derby'`

- [ ] **Step 5: Write config.py**

`neural-derby/derby/__init__.py`: empty file.

`neural-derby/derby/config.py`:

```python
"""Every knob the show has, in one place. Plain constants, no logic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"

CLASSES = ["bottle", "pen", "book", "phone", "backpack", "shoe"]
IMG = 96

LAPS = 20
BATCH = 64
LAP_FLOOR_S = 4.0
LR = 2e-3
SEED = 1234

CAPS = {"per_device": 3, "total": 120, "max_bytes": 5_000_000}
MIN_LIVE_FOR_SPLIT = 12
BACKUP_TEST_FRAC = 0.15

PORTS = {"http": 8000, "https": 8443}

HORSES_ENABLED = {
    "textbook": True,
    "stairs": True,
    "elevator": True,
    "featherweight": True,
    "spotlight": True,
    "trifocals": True,
    "influencer": True,
}
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_config.py`
Expected: `5 passed`

- [ ] **Step 7: Write docs/setup.md**

`neural-derby/docs/setup.md`:

```markdown
# Setup (from clone to a running show)

1. Install Python 3.12 (the course version).
2. From `neural-derby/`:
   - `py -3.12 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
   - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   - `pip install -r requirements.txt`
   - `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.
3. One-time downloads (internet needed, done before the demo):
   - `python tools/vendor_assets.py` (font and onnxruntime-web into `web/vendor/`)
   - `python tools/build_backup_dataset.py` (about 0.5 GB of photos into `data/backup/`)
4. Tests: `pytest`.
5. Run the show: `python -m derby.server.main`. Open `http://localhost:8000/stadium/` on the projector, `http://localhost:8000/control/` on a second window, and the QR on the stadium page on phones.
```

- [ ] **Step 8: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/requirements.txt neural-derby/pytest.ini neural-derby/derby neural-derby/tests neural-derby/docs/setup.md
git commit -m "Scaffold the Neural Derby project with config and setup notes"
```

---

### Task 2: Ma'am's blocks

**Files:**
- Create: `neural-derby/derby/horses/__init__.py`, `neural-derby/derby/horses/blocks.py`, `neural-derby/tests/test_blocks.py`

**Interfaces:**
- Produces (all `nn.Module`): `ConvBlock(cin, cout, k=3)`, `PlainPair(c)`, `ResidualBlock(c)`, `ChannelAttention(c, r=16)`, `SpatialAttention(k=7)`, `InceptionBlock(cin, branch)` (output channels `3*branch`), `DepthwiseSeparable(cin, cout)`, `PatchEmbed(cin=3, dim=64, patch=8, img=96)` (output `B x T x dim`, `T = (img/patch)^2`), `TransformerEncoderBlock(dim=64, heads=4, mlp=128)` (`B x T x dim` in and out).

- [ ] **Step 1: Write the failing tests**

`neural-derby/derby/horses/__init__.py`: empty.

`neural-derby/tests/test_blocks.py`:

```python
import torch
from derby.horses import blocks as B

X = torch.randn(2, 3, 96, 96)


def test_conv_block_keeps_size_changes_channels():
    assert B.ConvBlock(3, 24)(X).shape == (2, 24, 96, 96)
    assert B.ConvBlock(3, 24, k=5)(X).shape == (2, 24, 96, 96)


def test_plain_pair_and_residual_same_shape():
    x = torch.randn(2, 24, 48, 48)
    assert B.PlainPair(24)(x).shape == x.shape
    assert B.ResidualBlock(24)(x).shape == x.shape


def test_channel_attention_scales_each_channel_between_0_and_1():
    x = torch.rand(2, 32, 12, 12) + 0.1
    y = B.ChannelAttention(32)(x)
    ratio = y / x
    assert torch.all(ratio >= 0) and torch.all(ratio <= 1)
    # the same channel gets the same scale at every pixel
    assert torch.allclose(ratio[0, 0].flatten(), ratio[0, 0].flatten()[0].expand(144), atol=1e-5)


def test_spatial_attention_masks_every_channel_the_same():
    x = torch.rand(2, 32, 12, 12) + 0.1
    y = B.SpatialAttention()(x)
    ratio = y / x
    assert torch.all(ratio >= 0) and torch.all(ratio <= 1)
    assert torch.allclose(ratio[0, 0], ratio[0, 5], atol=1e-5)


def test_inception_concatenates_three_branches():
    x = torch.randn(2, 24, 48, 48)
    assert B.InceptionBlock(24, 16)(x).shape == (2, 48, 48, 48)


def test_depthwise_separable_shape():
    x = torch.randn(2, 24, 48, 48)
    assert B.DepthwiseSeparable(24, 48)(x).shape == (2, 48, 48, 48)


def test_patch_embed_and_transformer_block():
    tokens = B.PatchEmbed(3, 64, 8, 96)(X)
    assert tokens.shape == (2, 144, 64)
    assert B.TransformerEncoderBlock(64, 4, 128)(tokens).shape == (2, 144, 64)
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_blocks.py`
Expected: `ImportError` / `ModuleNotFoundError` for `derby.horses.blocks`.

- [ ] **Step 3: Write blocks.py**

`neural-derby/derby/horses/blocks.py`:

```python
"""The building blocks ma'am taught in Module 3, written the way her chapter writes them.

Shapes are (batch, channels, height, width) unless a docstring says tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """conv -> batchnorm -> relu. The basic unit. Padding keeps height and width."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class PlainPair(nn.Module):
    """Two ConvBlocks in a row with NO shortcut. STAIRS is made of these."""

    def __init__(self, c: int):
        super().__init__()
        self.a = ConvBlock(c, c)
        self.b = ConvBlock(c, c)

    def forward(self, x):
        return self.b(self.a(x))


class ResidualBlock(nn.Module):
    """Two convs plus the input added back (x + F(x)). ELEVATOR is made of these."""

    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChannelAttention(nn.Module):
    """A volume knob per channel: squeeze to 1x1, score with two 1x1 convs, sigmoid, multiply."""

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(c // r, 4)
        self.score = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, c, 1))

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        scale = torch.sigmoid(self.score(avg) + self.score(mx))
        return x * scale


class SpatialAttention(nn.Module):
    """A where-mask: mean and max over channels, one 7x7 conv, sigmoid, multiply."""

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        mask = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * mask


class InceptionBlock(nn.Module):
    """Three lens sizes at once (1x1, 3x3, 5x5), glued along channels. Output has 3*branch channels."""

    def __init__(self, cin: int, branch: int):
        super().__init__()
        self.b1 = ConvBlock(cin, branch, k=1)
        self.b3 = ConvBlock(cin, branch, k=3)
        self.b5 = ConvBlock(cin, branch, k=5)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)


class DepthwiseSeparable(nn.Module):
    """One 3x3 filter per input channel (groups=cin), then a 1x1 conv to mix channels."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class PatchEmbed(nn.Module):
    """Cut the image into patch x patch squares, turn each into a token of size dim. Output: tokens (B, T, dim)."""

    def __init__(self, cin: int = 3, dim: int = 64, patch: int = 8, img: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, patch, stride=patch)
        n_tokens = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos


class TransformerEncoderBlock(nn.Module):
    """Self-attention then a small feed-forward, each with a skip. Tokens (B, T, dim) in and out.

    Attention is written with plain Linear layers so the multiply counter in stats.py can see them.
    """

    def __init__(self, dim: int = 64, heads: int = 4, mlp: int = 128):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        b, t, d = x.shape
        h = self.heads
        y = self.ln1(x)
        q = self.q(y).view(b, t, h, d // h).transpose(1, 2)
        k = self.k(y).view(b, t, h, d // h).transpose(1, 2)
        v = self.v(y).view(b, t, h, d // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (d // h) ** 0.5
        y = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.o(y)
        return x + self.mlp(self.ln2(x))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_blocks.py`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/horses neural-derby/tests/test_blocks.py
git commit -m "Add the Module 3 building blocks with shape tests"
```

---

### Task 3: The stable and the stats

**Files:**
- Create: `neural-derby/derby/horses/stable.py`, `neural-derby/derby/horses/stats.py`, `neural-derby/tests/test_stable.py`, `neural-derby/tests/test_stats.py`

**Interfaces:**
- Consumes: every class in `derby.horses.blocks`; `config.HORSES_ENABLED`, `config.IMG`.
- Produces: `Horse` dataclass `(key, name, tagline, colour, block_name, build: Callable[[], nn.Module])`; `STABLE: list[Horse]` in the fixed order textbook, stairs, elevator, featherweight, spotlight, trifocals, influencer; `enabled_stable(enabled: dict | None = None) -> list[Horse]`; `by_key(key) -> Horse`; `DEEP_STAGES` (the STAIRS/ELEVATOR depth knob). `stats.count_params(m) -> int`, `stats.count_macs(m, img=96) -> int`, `stats.depth(m) -> int`, `stats.layer_table(m, img=96) -> list[dict]` with keys `layer, type, out_shape, params`, `stats.horse_stats(horse) -> dict` with keys `params, macs, depth, table`.

- [ ] **Step 1: Write the failing stable tests**

`neural-derby/tests/test_stable.py`:

```python
import torch
from derby.horses import stable


def test_stable_order_and_keys():
    assert [h.key for h in stable.STABLE] == [
        "textbook", "stairs", "elevator", "featherweight", "spotlight", "trifocals", "influencer",
    ]


def test_every_horse_maps_image_to_six_logits():
    x = torch.randn(1, 3, 96, 96)
    for h in stable.STABLE:
        m = h.build()
        m.eval()
        assert m(x).shape == (1, 6), h.key


def test_enabled_stable_respects_switches():
    chosen = stable.enabled_stable({"textbook": True, "influencer": False})
    keys = [h.key for h in chosen]
    assert "influencer" not in keys
    assert "textbook" in keys and "stairs" in keys  # missing keys default to enabled


def test_stairs_and_elevator_differ_only_by_shortcuts():
    from derby.horses.stats import count_params
    s = count_params(stable.build_stairs())
    e = count_params(stable.build_elevator())
    assert s == e  # the shortcut adds no parameters


def test_every_horse_has_card_text():
    for h in stable.STABLE:
        assert h.name.isupper() or h.name.startswith("THE ")
        assert len(h.tagline) > 20
        assert h.colour.startswith("#") and len(h.colour) == 7
```

- [ ] **Step 2: Write the failing stats tests**

`neural-derby/tests/test_stats.py`:

```python
import torch.nn as nn
from derby.horses import stats, stable


def test_count_params_matches_torch():
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(8 * 8 * 8, 4))
    assert stats.count_params(m) == sum(p.numel() for p in m.parameters())


def test_count_macs_conv_and_linear_by_hand():
    # conv: 8 output maps of 8x8 = 512 outputs, each needs 3*3*3 = 27 multiplies -> 13824
    # linear: 512 inputs x 4 outputs -> 2048
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(512, 4))
    assert stats.count_macs(m, img=8) == 13824 + 2048


def test_depthwise_has_fewer_macs_than_textbook():
    assert stats.count_macs(stable.build_featherweight()) < stats.count_macs(stable.build_textbook())


def test_depth_counts_weight_layers():
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3), nn.Flatten(), nn.Linear(8, 2))
    assert stats.depth(m) == 3


def test_layer_table_rows_have_shapes_and_params():
    rows = stats.layer_table(stable.build_textbook())
    assert rows[0]["type"] == "Conv2d"
    assert rows[0]["out_shape"] == [24, 96, 96]
    assert rows[0]["params"] == 3 * 24 * 9
    assert rows[-1]["type"] == "Linear" and rows[-1]["out_shape"] == [6]


def test_horse_stats_bundle():
    s = stats.horse_stats(stable.by_key("textbook"))
    assert set(s) == {"params", "macs", "depth", "table"}
    assert s["params"] > 0 and s["macs"] > 0 and s["depth"] == 4
```

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/test_stable.py tests/test_stats.py`
Expected: import errors for `derby.horses.stable` and `derby.horses.stats`.

- [ ] **Step 4: Write stable.py**

`neural-derby/derby/horses/stable.py`:

```python
"""The seven horses. Each is one of ma'am's blocks with the same head: global average pool + one linear layer."""
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from derby import config
from derby.horses.blocks import (
    ChannelAttention, ConvBlock, DepthwiseSeparable, InceptionBlock, PatchEmbed,
    PlainPair, ResidualBlock, SpatialAttention, TransformerEncoderBlock,
)

N_CLASSES = len(config.CLASSES)


def head(c: int) -> nn.Sequential:
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, N_CLASSES))


def pool() -> nn.Module:
    return nn.MaxPool2d(2)


def build_textbook() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        ConvBlock(24, 48), pool(),
        ConvBlock(48, 96), pool(),
        head(96),
    )


# The depth knob for STAIRS and ELEVATOR: (width, number of pairs) per stage.
# Both horses use exactly this; only the pair type differs. Tune in prep (Task 14).
DEEP_STAGES = [(24, 1), (48, 1), (96, 2)]


def _tower(skips: bool) -> nn.Module:
    layers: list[nn.Module] = [ConvBlock(3, 24)]
    prev = 24
    for width, pairs in DEEP_STAGES:
        if width != prev:
            layers.append(ConvBlock(prev, width))
            prev = width
        for _ in range(pairs):
            layers.append(ResidualBlock(width) if skips else PlainPair(width))
        layers.append(pool())
    return nn.Sequential(*layers, head(prev))


def build_stairs() -> nn.Module:
    return _tower(skips=False)


def build_elevator() -> nn.Module:
    return _tower(skips=True)


def build_featherweight() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        DepthwiseSeparable(24, 48), pool(),
        DepthwiseSeparable(48, 96), pool(),
        head(96),
    )


def build_spotlight() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        ConvBlock(24, 48), ChannelAttention(48), SpatialAttention(), pool(),
        ConvBlock(48, 96), ChannelAttention(96), SpatialAttention(), pool(),
        head(96),
    )


def build_trifocals() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        InceptionBlock(24, 16), pool(),     # 48 channels out
        InceptionBlock(48, 32), pool(),     # 96 channels out
        head(96),
    )


class Influencer(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 3, heads: int = 4):
        super().__init__()
        self.embed = PatchEmbed(3, dim, 8, config.IMG)
        self.blocks = nn.Sequential(*[TransformerEncoderBlock(dim, heads, dim * 2) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        tokens = self.blocks(self.embed(x))
        return self.fc(self.norm(tokens.mean(dim=1)))


def build_influencer() -> nn.Module:
    return Influencer()


@dataclass(frozen=True)
class Horse:
    key: str
    name: str
    tagline: str
    colour: str
    block_name: str
    build: Callable[[], nn.Module]


STABLE: list[Horse] = [
    Horse("textbook", "TEXTBOOK",
          "Conv, pool, repeat. The recipe every network starts from.",
          "#ffd23f", "plain conv + maxpool", build_textbook),
    Horse("stairs", "STAIRS",
          "Eleven floors deep and the learning signal has to climb every step. Tires early, strong finish.",
          "#ff6b35", "deep conv, no shortcuts", build_stairs),
    Horse("elevator", "ELEVATOR",
          "Same building, but with shortcuts. The signal rides straight through.",
          "#3fb8ff", "residual (skip connection)", build_elevator),
    Horse("featherweight", "FEATHERWEIGHT",
          "One cheap step per colour channel, then mix. Eight times fewer multiplications.",
          "#7bff6b", "depthwise separable conv", build_featherweight),
    Horse("spotlight", "SPOTLIGHT",
          "Turns up the volume on useful features and points a light at where the object is.",
          "#ff3fd0", "channel + spatial attention", build_spotlight),
    Horse("trifocals", "TRIFOCALS",
          "Three lens sizes at once. Sees the whole shoe and the shoelace.",
          "#b56bff", "inception (multiscale)", build_trifocals),
    Horse("influencer", "THE INFLUENCER",
          "The architecture behind ChatGPT. Famous. Needs a million photos to perform.",
          "#ff4040", "transformer encoder", build_influencer),
]


def enabled_stable(enabled: dict[str, bool] | None = None) -> list[Horse]:
    enabled = config.HORSES_ENABLED if enabled is None else enabled
    return [h for h in STABLE if enabled.get(h.key, True)]


def by_key(key: str) -> Horse:
    for h in STABLE:
        if h.key == key:
            return h
    raise KeyError(key)
```

Note: the STAIRS tagline says "Eleven floors" because `DEEP_STAGES` gives 1 stem + 2 transitions + 4 pairs x 2 = 11 conv layers. If Task 14 changes `DEEP_STAGES`, update the tagline number to `stats.depth(build_stairs()) - 1` (minus the linear head). Keep the card honest.

- [ ] **Step 5: Write stats.py**

`neural-derby/derby/horses/stats.py`:

```python
"""Numbers ma'am asks for in exams: parameters, multiplies, depth, and a per-layer table."""
import torch
import torch.nn as nn

from derby import config
from derby.horses.stable import Horse

WEIGHT_LAYERS = (nn.Conv2d, nn.Linear)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_macs(m: nn.Module, img: int = config.IMG) -> int:
    """Multiply-accumulates for one image, counting Conv2d and Linear layers.

    Conv: every output value needs (in_channels / groups) * kh * kw multiplies.
    Linear: every output value needs in_features multiplies.
    The attention matmuls inside the transformer are not counted (they are small here); the card says so.
    """
    total = 0
    hooks = []

    def conv_hook(mod, _inp, out):
        nonlocal total
        per_output = (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        total += (out.numel() // out.shape[0]) * per_output

    def linear_hook(mod, _inp, out):
        nonlocal total
        total += (out.numel() // out.shape[0]) * mod.in_features

    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 3, img, img))
    m.train(was_training)
    for h in hooks:
        h.remove()
    return total


def depth(m: nn.Module) -> int:
    return sum(isinstance(mod, WEIGHT_LAYERS) for mod in m.modules())


def layer_table(m: nn.Module, img: int = config.IMG) -> list[dict]:
    """One row per weight layer, in forward order: layer name, type, output shape (no batch), params."""
    rows: list[dict] = []
    hooks = []

    def make_hook(name):
        def hook(mod, _inp, out):
            rows.append({
                "layer": name,
                "type": type(mod).__name__,
                "out_shape": list(out.shape[1:]),
                "params": sum(p.numel() for p in mod.parameters()),
            })
        return hook

    for name, mod in m.named_modules():
        if isinstance(mod, WEIGHT_LAYERS):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 3, img, img))
    m.train(was_training)
    for h in hooks:
        h.remove()
    return rows


def horse_stats(horse: Horse, img: int = config.IMG) -> dict:
    m = horse.build()
    return {
        "params": count_params(m),
        "macs": count_macs(m, img),
        "depth": depth(m),
        "table": layer_table(m, img),
    }
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_stable.py tests/test_stats.py`
Expected: `11 passed`

- [ ] **Step 7: Print the stable's numbers and sanity-check them**

Run:

```powershell
python -c "from derby.horses import stable, stats; [print(f'{h.key:14s} params={s[\"params\"]:>8,} macs={s[\"macs\"]:>12,} depth={s[\"depth\"]}') for h in stable.STABLE for s in [stats.horse_stats(h)]]"
```

Expected: seven lines. FEATHERWEIGHT must have the fewest MACs; STAIRS and ELEVATOR must show identical params and MACs; THE INFLUENCER's params should be in the 100k to 200k range. Paste the output into the commit message body.

- [ ] **Step 8: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/horses neural-derby/tests/test_stable.py neural-derby/tests/test_stats.py
git commit -m "Add the seven horses and the parameter, multiply, and layer-table stats"
```

---

### Task 4: Backup dataset tool and loader

**Files:**
- Create: `neural-derby/derby/data/__init__.py`, `neural-derby/derby/data/images.py`, `neural-derby/derby/data/backup.py`, `neural-derby/tools/build_backup_dataset.py`, `neural-derby/tests/test_backup.py`, `neural-derby/tests/conftest.py`

**Interfaces:**
- Consumes: `config.CLASSES`, `config.IMG`, `config.DATA_DIR`, `config.BACKUP_TEST_FRAC`.
- Produces: `images.load_image(path: Path, img: int = 96) -> torch.Tensor` (uint8, `3 x img x img`, EXIF-rotated, shorter side resized to `img` then centre-cropped); `images.load_many(paths, img) -> torch.Tensor` (uint8 `N x 3 x img x img`); `backup.Loaded` dataclass `(x: Tensor, y: Tensor, paths: list[Path])`; `backup.load(root: Path = DATA_DIR/"backup", img=96) -> Loaded`; `backup.split_indices(n: int, seed: int, test_frac: float = 0.15) -> tuple[Tensor, Tensor]`.
- The tool is run once in prep with internet; tests never touch the network.

- [ ] **Step 1: Write a shared test fixture that makes a tiny image folder**

`neural-derby/tests/conftest.py`:

```python
from pathlib import Path

import pytest
from PIL import Image

from derby import config


def make_image(path: Path, colour, size=(200, 150)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG")


@pytest.fixture
def tiny_backup(tmp_path: Path) -> Path:
    """Two images per class, solid colours, landscape and portrait mixed."""
    root = tmp_path / "backup"
    for i, cls in enumerate(config.CLASSES):
        shade = 40 * i
        make_image(root / cls / "a.jpg", (shade, 100, 200), (200, 150))
        make_image(root / cls / "b.jpg", (200, shade, 100), (120, 180))
    return root
```

- [ ] **Step 2: Write the failing tests**

`neural-derby/tests/test_backup.py`:

```python
from pathlib import Path

import torch
from PIL import Image

from derby import config
from derby.data import backup, images


def test_load_image_is_square_uint8(tiny_backup: Path):
    t = images.load_image(tiny_backup / "bottle" / "a.jpg", 96)
    assert t.dtype == torch.uint8 and t.shape == (3, 96, 96)


def test_load_image_applies_exif_rotation(tmp_path: Path):
    # a 100x50 image tagged as rotated 90 degrees must load as portrait before cropping
    p = tmp_path / "rot.jpg"
    im = Image.new("RGB", (100, 50), (255, 0, 0))
    im.paste((0, 0, 255), (0, 0, 50, 50))          # left half blue
    exif = im.getexif()
    exif[274] = 6                                  # orientation 6 = rotate 90 clockwise
    im.save(p, "JPEG", exif=exif.tobytes())
    t = images.load_image(p, 32)
    # after rotating clockwise the blue half is on top
    top = t[:, :8, :].float().mean(dim=(1, 2))
    assert top[2] > top[0]


def test_load_backup_labels_follow_class_order(tiny_backup: Path):
    loaded = backup.load(tiny_backup, img=32)
    assert loaded.x.shape == (12, 3, 32, 32)
    assert loaded.y.tolist() == [i for i in range(6) for _ in range(2)]
    assert len(loaded.paths) == 12


def test_split_is_deterministic_and_disjoint():
    a_train, a_test = backup.split_indices(100, seed=7, test_frac=0.15)
    b_train, b_test = backup.split_indices(100, seed=7, test_frac=0.15)
    assert torch.equal(a_train, b_train) and torch.equal(a_test, b_test)
    assert len(a_test) == 15 and len(a_train) == 85
    assert set(a_train.tolist()).isdisjoint(a_test.tolist())
```

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/test_backup.py`
Expected: import errors for `derby.data`.

- [ ] **Step 4: Write images.py and backup.py**

`neural-derby/derby/data/__init__.py`: empty.

`neural-derby/derby/data/images.py`:

```python
"""Turn a photo file into the fixed-size uint8 tensor every horse eats."""
from pathlib import Path

import torch
from PIL import Image, ImageOps

from derby import config


def open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)   # phones store rotation as a tag; apply it
    return im.convert("RGB")


def to_square(im: Image.Image, img: int) -> Image.Image:
    w, h = im.size
    scale = img / min(w, h)
    im = im.resize((max(img, round(w * scale)), max(img, round(h * scale))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - img) // 2, (h - img) // 2
    return im.crop((left, top, left + img, top + img))


def load_image(path: Path, img: int = config.IMG) -> torch.Tensor:
    im = to_square(open_rgb(path), img)
    arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
    return arr.view(img, img, 3).permute(2, 0, 1).contiguous()


def load_many(paths: list[Path], img: int = config.IMG) -> torch.Tensor:
    if not paths:
        return torch.empty(0, 3, img, img, dtype=torch.uint8)
    return torch.stack([load_image(p, img) for p in paths])
```

`neural-derby/derby/data/backup.py`:

```python
"""The safety-net dataset built from LVIS/COCO crops. Folder per class, label = index in config.CLASSES."""
from dataclasses import dataclass
from pathlib import Path

import torch

from derby import config
from derby.data.images import load_many

EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Loaded:
    x: torch.Tensor          # uint8, N x 3 x img x img
    y: torch.Tensor          # long, N
    paths: list[Path]


def list_class_files(root: Path) -> list[tuple[Path, int]]:
    out = []
    for label, cls in enumerate(config.CLASSES):
        folder = root / cls
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                out.append((p, label))
    return out


def load(root: Path = config.DATA_DIR / "backup", img: int = config.IMG) -> Loaded:
    files = list_class_files(root)
    paths = [p for p, _ in files]
    y = torch.tensor([label for _, label in files], dtype=torch.long)
    return Loaded(load_many(paths, img), y, paths)


def split_indices(n: int, seed: int, test_frac: float = config.BACKUP_TEST_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = int(round(n * test_frac))
    return perm[n_test:], perm[:n_test]
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_backup.py`
Expected: `4 passed`

- [ ] **Step 6: Write the LVIS crop tool**

`neural-derby/tools/build_backup_dataset.py`:

```python
"""Build data/backup/<class>/ from LVIS v1 annotations over COCO 2017 images. Run once, with internet.

Usage (from neural-derby/):  python tools/build_backup_dataset.py --per-class 400 --min-side 64
Downloads only the photos it needs (about 0.5 GB). Safe to re-run; it skips what exists.
Licence: LVIS annotations CC BY 4.0; COCO images under Flickr terms (research use).
"""
import argparse
import collections
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402

LVIS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
GROUPS = {
    "bottle": {"water_bottle", "bottle"},
    "pen": {"pen", "pencil"},
    "book": {"book", "notebook"},
    "phone": {"cellular_telephone"},
    "backpack": {"backpack"},
    "shoe": {"shoe"},
}


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:  # network hiccup: wait and retry
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--max-per-image", type=int, default=2)
    ap.add_argument("--long-side", type=int, default=160)
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "backup")
    args = ap.parse_args()

    cache = config.DATA_DIR / "cache"
    print("1/4 fetching LVIS annotations (350 MB zip)...")
    zpath = fetch(LVIS_URL, cache / "lvis_v1_train.json.zip")
    with zipfile.ZipFile(zpath) as z:
        lvis = json.load(z.open("lvis_v1_train.json"))

    name_to_group = {n: g for g, names in GROUPS.items() for n in names}
    cat_to_group = {c["id"]: name_to_group[c["name"]] for c in lvis["categories"] if c["name"] in name_to_group}
    images = {im["id"]: im for im in lvis["images"]}

    print("2/4 choosing boxes...")
    picked: dict[str, list] = collections.defaultdict(list)
    per_image: dict[tuple, int] = collections.Counter()
    for a in lvis["annotations"]:
        g = cat_to_group.get(a["category_id"])
        if g is None or len(picked[g]) >= args.per_class:
            continue
        x, y, w, h = a["bbox"]
        if min(w, h) < args.min_side or per_image[(a["image_id"], g)] >= args.max_per_image:
            continue
        per_image[(a["image_id"], g)] += 1
        picked[g].append((a["image_id"], x, y, w, h))
    for g in GROUPS:
        print(f"   {g}: {len(picked[g])} crops")

    print("3/4 downloading photos and cropping...")
    for g, items in picked.items():
        out_dir = args.out / g
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_id, x, y, w, h) in enumerate(items):
            dest = out_dir / f"{i:04d}.jpg"
            if dest.exists():
                continue
            url = images[img_id]["coco_url"]
            src = fetch(url, cache / "coco" / Path(url).name)
            im = Image.open(src).convert("RGB").crop((int(x), int(y), int(x + w), int(y + h)))
            im.thumbnail((args.long_side, args.long_side))
            im.save(dest, "JPEG", quality=90)
            if i % 50 == 0:
                print(f"   {g} {i}/{len(items)}")

    print("4/4 done. Counts:")
    for g in GROUPS:
        print(f"   {g}: {len(list((args.out / g).glob('*.jpg')))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tool (internet needed) and eyeball the crops**

Run: `python tools/build_backup_dataset.py`
Expected: about 400 crops per class printed in the final counts (pen may be fewer). Open `data/backup/pen/` and `data/backup/shoe/` in the file explorer and look at 20 crops each: they should mostly be the object, not background. If pen is under 250, lower `--min-side` to 48 for pen only by re-running with `--min-side 48 --per-class 400` (existing crops are kept, new ones added).

Then run: `python -c "from derby.data import backup; L = backup.load(); print(L.x.shape, L.y.bincount().tolist())"`
Expected: a shape like `torch.Size([2300, 3, 96, 96])` and six counts.

- [ ] **Step 8: Commit (data stays git-ignored)**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/data neural-derby/tools/build_backup_dataset.py neural-derby/tests/conftest.py neural-derby/tests/test_backup.py
git commit -m "Add image loading, the backup dataset loader, and the LVIS crop tool"
```

---
### Task 5: Live uploads

**Files:**
- Create: `neural-derby/derby/data/live.py`, `neural-derby/tests/test_live.py`

**Interfaces:**
- Consumes: `config.CAPS`, `config.CLASSES`, `config.MIN_LIVE_FOR_SPLIT`, `images.load_many`.
- Produces: `Photo` dataclass `(device: str, class_name: str, path: Path)` with property `label: int`; `LiveError(Exception)` whose message is safe to show on a phone; `LiveStore(root: Path, caps=config.CAPS, classes=config.CLASSES, long_side=160)` with `add(device, class_name, data: bytes) -> Photo`, `count() -> int`, `count_for(device) -> int`, `photos() -> list[Photo]`, `split_by_device(seed, min_for_split=12) -> tuple[list[Photo], list[Photo]]` (train, test), `tensors(photos, img=96) -> tuple[Tensor, Tensor]` (static), `thumbnail_b64(photo, size=96) -> str` (static, JPEG base64 without the data-URL prefix).
- Device ids must match `[A-Za-z0-9_-]{4,32}`; anything else is a `LiveError`.

- [ ] **Step 1: Write the failing tests**

`neural-derby/tests/test_live.py`:

```python
import io
from pathlib import Path

import pytest
from PIL import Image

from derby.data.live import LiveError, LiveStore


def jpeg_bytes(size=(120, 80), colour=(10, 200, 30), orientation=None) -> bytes:
    im = Image.new("RGB", size, colour)
    buf = io.BytesIO()
    if orientation:
        exif = im.getexif()
        exif[274] = orientation
        im.save(buf, "JPEG", exif=exif.tobytes())
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path: Path) -> LiveStore:
    return LiveStore(tmp_path / "live", caps={"per_device": 3, "total": 5, "max_bytes": 50_000})


def test_add_stores_photo_under_class_folder(store: LiveStore):
    p = store.add("dev_0001", "bottle", jpeg_bytes())
    assert p.path.exists() and p.path.parent.name == "bottle"
    assert p.label == 0 and store.count() == 1


def test_per_device_cap(store: LiveStore):
    for _ in range(3):
        store.add("dev_0001", "pen", jpeg_bytes())
    with pytest.raises(LiveError, match="3 photos"):
        store.add("dev_0001", "pen", jpeg_bytes())


def test_total_cap(store: LiveStore):
    for i in range(5):
        store.add(f"dev_{i:04d}", "book", jpeg_bytes())
    with pytest.raises(LiveError, match="full"):
        store.add("dev_9999", "book", jpeg_bytes())


def test_size_cap_and_bad_bytes_and_bad_class(store: LiveStore):
    with pytest.raises(LiveError, match="too big"):
        store.add("dev_0001", "shoe", b"x" * 60_000)
    with pytest.raises(LiveError, match="not a photo"):
        store.add("dev_0001", "shoe", b"definitely not an image")
    with pytest.raises(LiveError, match="six objects"):
        store.add("dev_0001", "laptop", jpeg_bytes())
    with pytest.raises(LiveError):
        store.add("../evil", "shoe", jpeg_bytes())


def test_exif_rotation_is_applied_on_save(store: LiveStore):
    p = store.add("dev_0001", "phone", jpeg_bytes(size=(100, 50), orientation=6))
    with Image.open(p.path) as im:
        assert im.size == (50, 100)


def test_split_by_device_never_leaks_a_device(store: LiveStore):
    store = LiveStore(store.root, caps={"per_device": 3, "total": 100, "max_bytes": 50_000})
    for d in range(6):
        for _ in range(2):
            store.add(f"dev_{d:04d}", "backpack", jpeg_bytes())
    train, test = store.split_by_device(seed=3)
    assert len(train) + len(test) == 12 and train and test
    assert {p.device for p in train}.isdisjoint({p.device for p in test})


def test_fewer_than_twelve_all_go_to_test(store: LiveStore):
    for d in range(4):
        store.add(f"dev_{d:04d}", "backpack", jpeg_bytes())
    train, test = store.split_by_device(seed=3)
    assert train == [] and len(test) == 4


def test_tensors_and_thumbnail(store: LiveStore):
    p = store.add("dev_0001", "shoe", jpeg_bytes())
    x, y = LiveStore.tensors([p], img=32)
    assert x.shape == (1, 3, 32, 32) and y.tolist() == [5]
    assert LiveStore.thumbnail_b64(p).startswith("/9j/")  # base64 of a JPEG starts like this
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_live.py`
Expected: `ModuleNotFoundError: derby.data.live`

- [ ] **Step 3: Write live.py**

`neural-derby/derby/data/live.py`:

```python
"""Photos from the room's phones: validation, caps, EXIF rotation, per-device split."""
import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from derby import config
from derby.data.images import load_many

DEVICE_RE = re.compile(r"[A-Za-z0-9_-]{4,32}")


@dataclass(frozen=True)
class Photo:
    device: str
    class_name: str
    path: Path

    @property
    def label(self) -> int:
        return config.CLASSES.index(self.class_name)


class LiveError(Exception):
    """A rejection whose message is safe to show on the phone."""


class LiveStore:
    def __init__(self, root: Path, caps: dict = config.CAPS, classes: list[str] = config.CLASSES, long_side: int = 160):
        self.root = Path(root)
        self.caps = caps
        self.classes = classes
        self.long_side = long_side
        self._photos: list[Photo] = []
        self.root.mkdir(parents=True, exist_ok=True)

    def count(self) -> int:
        return len(self._photos)

    def count_for(self, device: str) -> int:
        return sum(p.device == device for p in self._photos)

    def photos(self) -> list[Photo]:
        return list(self._photos)

    def add(self, device: str, class_name: str, data: bytes) -> Photo:
        if not DEVICE_RE.fullmatch(device or ""):
            raise LiveError("Reload the page and try again.")
        if class_name not in self.classes:
            raise LiveError("Pick one of the six objects first.")
        if len(data) > self.caps["max_bytes"]:
            raise LiveError("That photo is too big. Try again with a smaller one.")
        if len(self._photos) >= self.caps["total"]:
            raise LiveError("The paddock is full. Thanks! Now watch the race.")
        if self.count_for(device) >= self.caps["per_device"]:
            raise LiveError(f"You have sent your {self.caps['per_device']} photos. Now place a bet!")
        try:
            probe = Image.open(io.BytesIO(data))
            probe.verify()
            im = Image.open(io.BytesIO(data))
            if im.format not in ("JPEG", "PNG"):
                raise LiveError("Only JPEG or PNG photos, please.")
            im = ImageOps.exif_transpose(im).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError):
            raise LiveError("That file is not a photo we can read.")
        im.thumbnail((self.long_side, self.long_side))
        folder = self.root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{device}_{self.count_for(device)}.jpg"
        im.save(path, "JPEG", quality=90)
        photo = Photo(device, class_name, path)
        self._photos.append(photo)
        return photo

    def split_by_device(self, seed: int, min_for_split: int = config.MIN_LIVE_FOR_SPLIT) -> tuple[list[Photo], list[Photo]]:
        """Half the devices go to test, half to train. Too few photos: everything is test."""
        photos = list(self._photos)
        if len(photos) < min_for_split:
            return [], photos
        devices = sorted({p.device for p in photos})
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(devices), generator=g).tolist()
        test_devices = {devices[i] for i in perm[: len(devices) // 2]}
        train = [p for p in photos if p.device not in test_devices]
        test = [p for p in photos if p.device in test_devices]
        return train, test

    @staticmethod
    def tensors(photos: list[Photo], img: int = config.IMG) -> tuple[torch.Tensor, torch.Tensor]:
        x = load_many([p.path for p in photos], img)
        y = torch.tensor([p.label for p in photos], dtype=torch.long)
        return x, y

    @staticmethod
    def thumbnail_b64(photo: Photo, size: int = 96) -> str:
        with Image.open(photo.path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_live.py`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/data/live.py neural-derby/tests/test_live.py
git commit -m "Add the live photo store with caps, validation, and per-device split"
```

---

### Task 6: GPU augmentation and dataset assembly

**Files:**
- Create: `neural-derby/derby/data/augment.py`, `neural-derby/derby/data/dataset.py`, `neural-derby/tests/test_augment.py`, `neural-derby/tests/test_dataset.py`

**Interfaces:**
- Consumes: `backup.Loaded`, `backup.split_indices`.
- Produces: `augment.to_float(x_uint8) -> Tensor` (float in [0,1]); `augment.normalize(x_float) -> Tensor` (`(x-0.5)/0.5`); `augment.augment(x_uint8, gen: torch.Generator) -> Tensor` (float in [0,1], same shape; the generator must live on the same device as `x`); `augment.preview_pairs(x_uint8_cpu, n=48, seed=0) -> list[tuple[str, str]]` (PNG base64 strings, original and augmented); `dataset.RaceData` dataclass `(train_x, train_y, test_x, test_y, test_is_live: BoolTensor, n_live_train: int, n_live_test: int)`; `dataset.assemble(backup: Loaded, live_train: tuple[Tensor, Tensor] | None, live_test: tuple[Tensor, Tensor] | None, seed: int, device: str | torch.device = "cpu") -> RaceData`.
- Accuracies, images, and masks follow the same index order within `test_x`: backup test rows first, then live test rows.

- [ ] **Step 1: Write the failing augmentation tests**

`neural-derby/tests/test_augment.py`:

```python
import base64

import torch

from derby.data import augment


def batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, 96, 96), generator=g, dtype=torch.uint8)


def test_to_float_and_normalize_ranges():
    x = augment.to_float(batch())
    assert x.dtype == torch.float32 and x.min() >= 0 and x.max() <= 1
    z = augment.normalize(x)
    assert z.min() >= -1 and z.max() <= 1


def test_augment_shape_range_and_determinism():
    x = batch()
    a = augment.augment(x, torch.Generator().manual_seed(1))
    b = augment.augment(x, torch.Generator().manual_seed(1))
    c = augment.augment(x, torch.Generator().manual_seed(2))
    assert a.shape == x.shape and a.dtype == torch.float32
    assert a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_augment_actually_changes_the_image():
    x = batch()
    a = augment.augment(x, torch.Generator().manual_seed(1))
    assert (a - augment.to_float(x)).abs().mean() > 0.05


def test_preview_pairs_are_png_base64():
    pairs = augment.preview_pairs(batch(6), n=4, seed=0)
    assert len(pairs) == 4
    for orig, aug in pairs:
        assert base64.b64decode(orig)[:8] == b"\x89PNG\r\n\x1a\n"
        assert base64.b64decode(aug)[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Write the failing dataset tests**

`neural-derby/tests/test_dataset.py`:

```python
import torch

from derby.data import backup, dataset


def fake_loaded(n=40):
    x = torch.randint(0, 256, (n, 3, 96, 96), dtype=torch.uint8)
    y = torch.arange(n) % 6
    return backup.Loaded(x, y, [None] * n)


def test_assemble_backup_only():
    d = dataset.assemble(fake_loaded(40), None, None, seed=1)
    assert d.train_x.shape[0] == 34 and d.test_x.shape[0] == 6
    assert d.test_is_live.dtype == torch.bool and not d.test_is_live.any()
    assert d.n_live_train == 0 and d.n_live_test == 0


def test_assemble_with_live_rows_appended_and_flagged():
    live_tr = (torch.zeros(5, 3, 96, 96, dtype=torch.uint8), torch.zeros(5, dtype=torch.long))
    live_te = (torch.zeros(3, 3, 96, 96, dtype=torch.uint8), torch.ones(3, dtype=torch.long))
    d = dataset.assemble(fake_loaded(40), live_tr, live_te, seed=1)
    assert d.train_x.shape[0] == 39 and d.test_x.shape[0] == 9
    assert d.test_is_live.tolist() == [False] * 6 + [True] * 3
    assert d.n_live_train == 5 and d.n_live_test == 3
    assert d.train_y.dtype == torch.long and d.test_x.dtype == torch.uint8
```

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/test_augment.py tests/test_dataset.py`
Expected: import errors.

- [ ] **Step 4: Write augment.py**

`neural-derby/derby/data/augment.py`:

```python
"""Module 4 on the GPU: one affine warp (crop, rotate, flip), brightness and contrast, noise, one erased box.

Everything here works on a whole batch at once with torch ops, so a lap stays fast.
The generator passed in must be on the same device as the images.
"""
import base64
import io
import math

import torch
import torch.nn.functional as F
from PIL import Image


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float().div_(255.0)


def normalize(x_float: torch.Tensor) -> torch.Tensor:
    return (x_float - 0.5) / 0.5


def _uniform(shape, lo: float, hi: float, gen: torch.Generator, device) -> torch.Tensor:
    return torch.rand(shape, generator=gen, device=device) * (hi - lo) + lo


def augment(x_uint8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    x = to_float(x_uint8)
    b = x.shape[0]
    dev = x.device

    # 1. crop + rotate + flip as one affine warp
    scale = _uniform((b,), 0.6, 1.0, gen, dev)                       # 1.0 = whole image, 0.6 = zoomed crop
    angle = _uniform((b,), -15.0, 15.0, gen, dev) * math.pi / 180.0
    flip = torch.where(torch.rand((b,), generator=gen, device=dev) < 0.5, -1.0, 1.0)
    tx = _uniform((b,), -1.0, 1.0, gen, dev) * (1.0 - scale)
    ty = _uniform((b,), -1.0, 1.0, gen, dev) * (1.0 - scale)
    cos, sin = torch.cos(angle), torch.sin(angle)
    row0 = torch.stack([scale * cos * flip, -scale * sin, tx], dim=1)
    row1 = torch.stack([scale * sin * flip, scale * cos, ty], dim=1)
    theta = torch.stack([row0, row1], dim=1)                          # b x 2 x 3
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    # 2. brightness (add) and contrast (multiply around the middle grey)
    contrast = _uniform((b, 1, 1, 1), 0.7, 1.3, gen, dev)
    brightness = _uniform((b, 1, 1, 1), -0.3, 0.3, gen, dev)
    x = (x - 0.5) * contrast + 0.5 + brightness

    # 3. noise
    x = x + torch.randn(x.shape, generator=gen, device=dev) * 0.02

    # 4. erase one rectangle on 30 percent of the images
    x = _erase(x, gen, p=0.3)
    return x.clamp_(0.0, 1.0)


def _erase(x: torch.Tensor, gen: torch.Generator, p: float) -> torch.Tensor:
    b, _, h, w = x.shape
    dev = x.device
    do = torch.rand((b,), generator=gen, device=dev) < p
    eh = (_uniform((b,), 0.2, 0.5, gen, dev) * h).long()
    ew = (_uniform((b,), 0.2, 0.5, gen, dev) * w).long()
    y0 = (torch.rand((b,), generator=gen, device=dev) * (h - eh)).long()
    x0 = (torch.rand((b,), generator=gen, device=dev) * (w - ew)).long()
    ys = torch.arange(h, device=dev).view(1, h, 1)
    xs = torch.arange(w, device=dev).view(1, 1, w)
    inside = (ys >= y0.view(b, 1, 1)) & (ys < (y0 + eh).view(b, 1, 1)) \
        & (xs >= x0.view(b, 1, 1)) & (xs < (x0 + ew).view(b, 1, 1))
    mask = (inside & do.view(b, 1, 1)).unsqueeze(1)
    return x.masked_fill(mask, 0.0)


def _png_b64(x_float_3hw: torch.Tensor) -> str:
    arr = (x_float_3hw.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview_pairs(x_uint8_cpu: torch.Tensor, n: int = 48, seed: int = 0) -> list[tuple[str, str]]:
    """Real before/after pairs for the MULTIPLY act. Runs on CPU on a sample."""
    x = x_uint8_cpu[:n].cpu()
    aug = augment(x, torch.Generator().manual_seed(seed))
    orig = to_float(x)
    return [(_png_b64(orig[i]), _png_b64(aug[i])) for i in range(x.shape[0])]
```

- [ ] **Step 5: Write dataset.py**

`neural-derby/derby/data/dataset.py`:

```python
"""Put backup and live photos together into the tensors a race trains and tests on."""
from dataclasses import dataclass

import torch

from derby.data.backup import Loaded, split_indices


@dataclass
class RaceData:
    train_x: torch.Tensor      # uint8 N x 3 x 96 x 96
    train_y: torch.Tensor      # long N
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_is_live: torch.Tensor  # bool, True for rows that came from the room's phones
    n_live_train: int
    n_live_test: int

    @property
    def n_train(self) -> int:
        return int(self.train_y.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test_y.shape[0])


def assemble(backup: Loaded, live_train, live_test, seed: int, device="cpu") -> RaceData:
    tr_idx, te_idx = split_indices(int(backup.y.shape[0]), seed)
    train_x = [backup.x[tr_idx]]
    train_y = [backup.y[tr_idx]]
    test_x = [backup.x[te_idx]]
    test_y = [backup.y[te_idx]]
    is_live = [torch.zeros(len(te_idx), dtype=torch.bool)]
    n_live_train = n_live_test = 0
    if live_train is not None and live_train[0].shape[0] > 0:
        train_x.append(live_train[0]); train_y.append(live_train[1])
        n_live_train = int(live_train[0].shape[0])
    if live_test is not None and live_test[0].shape[0] > 0:
        test_x.append(live_test[0]); test_y.append(live_test[1])
        is_live.append(torch.ones(live_test[0].shape[0], dtype=torch.bool))
        n_live_test = int(live_test[0].shape[0])
    return RaceData(
        torch.cat(train_x).to(device), torch.cat(train_y).to(device),
        torch.cat(test_x).to(device), torch.cat(test_y).to(device),
        torch.cat(is_live).to(device), n_live_train, n_live_test,
    )
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_augment.py tests/test_dataset.py`
Expected: `6 passed`

- [ ] **Step 7: Look at the augmentation with your own eyes**

Run (writes a contact sheet you must open):

```powershell
python -c "import torch; from derby.data import backup, augment; from PIL import Image; import base64, io; L = backup.load(); pairs = augment.preview_pairs(L.x[::max(1,len(L.x)//12)], n=12, seed=3); sheet = Image.new('RGB', (12*96, 2*96)); [sheet.paste(Image.open(io.BytesIO(base64.b64decode(p[r]))), (i*96, r*96)) for i, p in enumerate(pairs) for r in (0, 1)]; sheet.save('data/aug_sheet.png'); print('wrote data/aug_sheet.png')"
```

Open `data/aug_sheet.png`. Top row originals, bottom row augmented. Expected: visible rotation, zoom, brightness changes, a few black rectangles, nothing turned to garbage. If the bottom row is unrecognisable, the ranges in `augment()` are too strong; reduce rotation to 10 and contrast to 0.8 to 1.2.

- [ ] **Step 8: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/data/augment.py neural-derby/derby/data/dataset.py neural-derby/tests/test_augment.py neural-derby/tests/test_dataset.py
git commit -m "Add GPU augmentation with preview pairs and race data assembly"
```

---

### Task 7: Lock-step trainer

**Files:**
- Create: `neural-derby/derby/race/__init__.py`, `neural-derby/derby/race/trainer.py`, `neural-derby/tests/test_trainer.py`

**Interfaces:**
- Consumes: `stable.Horse`, `dataset.RaceData`, `augment.augment/normalize/to_float`, `config.LAPS/BATCH/LAP_FLOOR_S/LR/SEED`.
- Produces: `LapEvent` dataclass `(horse: str, lap: int, train_acc: float, test_acc: float, test_acc_internet: float | None, test_acc_live: float | None, loss: float, ms: float, t: float)` with `to_dict()`; accuracies are fractions in [0, 1]; `t` is seconds since the race started. `Race(stable: list[Horse], data: RaceData, laps=20, batch=64, lap_floor_s=4.0, seed=1234, lr=2e-3, device=None)` with `run() -> Iterator[LapEvent]`, `models: dict[str, nn.Module]`, `events: list[LapEvent]`, `evaluate(key) -> dict` with keys `all, internet, live`.
- Event order is strictly lap-major: all horses report lap 1 before any reports lap 2.

- [ ] **Step 1: Write the failing tests**

`neural-derby/tests/test_trainer.py`:

```python
import time

import torch

from derby.data.dataset import RaceData
from derby.horses import stable
from derby.race.trainer import LapEvent, Race


def synthetic(n_train=64, n_test=16, n_live_test=0):
    g = torch.Generator().manual_seed(0)
    tx = torch.randint(0, 256, (n_train, 3, 96, 96), generator=g, dtype=torch.uint8)
    ty = torch.arange(n_train) % 6
    ex = torch.randint(0, 256, (n_test + n_live_test, 3, 96, 96), generator=g, dtype=torch.uint8)
    ey = torch.arange(n_test + n_live_test) % 6
    is_live = torch.tensor([False] * n_test + [True] * n_live_test)
    return RaceData(tx, ty, ex, ey, is_live, 0, n_live_test)


def two_horses():
    return [stable.by_key("textbook"), stable.by_key("featherweight")]


def test_events_are_lap_major_and_complete():
    race = Race(two_horses(), synthetic(), laps=2, batch=16, lap_floor_s=0.0, seed=1, device="cpu")
    events = list(race.run())
    assert [(e.horse, e.lap) for e in events] == [
        ("textbook", 1), ("featherweight", 1), ("textbook", 2), ("featherweight", 2),
    ]
    assert race.events == events
    for e in events:
        assert isinstance(e, LapEvent)
        assert 0 <= e.train_acc <= 1 and 0 <= e.test_acc <= 1
        assert e.test_acc_live is None and e.test_acc_internet == e.test_acc
        assert e.ms > 0 and e.t >= 0
    assert set(race.models) == {"textbook", "featherweight"}


def test_live_accuracy_reported_when_live_rows_exist():
    race = Race(two_horses()[:1], synthetic(n_live_test=4), laps=1, batch=16, lap_floor_s=0.0, seed=1, device="cpu")
    (e,) = list(race.run())
    assert e.test_acc_live is not None and e.test_acc_internet is not None


def test_pacing_floor_is_honoured():
    race = Race(two_horses()[:1], synthetic(n_train=16, n_test=8), laps=3, batch=16, lap_floor_s=0.25, seed=1, device="cpu")
    start = time.perf_counter()
    list(race.run())
    assert time.perf_counter() - start >= 0.5  # floors apply between laps 1->2 and 2->3


def test_same_seed_same_events():
    a = list(Race(two_horses()[:1], synthetic(), laps=1, batch=16, lap_floor_s=0.0, seed=5, device="cpu").run())
    b = list(Race(two_horses()[:1], synthetic(), laps=1, batch=16, lap_floor_s=0.0, seed=5, device="cpu").run())
    assert a[0].loss == b[0].loss and a[0].test_acc == b[0].test_acc


def test_to_dict_round_trip():
    e = LapEvent("textbook", 1, 0.5, 0.4, 0.4, None, 1.2, 10.0, 0.3)
    assert LapEvent(**e.to_dict()) == e
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_trainer.py`
Expected: `ModuleNotFoundError: derby.race`

- [ ] **Step 3: Write trainer.py**

`neural-derby/derby/race/__init__.py`: empty.

`neural-derby/derby/race/trainer.py`:

```python
"""The race engine. One lap = one epoch for every horse, in lock-step, with the same shuffle order.

The trainer knows nothing about the web. It yields LapEvents; whoever calls it decides what to do with them.
"""
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from derby import config
from derby.data.augment import augment, normalize, to_float
from derby.data.dataset import RaceData
from derby.horses.stable import Horse


@dataclass(frozen=True)
class LapEvent:
    horse: str
    lap: int
    train_acc: float
    test_acc: float
    test_acc_internet: float | None
    test_acc_live: float | None
    loss: float
    ms: float      # wall time this horse spent on this lap
    t: float       # seconds since the race started

    def to_dict(self) -> dict:
        return asdict(self)


class Race:
    def __init__(self, stable: list[Horse], data: RaceData, laps: int = config.LAPS, batch: int = config.BATCH,
                 lap_floor_s: float = config.LAP_FLOOR_S, seed: int = config.SEED, lr: float = config.LR,
                 device=None):
        self.stable = stable
        self.data = data
        self.laps = laps
        self.batch = batch
        self.lap_floor_s = lap_floor_s
        self.seed = seed
        self.device = torch.device(device) if device is not None else data.train_x.device
        self.models: dict[str, nn.Module] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        for h in stable:
            torch.manual_seed(seed + zlib.crc32(h.key.encode()))
            m = h.build().to(self.device)
            self.models[h.key] = m
            self.opts[h.key] = torch.optim.Adam(m.parameters(), lr=lr)
        self.events: list[LapEvent] = []

    def run(self) -> Iterator[LapEvent]:
        t0 = time.perf_counter()
        n = self.data.n_train
        aug_gen = torch.Generator(device=self.device)
        for lap in range(1, self.laps + 1):
            lap_start = time.perf_counter()
            order = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed * 1000 + lap)).to(self.device)
            for i, h in enumerate(self.stable):
                aug_gen.manual_seed(self.seed * 100_000 + lap * 100 + i)
                ev = self._train_one_lap(h.key, order, aug_gen, lap, t0)
                self.events.append(ev)
                yield ev
            if lap < self.laps:
                remaining = self.lap_floor_s - (time.perf_counter() - lap_start)
                if remaining > 0:
                    time.sleep(remaining)

    def _train_one_lap(self, key: str, order: torch.Tensor, gen: torch.Generator, lap: int, t0: float) -> LapEvent:
        m, opt = self.models[key], self.opts[key]
        m.train()
        start = time.perf_counter()
        n = self.data.n_train
        correct = 0
        loss_sum = 0.0
        for i in range(0, n, self.batch):
            idx = order[i:i + self.batch]
            x = normalize(augment(self.data.train_x[idx], gen))
            y = self.data.train_y[idx]
            logits = m(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += int((logits.argmax(1) == y).sum().item())
            loss_sum += float(loss.item()) * int(idx.shape[0])
        test = self.evaluate(key)
        now = time.perf_counter()
        return LapEvent(key, lap, correct / n, test["all"], test["internet"], test["live"],
                        loss_sum / n, (now - start) * 1000.0, now - t0)

    @torch.no_grad()
    def evaluate(self, key: str) -> dict:
        m = self.models[key]
        m.eval()
        preds = []
        for i in range(0, self.data.n_test, 256):
            x = normalize(to_float(self.data.test_x[i:i + 256]))
            preds.append(m(x).argmax(1))
        hit = torch.cat(preds) == self.data.test_y
        live = self.data.test_is_live

        def acc(mask):
            return float(hit[mask].float().mean().item()) if bool(mask.any()) else None

        return {"all": float(hit.float().mean().item()), "internet": acc(~live), "live": acc(live)}
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_trainer.py`
Expected: `5 passed` (the pacing test takes about a second).

- [ ] **Step 5: Smoke-race on the GPU with real data and record lap times**

Run:

```powershell
python -c "import torch, time; from derby.data import backup, dataset; from derby.horses import stable; from derby.race.trainer import Race; L = backup.load(); d = dataset.assemble(L, None, None, seed=1, device='cuda'); r = Race(stable.enabled_stable(), d, laps=3, lap_floor_s=0.0, device='cuda'); [print(f'{e.horse:14s} lap {e.lap} train {e.train_acc:.2f} test {e.test_acc:.2f} {e.ms:7.0f} ms') for e in r.run()]"
```

Expected: 21 lines; every horse's `ms` per lap should be well under 4000 (a lap floor of 4 s). Accuracy should climb above chance (0.17) by lap 3 for most horses. Write the slowest horse's ms into the commit message. If any horse is slower than 4000 ms per lap, halve its widths in `stable.py` (keep the block type) and re-run.

- [ ] **Step 6: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/race neural-derby/tests/test_trainer.py
git commit -m "Add the lock-step race trainer with paced laps and lap events"
```

---

### Task 8: Scoring and recording

**Files:**
- Create: `neural-derby/derby/race/scoring.py`, `neural-derby/derby/race/recorder.py`, `neural-derby/tests/test_scoring.py`, `neural-derby/tests/test_recorder.py`

**Interfaces:**
- Consumes: `LapEvent`.
- Produces in `scoring`: `Standing(horse, final_acc, lap_reached, rank)`, `race_standings(events) -> list[Standing]`; `SprintStanding(horse, ms, rank, device)`, `sprint_standings(ms_by_horse: dict[str, float], device: str = "phone") -> list[SprintStanding]`; `ChampionStanding(horse, race_rank, sprint_rank, total, final_acc, rank)`, `champion_table(race, sprint) -> list[ChampionStanding]` (a horse missing from the sprint gets sprint rank `len(race)+1`); `BetResult(champion_winners: list[str], race_winners: list[str])`, `settle_bets(bets: dict[str, str], champion_key: str, race_winner_key: str) -> BetResult`; `race_winner(standings) -> str`.
- Produces in `recorder`: `write_race(path, events, race, sprint, champion, verdicts: dict, meta: dict) -> Path`; `read_race(path) -> dict`; `replay(path, speed: float = 1.0) -> Iterator[LapEvent]` (sleeps the recorded gaps divided by `speed`); `form_guide(paths: list[Path]) -> dict[str, dict]` with keys per horse `race_wins, champion_wins, races`; `latest_race(folder) -> Path | None`.
- All dataclasses have `to_dict()` via `dataclasses.asdict`.

- [ ] **Step 1: Write the failing scoring tests**

`neural-derby/tests/test_scoring.py`:

```python
from derby.race import scoring
from derby.race.trainer import LapEvent


def ev(horse, lap, test_acc, train_acc=None):
    return LapEvent(horse, lap, train_acc if train_acc is not None else test_acc, test_acc, test_acc, None, 1.0, 10.0, float(lap))


def test_race_standings_order_and_tiebreak_by_earliest_lap():
    events = [
        ev("a", 1, 0.5), ev("a", 2, 0.9), ev("a", 3, 0.9),
        ev("b", 1, 0.9), ev("b", 2, 0.8), ev("b", 3, 0.9),   # same final, reached 0.9 on lap 1
        ev("c", 1, 0.2), ev("c", 2, 0.3), ev("c", 3, 0.4),
    ]
    st = scoring.race_standings(events)
    assert [s.horse for s in st] == ["b", "a", "c"]
    assert [s.rank for s in st] == [1, 2, 3]
    assert st[0].lap_reached == 1 and st[1].lap_reached == 2
    assert scoring.race_winner(st) == "b"


def test_sprint_standings_fastest_first():
    st = scoring.sprint_standings({"a": 30.0, "b": 12.5, "c": 12.5}, device="phone")
    assert [(s.horse, s.rank) for s in st] == [("b", 1), ("c", 2), ("a", 3)]
    assert st[0].device == "phone"


def test_champion_is_lowest_rank_sum_with_accuracy_tiebreak():
    race = scoring.race_standings([ev("a", 1, 0.9), ev("b", 1, 0.8), ev("c", 1, 0.7)])
    sprint = scoring.sprint_standings({"a": 30.0, "b": 10.0, "c": 20.0})
    champ = scoring.champion_table(race, sprint)
    # a: 1+3=4, b: 2+1=3, c: 3+2=5
    assert [(c.horse, c.total, c.rank) for c in champ] == [("b", 3, 1), ("a", 4, 2), ("c", 5, 3)]


def test_missing_sprint_time_gets_last_sprint_rank():
    race = scoring.race_standings([ev("a", 1, 0.9), ev("b", 1, 0.8)])
    sprint = scoring.sprint_standings({"a": 30.0})
    champ = scoring.champion_table(race, sprint)
    assert {c.horse: c.sprint_rank for c in champ} == {"a": 1, "b": 3}


def test_settle_bets():
    res = scoring.settle_bets({"d1": "a", "d2": "b", "d3": "a"}, champion_key="a", race_winner_key="b")
    assert res.champion_winners == ["d1", "d3"] and res.race_winners == ["d2"]
```

- [ ] **Step 2: Write the failing recorder tests**

`neural-derby/tests/test_recorder.py`:

```python
import time
from pathlib import Path

from derby.race import recorder, scoring
from derby.race.trainer import LapEvent


def events():
    return [
        LapEvent("a", 1, 0.5, 0.4, 0.4, None, 1.0, 5.0, 0.00),
        LapEvent("b", 1, 0.5, 0.6, 0.6, None, 1.0, 5.0, 0.05),
        LapEvent("a", 2, 0.7, 0.7, 0.7, None, 0.8, 5.0, 0.10),
        LapEvent("b", 2, 0.7, 0.6, 0.6, None, 0.8, 5.0, 0.15),
    ]


def test_write_read_round_trip(tmp_path: Path):
    evs = events()
    race = scoring.race_standings(evs)
    sprint = scoring.sprint_standings({"a": 10.0, "b": 20.0})
    champ = scoring.champion_table(race, sprint)
    p = recorder.write_race(tmp_path / "r.json", evs, race, sprint, champ, {"a": ["fast"]}, {"seed": 1})
    doc = recorder.read_race(p)
    assert doc["meta"]["seed"] == 1
    assert doc["events"][0]["horse"] == "a" and len(doc["events"]) == 4
    assert doc["race"][0]["horse"] == "a" and doc["champion"][0]["rank"] == 1
    assert doc["verdicts"]["a"] == ["fast"]


def test_replay_reproduces_events_and_respects_speed(tmp_path: Path):
    evs = events()
    p = recorder.write_race(tmp_path / "r.json", evs, [], [], [], {}, {})
    start = time.perf_counter()
    out = list(recorder.replay(p, speed=1.0))
    elapsed = time.perf_counter() - start
    assert out == evs
    assert elapsed >= 0.15 - 0.02


def test_form_guide_counts_race_and_champion_wins(tmp_path: Path):
    evs = events()
    race = scoring.race_standings(evs)                      # a wins (0.7 vs 0.6)
    sprint = scoring.sprint_standings({"a": 50.0, "b": 10.0})
    champ = scoring.champion_table(race, sprint)            # a: 1+2=3, b: 2+1=3 -> tie, a has higher acc -> a
    p1 = recorder.write_race(tmp_path / "1.json", evs, race, sprint, champ, {}, {})
    p2 = recorder.write_race(tmp_path / "2.json", evs, race, sprint, champ, {}, {})
    fg = recorder.form_guide([p1, p2])
    assert fg["a"] == {"race_wins": 2, "champion_wins": 2, "races": 2}
    assert fg["b"] == {"race_wins": 0, "champion_wins": 0, "races": 2}
    assert recorder.latest_race(tmp_path) == p2
```

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/test_scoring.py tests/test_recorder.py`
Expected: import errors.

- [ ] **Step 4: Write scoring.py**

`neural-derby/derby/race/scoring.py`:

```python
"""Standings, the champion rule, and bet settlement. Pure functions, no I/O."""
from collections import defaultdict
from dataclasses import asdict, dataclass

from derby.race.trainer import LapEvent


@dataclass(frozen=True)
class Standing:
    horse: str
    final_acc: float
    lap_reached: int      # first lap at which the horse was already at its final accuracy
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SprintStanding:
    horse: str
    ms: float
    rank: int
    device: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChampionStanding:
    horse: str
    race_rank: int
    sprint_rank: int
    total: int
    final_acc: float
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BetResult:
    champion_winners: list[str]
    race_winners: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def race_standings(events: list[LapEvent]) -> list[Standing]:
    by: dict[str, list[LapEvent]] = defaultdict(list)
    for e in events:
        by[e.horse].append(e)
    rows = []
    for horse, evs in by.items():
        evs = sorted(evs, key=lambda e: e.lap)
        final = evs[-1].test_acc
        lap_reached = min(e.lap for e in evs if e.test_acc >= final - 1e-9)
        rows.append((horse, final, lap_reached))
    rows.sort(key=lambda r: (-r[1], r[2], r[0]))
    return [Standing(h, f, l, i + 1) for i, (h, f, l) in enumerate(rows)]


def race_winner(standings: list[Standing]) -> str:
    return standings[0].horse


def sprint_standings(ms_by_horse: dict[str, float], device: str = "phone") -> list[SprintStanding]:
    rows = sorted(ms_by_horse.items(), key=lambda kv: (kv[1], kv[0]))
    return [SprintStanding(h, ms, i + 1, device) for i, (h, ms) in enumerate(rows)]


def champion_table(race: list[Standing], sprint: list[SprintStanding]) -> list[ChampionStanding]:
    sprint_rank = {s.horse: s.rank for s in sprint}
    missing = len(race) + 1
    rows = []
    for r in race:
        sr = sprint_rank.get(r.horse, missing)
        rows.append((r.horse, r.rank, sr, r.rank + sr, r.final_acc))
    rows.sort(key=lambda x: (x[3], -x[4], x[0]))
    return [ChampionStanding(h, rr, sr, tot, acc, i + 1) for i, (h, rr, sr, tot, acc) in enumerate(rows)]


def settle_bets(bets: dict[str, str], champion_key: str, race_winner_key: str) -> BetResult:
    champion_winners = sorted(d for d, h in bets.items() if h == champion_key)
    race_winners = sorted(d for d, h in bets.items() if h == race_winner_key)
    return BetResult(champion_winners, race_winners)
```

- [ ] **Step 5: Write recorder.py**

`neural-derby/derby/race/recorder.py`:

```python
"""Every race is written to a JSON file: for replay if the GPU dies, and for the form guide (betting odds)."""
import json
import time
from pathlib import Path
from typing import Iterator

from derby.race.trainer import LapEvent


def write_race(path: Path, events: list[LapEvent], race, sprint, champion, verdicts: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "meta": meta,
        "events": [e.to_dict() for e in events],
        "race": [s.to_dict() for s in race],
        "sprint": [s.to_dict() for s in sprint],
        "champion": [c.to_dict() for c in champion],
        "verdicts": verdicts,
    }
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def read_race(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay(path: Path, speed: float = 1.0) -> Iterator[LapEvent]:
    events = [LapEvent(**e) for e in read_race(path)["events"]]
    prev_t = 0.0
    for e in sorted(events, key=lambda e: e.t):
        gap = (e.t - prev_t) / max(speed, 1e-6)
        if gap > 0:
            time.sleep(gap)
        prev_t = e.t
        yield e


def form_guide(paths: list[Path]) -> dict[str, dict]:
    guide: dict[str, dict] = {}
    for p in paths:
        doc = read_race(p)
        horses = {s["horse"] for s in doc["race"]}
        race_win = next((s["horse"] for s in doc["race"] if s["rank"] == 1), None)
        champ_win = next((c["horse"] for c in doc["champion"] if c["rank"] == 1), None)
        for h in horses:
            g = guide.setdefault(h, {"race_wins": 0, "champion_wins": 0, "races": 0})
            g["races"] += 1
            g["race_wins"] += int(h == race_win)
            g["champion_wins"] += int(h == champ_win)
    return guide


def latest_race(folder: Path) -> Path | None:
    files = sorted(Path(folder).glob("*.json"))
    return files[-1] if files else None
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_scoring.py tests/test_recorder.py`
Expected: `8 passed`

- [ ] **Step 7: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/race/scoring.py neural-derby/derby/race/recorder.py neural-derby/tests/test_scoring.py neural-derby/tests/test_recorder.py
git commit -m "Add race scoring, champion rule, bet settlement, and race recording with replay"
```

---
### Task 9: Caller and verdicts

**Files:**
- Create: `neural-derby/derby/commentary/__init__.py`, `neural-derby/derby/commentary/caller.py`, `neural-derby/derby/commentary/verdict.py`, `neural-derby/tests/test_caller.py`, `neural-derby/tests/test_verdict.py`

**Interfaces:**
- Consumes: `LapEvent`, `scoring.race_standings`, `Standing`, `SprintStanding`, `ChampionStanding`.
- Produces in `caller`: `Caller(horses: list[str], names: dict[str, str], laps_total: int, seed: int | None = None)` with `on_event(ev: LapEvent) -> list[str]`; `pct(fraction) -> str` like `"87 percent"`.
- Produces in `verdict`: `Facts` dataclass `(horse, params, macs, depth, final_acc, acc_internet, acc_live, convergence_lap, max_gap, sprint_ms, race_rank, sprint_rank, champion_rank)` with `to_dict()`; `compute_facts(events, stats_by_horse: dict[str, dict], race, sprint, champion) -> dict[str, Facts]`; `verdicts(facts: dict[str, Facts], names: dict[str, str], n_train: int) -> dict[str, list[str]]`; thresholds `LAP_GAP = 3`, `MEMORISE_GAP = 0.20`, `ROOM_GAP = 0.10`.

- [ ] **Step 1: Write the failing caller tests**

`neural-derby/tests/test_caller.py`:

```python
from derby.commentary.caller import Caller, pct
from derby.race.trainer import LapEvent

NAMES = {"a": "ALPHA", "b": "BRAVO"}


def ev(horse, lap, test_acc, train_acc=None):
    return LapEvent(horse, lap, train_acc if train_acc is not None else test_acc, test_acc, test_acc, None, 1.0, 1.0, float(lap))


def feed(caller, stream):
    out = []
    for e in stream:
        out.append(caller.on_event(e))
    return out


def joined(lines_per_event):
    return " | ".join(" ".join(ls) for ls in lines_per_event)


def test_pct():
    assert pct(0.874) == "87 percent"


def test_start_first_leader_and_lead_change():
    c = Caller(["a", "b"], NAMES, laps_total=4, seed=0)
    out = feed(c, [ev("a", 1, 0.5), ev("b", 1, 0.6), ev("a", 2, 0.8), ev("b", 2, 0.7)])
    text = joined(out)
    assert "off" in text.lower()                       # start line
    assert "BRAVO" in " ".join(out[1]) and "lead" in " ".join(out[1]).lower()
    assert "ALPHA" in " ".join(out[3]) and "overtake" in " ".join(out[3]).lower()


def test_surge_and_stall_and_memorising_fire_on_thresholds():
    c = Caller(["a"], NAMES, laps_total=10, seed=0)
    out = feed(c, [ev("a", 1, 0.30), ev("a", 2, 0.40), ev("a", 3, 0.405), ev("a", 4, 0.408), ev("a", 5, 0.60, train_acc=0.95)])
    assert "surge" in " ".join(out[1]).lower()         # +10 points
    assert "wall" in " ".join(out[3]).lower()          # laps 2,3,4 within 1 point
    assert "memoris" in " ".join(out[4]).lower()       # train 95 vs test 60


def test_no_false_alarms_below_thresholds():
    c = Caller(["a"], NAMES, laps_total=10, seed=0)
    out = feed(c, [ev("a", 1, 0.30), ev("a", 2, 0.35), ev("a", 3, 0.42, train_acc=0.55)])
    text = joined(out).lower()
    assert "surge" not in text and "wall" not in text and "memoris" not in text


def test_halfway_final_lap_and_photo_finish():
    c = Caller(["a", "b"], NAMES, laps_total=4, seed=0)
    out = feed(c, [
        ev("a", 1, 0.5), ev("b", 1, 0.4),
        ev("a", 2, 0.6), ev("b", 2, 0.5),      # lap 2 = halfway
        ev("a", 3, 0.7), ev("b", 3, 0.69),     # lap 3 = final lap announcement
        ev("a", 4, 0.80), ev("b", 4, 0.79),    # finish, margin 1 point -> photo finish
    ])
    assert "halfway" in " ".join(out[3]).lower()
    assert "last lap" in " ".join(out[5]).lower()
    last = " ".join(out[7]).lower()
    assert "wins" in last and "photo finish" in last
```

- [ ] **Step 2: Write the failing verdict tests**

`neural-derby/tests/test_verdict.py`:

```python
from derby.commentary import verdict as V
from derby.race import scoring
from derby.race.trainer import LapEvent

NAMES = {"stairs": "STAIRS", "elevator": "ELEVATOR", "textbook": "TEXTBOOK", "featherweight": "FEATHERWEIGHT"}
STATS = {k: {"params": 1000 * (i + 1), "macs": 10_000 * (i + 1), "depth": 4} for i, k in enumerate(NAMES)}
STATS["featherweight"]["macs"] = 2_000


def ev(h, lap, test, train=None, live=None):
    return LapEvent(h, lap, train if train is not None else test, test, test, live, 1.0, 1.0, float(lap))


def curve(h, accs, trains=None):
    trains = trains or accs
    return [ev(h, i + 1, a, t) for i, (a, t) in enumerate(zip(accs, trains))]


def build(events, ms):
    race = scoring.race_standings(events)
    sprint = scoring.sprint_standings(ms)
    champ = scoring.champion_table(race, sprint)
    return V.compute_facts(events, STATS, race, sprint, champ)


def test_facts_convergence_lap_and_gap():
    events = curve("textbook", [0.2, 0.5, 0.85, 0.9, 0.9], trains=[0.2, 0.5, 0.9, 0.95, 0.99])
    f = build(events, {"textbook": 5.0})["textbook"]
    assert f.final_acc == 0.9 and f.convergence_lap == 3     # 0.85 >= 0.9 * 0.9
    assert abs(f.max_gap - 0.09) < 1e-9
    assert f.race_rank == 1 and f.sprint_rank == 1 and f.champion_rank == 1 and f.sprint_ms == 5.0


def test_slow_start_claim_only_when_lap_gap_is_three_or_more():
    slow = curve("stairs", [0.2, 0.3, 0.4, 0.6, 0.8, 0.9]) + curve("elevator", [0.6, 0.85, 0.9, 0.9, 0.9, 0.9])
    v = V.verdicts(build(slow, {"stairs": 10, "elevator": 10}), NAMES, n_train=600)
    assert any("stride" in s for s in v["stairs"])
    close = curve("stairs", [0.6, 0.8, 0.9, 0.9]) + curve("elevator", [0.6, 0.85, 0.9, 0.9])
    v2 = V.verdicts(build(close, {"stairs": 10, "elevator": 10}), NAMES, n_train=600)
    assert not any("stride" in s for s in v2["stairs"])
    assert any("did not change the pace" in s for s in v2["stairs"])


def test_memorising_claim_only_at_twenty_points():
    big = curve("textbook", [0.5, 0.6], trains=[0.5, 0.85])
    small = curve("textbook", [0.5, 0.6], trains=[0.5, 0.75])
    assert any("memoris" in s for s in V.verdicts(build(big, {"textbook": 5}), NAMES, 600)["textbook"])
    assert not any("memoris" in s for s in V.verdicts(build(small, {"textbook": 5}), NAMES, 600)["textbook"])


def test_featherweight_speed_claim_only_if_actually_faster():
    events = curve("featherweight", [0.7, 0.8]) + curve("textbook", [0.7, 0.85])
    fast = V.verdicts(build(events, {"featherweight": 4.0, "textbook": 9.0}), NAMES, 600)["featherweight"]
    assert any("That is why" in s for s in fast)
    slow = V.verdicts(build(events, {"featherweight": 9.0, "textbook": 4.0}), NAMES, 600)["featherweight"]
    assert any("not faster" in s for s in slow)


def test_room_versus_internet_sentence():
    events = [ev("textbook", 1, 0.8, live=0.6)]
    v = V.verdicts(build(events, {"textbook": 5}), NAMES, 600)["textbook"]
    assert any("room" in s.lower() and "internet" in s.lower() for s in v)
```

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/test_caller.py tests/test_verdict.py`
Expected: import errors.

- [ ] **Step 4: Write caller.py**

`neural-derby/derby/commentary/__init__.py`: empty.

`neural-derby/derby/commentary/caller.py`:

```python
"""The race-caller. Every line comes from a rule on real numbers; only the wording is random."""
import random

from derby.race.scoring import race_standings
from derby.race.trainer import LapEvent

SURGE_POINTS = 0.08
STALL_POINTS = 0.01
MEMORISE_POINTS = 0.20
PHOTO_FINISH_POINTS = 0.02

# Every phrasing of a rule contains that rule's keyword (lead, overtakes, surges, wall, memorising, halfway,
# last lap, wins, photo finish) so the ticker, the voice, and the tests all agree on what fired.
START = ["And they're off!", "The gates open. They're off!", "Seven brains, one dataset. They're off!"]
FIRST_LEADER = ["{h} takes the early lead at {acc}.", "First out of the gate: {h} leads at {acc}."]
LEAD_CHANGE = ["{y} overtakes {x}!", "New leader! {y} overtakes {x}.", "{x} loses the lead as {y} overtakes!"]
SURGE = ["{h} surges forward, up to {acc}!", "Big lap for {h}, it surges to {acc}!", "{h} finds another gear and surges to {acc}."]
STALL = ["{h} has hit a wall at {acc}.", "{h} is stuck at a wall, {acc}. Three laps, no progress.", "Nothing new from {h}, a wall at {acc}."]
MEMORISING = ["{h} is memorising the photos instead of learning. {train} on photos it has seen, {test} on new ones.",
              "Careful, {h}: {train} in training but only {test} on new photos. That is memorising."]
HALFWAY = ["Halfway there. {a} leads at {acc}, {b} second, {c} third.", "Halfway, lap {lap} of {total}: {a} in front at {acc}, then {b} and {c}."]
HALFWAY_TWO = ["Halfway there. {a} leads at {acc}, {b} chasing.", "Halfway, lap {lap} of {total}: {a} in front at {acc}, {b} behind."]
FINAL_LAP = ["Last lap! {a} by {margin} over {b}.", "Last lap. {a} leads {b} by {margin}."]
FINISH = ["{a} wins the Derby at {acc}!", "It's {a}! {a} wins the Derby at {acc}!"]
PHOTO = ["Photo finish! {a} by just {margin} over {b}.", "Too close to call... photo finish says {a} by {margin}!"]
CLEAR = ["{b} second at {bacc}."]


def pct(fraction: float) -> str:
    return f"{round(fraction * 100)} percent"


def points(diff: float) -> str:
    p = round(diff * 100)
    return f"{p} point" if p == 1 else f"{p} points"


class Caller:
    def __init__(self, horses: list[str], names: dict[str, str], laps_total: int, seed: int | None = None):
        self.horses = list(horses)
        self.names = names
        self.laps_total = laps_total
        self.rng = random.Random(seed)
        self.history: dict[str, dict[int, LapEvent]] = {h: {} for h in horses}
        self.leader: str | None = None
        self.started = False
        self.memorised: set[str] = set()
        self.last_stall_lap: dict[str, int] = {}

    def name(self, key: str) -> str:
        return self.names.get(key, key.upper())

    def say(self, options: list[str], **kw) -> str:
        return self.rng.choice(options).format(**kw)

    def on_event(self, ev: LapEvent) -> list[str]:
        lines: list[str] = []
        if not self.started:
            self.started = True
            lines.append(self.say(START))
        self.history.setdefault(ev.horse, {})[ev.lap] = ev
        lines += self._horse_rules(ev)
        if all(ev.lap in self.history.get(h, {}) for h in self.horses):
            lines += self._lap_rules(ev.lap)
        return lines

    def _horse_rules(self, ev: LapEvent) -> list[str]:
        lines = []
        h = self.name(ev.horse)
        hist = self.history[ev.horse]
        prev = hist.get(ev.lap - 1)
        if prev is not None and ev.test_acc - prev.test_acc >= SURGE_POINTS:
            lines.append(self.say(SURGE, h=h, acc=pct(ev.test_acc)))
        three = [hist.get(lap) for lap in (ev.lap - 2, ev.lap - 1, ev.lap)]
        if all(e is not None for e in three) and ev.lap < self.laps_total:
            accs = [e.test_acc for e in three]
            recently = ev.lap - self.last_stall_lap.get(ev.horse, -10) < 3
            if max(accs) - min(accs) < STALL_POINTS and not recently:
                self.last_stall_lap[ev.horse] = ev.lap
                lines.append(self.say(STALL, h=h, acc=pct(ev.test_acc)))
        if ev.train_acc - ev.test_acc >= MEMORISE_POINTS and ev.horse not in self.memorised:
            self.memorised.add(ev.horse)
            lines.append(self.say(MEMORISING, h=h, train=pct(ev.train_acc), test=pct(ev.test_acc)))
        return lines

    def _lap_rules(self, lap: int) -> list[str]:
        lines = []
        events = [e for h in self.horses for e in self.history[h].values() if e.lap <= lap]
        st = race_standings(events)
        leader = st[0].horse
        if self.leader is None:
            lines.append(self.say(FIRST_LEADER, h=self.name(leader), acc=pct(st[0].final_acc)))
        elif leader != self.leader:
            lines.append(self.say(LEAD_CHANGE, y=self.name(leader), x=self.name(self.leader)))
        self.leader = leader
        second = st[1] if len(st) > 1 else None
        margin = (st[0].final_acc - second.final_acc) if second else 0.0
        if lap == self.laps_total // 2 and lap not in (self.laps_total, self.laps_total - 1):
            if len(st) >= 3:
                lines.append(self.say(HALFWAY, a=self.name(st[0].horse), acc=pct(st[0].final_acc),
                                      b=self.name(st[1].horse), c=self.name(st[2].horse), lap=lap, total=self.laps_total))
            elif second:
                lines.append(self.say(HALFWAY_TWO, a=self.name(st[0].horse), acc=pct(st[0].final_acc),
                                      b=self.name(second.horse), lap=lap, total=self.laps_total))
        if lap == self.laps_total - 1 and second:
            lines.append(self.say(FINAL_LAP, a=self.name(leader), b=self.name(second.horse), margin=points(margin)))
        if lap == self.laps_total:
            lines.append(self.say(FINISH, a=self.name(leader), acc=pct(st[0].final_acc)))
            if second:
                if margin < PHOTO_FINISH_POINTS:
                    lines.append(self.say(PHOTO, a=self.name(leader), b=self.name(second.horse), margin=points(margin)))
                else:
                    lines.append(self.say(CLEAR, b=self.name(second.horse), bacc=pct(second.final_acc)))
        return lines
```

- [ ] **Step 5: Write verdict.py**

`neural-derby/derby/commentary/verdict.py`:

```python
"""Why each horse finished where it did, said only when the numbers back it up."""
from collections import defaultdict
from dataclasses import asdict, dataclass

from derby.race.scoring import ChampionStanding, SprintStanding, Standing
from derby.race.trainer import LapEvent

LAP_GAP = 3            # laps of difference before we call a slow start
MEMORISE_GAP = 0.20    # train minus test before we call memorising
ROOM_GAP = 0.10        # internet minus room accuracy before we call the room's photos harder
CONVERGED_FRAC = 0.9   # a horse "reached its stride" at 90 percent of its own final accuracy


@dataclass(frozen=True)
class Facts:
    horse: str
    params: int
    macs: int
    depth: int
    final_acc: float
    acc_internet: float | None
    acc_live: float | None
    convergence_lap: int
    max_gap: float
    sprint_ms: float | None
    race_rank: int
    sprint_rank: int
    champion_rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def pct(x: float) -> str:
    return f"{round(x * 100)} percent"


def compute_facts(events: list[LapEvent], stats_by_horse: dict[str, dict], race: list[Standing],
                  sprint: list[SprintStanding], champion: list[ChampionStanding]) -> dict[str, Facts]:
    by: dict[str, list[LapEvent]] = defaultdict(list)
    for e in events:
        by[e.horse].append(e)
    race_rank = {s.horse: s.rank for s in race}
    sprint_rank = {s.horse: s.rank for s in sprint}
    sprint_ms = {s.horse: s.ms for s in sprint}
    champ_rank = {c.horse: c.rank for c in champion}
    out = {}
    for horse, evs in by.items():
        evs = sorted(evs, key=lambda e: e.lap)
        last = evs[-1]
        target = last.test_acc * CONVERGED_FRAC
        convergence_lap = next(e.lap for e in evs if e.test_acc >= target - 1e-9)
        max_gap = max(e.train_acc - e.test_acc for e in evs)
        st = stats_by_horse.get(horse, {"params": 0, "macs": 0, "depth": 0})
        out[horse] = Facts(
            horse, st["params"], st["macs"], st["depth"], last.test_acc, last.test_acc_internet, last.test_acc_live,
            convergence_lap, max_gap, sprint_ms.get(horse), race_rank.get(horse, 0),
            sprint_rank.get(horse, len(race) + 1), champ_rank.get(horse, 0),
        )
    return out


def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def verdicts(facts: dict[str, Facts], names: dict[str, str], n_train: int) -> dict[str, list[str]]:
    def nm(k):
        return names.get(k, k.upper())

    out: dict[str, list[str]] = {}
    tb = facts.get("textbook")
    for key, f in facts.items():
        lines: list[str] = []
        ms = f"{f.sprint_ms:.0f} ms" if f.sprint_ms is not None else "no time"
        lines.append(f"Finished {ordinal(f.race_rank)} in the Derby at {pct(f.final_acc)}, "
                     f"{ordinal(f.sprint_rank)} in the Sprint ({ms}), {ordinal(f.champion_rank)} overall.")
        lines.append(f"{f.params:,} parameters, {f.macs:,} multiplies per photo, {f.depth} weight layers.")

        if key == "stairs" and "elevator" in facts:
            e = facts["elevator"]
            if f.convergence_lap - e.convergence_lap >= LAP_GAP:
                lines.append(f"{nm(key)} reached its stride on lap {f.convergence_lap}; {nm('elevator')} got there on lap "
                             f"{e.convergence_lap}. Same depth, one shortcut.")
            else:
                lines.append(f"At this depth the shortcuts did not change the pace (lap {f.convergence_lap} vs lap "
                             f"{e.convergence_lap}). Deeper buildings need elevators more.")
        if key == "elevator" and "stairs" in facts:
            s = facts["stairs"]
            if s.convergence_lap - f.convergence_lap >= LAP_GAP:
                lines.append(f"The shortcuts paid off: stride on lap {f.convergence_lap} against {nm('stairs')}'s lap "
                             f"{s.convergence_lap}, with exactly the same parameters.")
            else:
                lines.append(f"Same parameters as {nm('stairs')} and about the same pace this time "
                             f"(lap {f.convergence_lap} vs lap {s.convergence_lap}).")
        if key == "featherweight" and tb is not None:
            if f.sprint_ms is not None and tb.sprint_ms is not None and f.sprint_ms < tb.sprint_ms:
                lines.append(f"{f.macs:,} multiplies against {nm('textbook')}'s {tb.macs:,}. That is why it ran the Sprint in "
                             f"{f.sprint_ms:.0f} ms against {tb.sprint_ms:.0f} ms.")
            elif f.sprint_ms is not None and tb.sprint_ms is not None:
                lines.append(f"{f.macs:,} multiplies against {nm('textbook')}'s {tb.macs:,}, yet it was not faster on this device "
                             f"({f.sprint_ms:.0f} ms vs {tb.sprint_ms:.0f} ms). Fewer multiplies only help when the runtime "
                             f"is tuned for depthwise layers.")
        if key == "spotlight" and tb is not None:
            delta = f.final_acc - tb.final_acc
            sign = "+" if delta >= 0 else "-"
            lines.append(f"Attention added {f.params - tb.params:,} parameters over {nm('textbook')} and changed accuracy by "
                         f"{sign}{abs(round(delta * 100))} points.")
        if key == "trifocals" and tb is not None:
            lines.append(f"Three lens sizes cost {f.macs:,} multiplies against {nm('textbook')}'s {tb.macs:,}; "
                         f"accuracy {pct(f.final_acc)} against {pct(tb.final_acc)}.")
        if key == "influencer":
            lines.append(f"{pct(f.final_acc)} with {n_train:,} photos. Transformers have no built-in idea that nearby pixels "
                         f"belong together; they must learn it from data, and there was not enough.")
        if f.max_gap >= MEMORISE_GAP:
            lines.append(f"It memorised: at its worst, {pct(f.max_gap)} more on photos it had seen than on new ones.")
        if f.acc_internet is not None and f.acc_live is not None:
            line = f"On internet photos {pct(f.acc_internet)}; on the room's photos {pct(f.acc_live)}."
            if f.acc_internet - f.acc_live >= ROOM_GAP:
                line += " The room's photos are harder: different light, different backgrounds."
            lines.append(line)
        out[key] = lines
    return out
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_caller.py tests/test_verdict.py`
Expected: `10 passed`

- [ ] **Step 7: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/commentary neural-derby/tests/test_caller.py neural-derby/tests/test_verdict.py
git commit -m "Add the race-caller rules and the number-backed verdict engine"
```

---

### Task 10: ONNX export and laptop bench

**Files:**
- Create: `neural-derby/derby/export/__init__.py`, `neural-derby/derby/export/onnx_export.py`, `neural-derby/derby/export/bench.py`, `neural-derby/tests/test_export.py`

**Interfaces:**
- Produces: `export_onnx(model: nn.Module, path: Path, img: int = 96) -> Path` (copies the model to CPU in eval mode, exports opset 17 with input `image` of shape `batch x 3 x 96 x 96` and output `logits`, then checks onnxruntime matches torch within 1e-4; raises `ExportMismatch` otherwise); `bench_cpu(path: Path, n: int = 30, img: int = 96) -> float` (median milliseconds for one image).

- [ ] **Step 1: Write the failing tests**

`neural-derby/tests/test_export.py`:

```python
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from derby.export.bench import bench_cpu
from derby.export.onnx_export import export_onnx
from derby.horses import stable


def test_every_horse_exports_and_matches(tmp_path: Path):
    x = torch.randn(2, 3, 96, 96)
    for h in stable.STABLE:
        m = h.build().eval()
        p = export_onnx(m, tmp_path / f"{h.key}.onnx")
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        out = sess.run(["logits"], {"image": x.numpy()})[0]
        with torch.no_grad():
            ref = m(x).numpy()
        assert np.abs(out - ref).max() < 1e-4, h.key
        assert sess.get_inputs()[0].name == "image"


def test_export_leaves_original_model_on_its_device_and_mode(tmp_path: Path):
    m = stable.build_textbook().train()
    export_onnx(m, tmp_path / "t.onnx")
    assert m.training


def test_bench_cpu_returns_positive_ms(tmp_path: Path):
    p = export_onnx(stable.build_featherweight(), tmp_path / "f.onnx")
    ms = bench_cpu(p, n=5)
    assert ms > 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_export.py`
Expected: `ModuleNotFoundError: derby.export`

- [ ] **Step 3: Write onnx_export.py and bench.py**

`neural-derby/derby/export/__init__.py`: empty.

`neural-derby/derby/export/onnx_export.py`:

```python
"""Turn a trained horse into an ONNX file the phone can run, and prove it gives the same answers."""
import copy
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from derby import config


class ExportMismatch(RuntimeError):
    pass


def export_onnx(model: nn.Module, path: Path, img: int = config.IMG) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, 3, img, img)
    kwargs = dict(input_names=["image"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(m, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(m, dummy, str(path), **kwargs)
    x = torch.randn(2, 3, img, img)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"image": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    if diff > 1e-4:
        raise ExportMismatch(f"onnx output differs from torch by {diff}")
    return path
```

`neural-derby/derby/export/bench.py`:

```python
"""Laptop-side timing, the fallback when the team phone is not reachable for the Sprint."""
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from derby import config


def bench_cpu(path: Path, n: int = 30, img: int = config.IMG) -> float:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # one core, like a phone browser without threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img, img).astype(np.float32)
    for _ in range(3):
        sess.run(["logits"], {"image": x})
    times = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(["logits"], {"image": x})
        times.append((time.perf_counter() - t) * 1000.0)
    return float(statistics.median(times))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_export.py`
Expected: `3 passed`. If THE INFLUENCER fails to export, the usual cause is `attention.softmax` on an older opset; it is fine at 17. If the legacy exporter warns about deprecation, that is expected.

- [ ] **Step 5: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/export neural-derby/tests/test_export.py
git commit -m "Add ONNX export with a torch match check and a CPU timing fallback"
```

---

### Task 11: Show state machine, websocket hub, and the server

**Files:**
- Create: `neural-derby/derby/server/__init__.py`, `neural-derby/derby/server/show.py`, `neural-derby/derby/server/hub.py`, `neural-derby/derby/server/app.py`, `neural-derby/tests/test_show.py`, `neural-derby/tests/test_app.py`

**Interfaces:**
- Consumes: everything from Tasks 3 to 10.
- Produces in `show`: `Act(str, Enum)` with values `LOBBY, PADDOCK, MULTIPLY, HORSES, DERBY, SPRINT, VERDICT, SHIP`; `GateError(Exception)`; `Show()` with fields `act, photos_open, bets_open, data_loaded`, methods `advance() -> Act` (raises `GateError` if at SHIP or if entering DERBY while `data_loaded` is False), `close_gate()`, `to_dict()`.
- Produces in `hub`: `Hub()` with `async connect(ws, role)`, `disconnect(ws)`, `async broadcast(msg: dict, roles: set[str] | None = None)`, `broadcast_threadsafe(loop, msg, roles=None)`, `count(role) -> int`.
- Produces in `app`: `Settings` dataclass `(data_dir, web_dir, laps, batch, lap_floor_s, seed, enabled, load_backup: bool, device: str | None)`; `AppState`; `create_app(settings: Settings | None = None) -> FastAPI` with `app.state.derby: AppState`; routes listed in Step 5; websocket message types `state, photo, previews, horses, lap, race_done, sprint_request, sprint, verdict, ship, error`.
- The room-code idea from the spec is replaced by a short path: phones can type `http://<ip>:8000/p`, which redirects to the phone page.

- [ ] **Step 1: Write the failing state-machine tests**

`neural-derby/tests/test_show.py`:

```python
import pytest

from derby.server.show import Act, GateError, Show


def test_acts_in_order_with_gates():
    s = Show()
    assert s.act == Act.LOBBY and not s.photos_open and not s.bets_open
    assert s.advance() == Act.PADDOCK and s.photos_open
    s.close_gate()
    assert not s.photos_open
    assert s.advance() == Act.MULTIPLY
    assert s.advance() == Act.HORSES and s.bets_open
    with pytest.raises(GateError):
        s.advance()                     # DERBY needs data_loaded
    s.data_loaded = True
    assert s.advance() == Act.DERBY and not s.bets_open
    assert s.advance() == Act.SPRINT
    assert s.advance() == Act.VERDICT
    assert s.advance() == Act.SHIP
    with pytest.raises(GateError):
        s.advance()


def test_to_dict():
    d = Show().to_dict()
    assert d == {"act": "LOBBY", "photos_open": False, "bets_open": False, "data_loaded": False}
```

- [ ] **Step 2: Write the failing app tests**

`neural-derby/tests/test_app.py`:

```python
import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from derby import config
from derby.data.backup import Loaded
from derby.server.app import Settings, create_app

# only two horses, so the CPU race in the tests stays fast; missing keys would default to enabled
TWO = {k: k in ("textbook", "featherweight") for k in config.HORSES_ENABLED}


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (30, 60, 90)).save(buf, "JPEG")
    return buf.getvalue()


def fake_backup(n=48):
    x = torch.randint(0, 256, (n, 3, 96, 96), dtype=torch.uint8)
    return Loaded(x, torch.arange(n) % 6, [None] * n)


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, laps=1, batch=16, lap_floor_s=0.0, seed=1,
                        enabled=TWO, load_backup=False, device="cpu")
    app = create_app(settings)
    app.state.derby.backup = fake_backup()
    with TestClient(app) as c:
        yield c


def test_state_and_join(client: TestClient):
    assert client.get("/api/state").json()["act"] == "LOBBY"
    r = client.post("/api/join", json={"device": "dev_0001", "nick": "Ranvir"})
    assert r.status_code == 200 and r.json()["ok"]


def test_photo_only_when_paddock_open_and_caps_apply(client: TestClient):
    files = {"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}
    r = client.post("/api/photo", data={"device": "dev_0001", "class_name": "pen"}, files=files)
    assert r.status_code == 400 and "closed" in r.json()["error"].lower()
    client.post("/api/advance")                       # PADDOCK
    for _ in range(3):
        r = client.post("/api/photo", data={"device": "dev_0001", "class_name": "pen"}, files=files)
        assert r.status_code == 200 and r.json()["thumb"]
    r = client.post("/api/photo", data={"device": "dev_0001", "class_name": "pen"}, files=files)
    assert r.status_code == 400 and "3 photos" in r.json()["error"]
    assert client.get("/api/state").json()["photos"] == 3


def test_bets_only_in_horses_act(client: TestClient):
    r = client.post("/api/bet", json={"device": "dev_0001", "horse": "textbook"})
    assert r.status_code == 400
    for _ in range(3):
        client.post("/api/advance")                   # PADDOCK, MULTIPLY, HORSES
    assert client.get("/api/state").json()["act"] == "HORSES"
    assert client.post("/api/bet", json={"device": "dev_0001", "horse": "textbook"}).status_code == 200
    assert client.post("/api/bet", json={"device": "dev_0001", "horse": "unicorn"}).status_code == 400


def test_full_run_broadcasts_lap_events_and_finishes(client: TestClient):
    with client.websocket_connect("/ws?role=stadium") as ws:
        first = ws.receive_json()
        assert first["type"] == "state"
        for _ in range(3):
            client.post("/api/advance")
            msg = ws.receive_json()
            while msg["type"] not in ("state",):
                msg = ws.receive_json()
        # HORSES: server loads data in the background; wait for data_loaded
        for _ in range(200):
            if client.get("/api/state").json()["data_loaded"]:
                break
            import time; time.sleep(0.05)
        assert client.post("/api/advance").status_code == 200      # DERBY
        seen = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "lap":
                seen.append(msg["event"]["horse"])
            if msg["type"] == "race_done":
                break
        assert seen == ["textbook", "featherweight"]
        assert client.get("/api/state").json()["act"] == "DERBY"
        assert client.get("/models/textbook.onnx").status_code == 200
```

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/test_show.py tests/test_app.py`
Expected: import errors.

- [ ] **Step 4: Write show.py and hub.py**

`neural-derby/derby/server/__init__.py`: empty.

`neural-derby/derby/server/show.py`:

```python
"""The six acts as a state machine. Space on the stadium page calls advance()."""
from enum import Enum


class Act(str, Enum):
    LOBBY = "LOBBY"
    PADDOCK = "PADDOCK"
    MULTIPLY = "MULTIPLY"
    HORSES = "HORSES"
    DERBY = "DERBY"
    SPRINT = "SPRINT"
    VERDICT = "VERDICT"
    SHIP = "SHIP"


ORDER = list(Act)


class GateError(Exception):
    pass


class Show:
    def __init__(self):
        self.act = Act.LOBBY
        self.photos_open = False
        self.bets_open = False
        self.data_loaded = False

    def advance(self) -> Act:
        i = ORDER.index(self.act)
        if i + 1 >= len(ORDER):
            raise GateError("The show is over.")
        nxt = ORDER[i + 1]
        if nxt == Act.DERBY and not self.data_loaded:
            raise GateError("Photos are still loading onto the GPU. Try again in a moment.")
        self.act = nxt
        self.photos_open = nxt == Act.PADDOCK
        self.bets_open = nxt == Act.HORSES
        return nxt

    def close_gate(self) -> None:
        self.photos_open = False
        self.bets_open = False

    def to_dict(self) -> dict:
        return {"act": self.act.value, "photos_open": self.photos_open,
                "bets_open": self.bets_open, "data_loaded": self.data_loaded}
```

`neural-derby/derby/server/hub.py`:

```python
"""Fan-out of JSON messages to every connected page. Roles: stadium, phone, team, control."""
import asyncio
import json

from fastapi import WebSocket


class Hub:
    def __init__(self):
        self.clients: dict[WebSocket, str] = {}

    async def connect(self, ws: WebSocket, role: str) -> None:
        await ws.accept()
        self.clients[ws] = role

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.pop(ws, None)

    def count(self, role: str) -> int:
        return sum(r == role for r in self.clients.values())

    async def broadcast(self, msg: dict, roles: set[str] | None = None) -> None:
        text = json.dumps(msg)
        dead = []
        for ws, role in list(self.clients.items()):
            if roles is not None and role not in roles:
                continue
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_threadsafe(self, loop: asyncio.AbstractEventLoop, msg: dict, roles: set[str] | None = None) -> None:
        asyncio.run_coroutine_threadsafe(self.broadcast(msg, roles), loop)
```

- [ ] **Step 5: Write app.py**

`neural-derby/derby/server/app.py`:

```python
"""The one server: routes for phones and the operator, the websocket hub, and the race thread."""
import asyncio
import json
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from derby import config
from derby.commentary.caller import Caller
from derby.commentary.verdict import compute_facts, verdicts as make_verdicts
from derby.data import augment, backup as backup_mod, dataset
from derby.data.live import LiveError, LiveStore
from derby.export.bench import bench_cpu
from derby.export.onnx_export import export_onnx
from derby.horses import stable as stable_mod
from derby.horses.stats import horse_stats
from derby.race import recorder, scoring
from derby.race.trainer import LapEvent, Race
from derby.server.hub import Hub
from derby.server.show import Act, GateError, Show


@dataclass
class Settings:
    data_dir: Path = config.DATA_DIR
    web_dir: Path = config.WEB_DIR
    laps: int = config.LAPS
    batch: int = config.BATCH
    lap_floor_s: float = config.LAP_FLOOR_S
    seed: int = config.SEED
    enabled: dict = field(default_factory=lambda: dict(config.HORSES_ENABLED))
    load_backup: bool = True
    device: str | None = None     # None = cuda if available else cpu
    public_url: str | None = None  # what the QR points at; None = detect LAN ip


@dataclass
class AppState:
    settings: Settings
    show: Show
    hub: Hub
    live: LiveStore
    stable: list
    stats: dict
    names: dict
    nicks: dict = field(default_factory=dict)
    bets: dict = field(default_factory=dict)
    backup: backup_mod.Loaded | None = None
    data: dataset.RaceData | None = None
    race: Race | None = None
    caller: Caller | None = None
    events: list = field(default_factory=list)
    race_standings: list = field(default_factory=list)
    sprint_ms: dict = field(default_factory=dict)
    sprint_device: str = "phone"
    sprint_standings: list = field(default_factory=list)
    champion: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    model_paths: dict = field(default_factory=dict)
    form_guide: dict = field(default_factory=dict)
    loop: asyncio.AbstractEventLoop | None = None

    @property
    def device(self) -> str:
        if self.settings.device:
            return self.settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; this just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def horse_cards(state: AppState) -> list[dict]:
    cards = []
    for h in state.stable:
        st = state.stats[h.key]
        fg = state.form_guide.get(h.key, {"race_wins": 0, "champion_wins": 0, "races": 0})
        cards.append({"key": h.key, "name": h.name, "tagline": h.tagline, "colour": h.colour, "block": h.block_name,
                      "params": st["params"], "macs": st["macs"], "depth": st["depth"], "table": st["table"],
                      "form": fg})
    return cards


def state_payload(state: AppState) -> dict:
    d = state.show.to_dict()
    d.update({"photos": state.live.count(), "bets": len(state.bets), "phones": state.hub.count("phone"),
              "horses": [h.key for h in state.stable], "n_live_train": state.data.n_live_train if state.data else 0,
              "n_live_test": state.data.n_live_test if state.data else 0,
              "champion": state.champion[0].horse if state.champion else None})
    return d


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    stable = stable_mod.enabled_stable(settings.enabled)
    state = AppState(settings=settings, show=Show(), hub=Hub(),
                     live=LiveStore(settings.data_dir / "live"), stable=stable,
                     stats={h.key: horse_stats(h) for h in stable}, names={h.key: h.name for h in stable})
    fg_path = settings.data_dir / "form_guide.json"
    if fg_path.exists():
        state.form_guide = json.loads(fg_path.read_text(encoding="utf-8"))

    app = FastAPI(title="Neural Derby")
    app.state.derby = state

    @app.on_event("startup")
    async def _startup():
        state.loop = asyncio.get_running_loop()
        if settings.load_backup:
            state.backup = backup_mod.load(settings.data_dir / "backup")

    async def push_state():
        await state.hub.broadcast({"type": "state", **state_payload(state)})

    # ---------- pages ----------
    if (settings.web_dir / "stadium").is_dir():
        app.mount("/stadium", StaticFiles(directory=settings.web_dir / "stadium", html=True), name="stadium")
        app.mount("/phone", StaticFiles(directory=settings.web_dir / "phone", html=True), name="phone")
        app.mount("/control", StaticFiles(directory=settings.web_dir / "control", html=True), name="control")
        app.mount("/vendor", StaticFiles(directory=settings.web_dir / "vendor"), name="vendor")

    @app.get("/p")
    def short_link():
        return RedirectResponse("/phone/")

    @app.get("/api/qr.png")
    def qr_png():
        import qrcode
        url = settings.public_url or f"http://{lan_ip()}:{config.PORTS['http']}/p"
        img = qrcode.make(url)
        import io
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"X-Join-Url": url})

    # ---------- state and operator ----------
    @app.get("/api/state")
    def get_state():
        return state_payload(state)

    @app.post("/api/advance")
    async def advance():
        try:
            act = state.show.advance()
        except GateError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        await push_state()
        if act == Act.MULTIPLY:
            await send_previews()
        elif act == Act.HORSES:
            await state.hub.broadcast({"type": "horses", "cards": horse_cards(state)})
            threading.Thread(target=load_data, daemon=True).start()
        elif act == Act.DERBY:
            start_race()
        elif act == Act.SPRINT:
            models = [{"key": k, "url": f"/models/{k}.onnx"} for k in state.model_paths]
            await state.hub.broadcast({"type": "sprint_request", "models": models}, roles={"team", "control"})
        elif act == Act.VERDICT:
            await finish_verdict()
        elif act == Act.SHIP:
            await ship()
        return {"act": act.value}

    @app.post("/api/gate/close")
    async def close_gate():
        state.show.close_gate()
        await push_state()
        return {"ok": True}

    # ---------- phones ----------
    @app.post("/api/join")
    async def join(body: dict):
        device, nick = str(body.get("device", "")), str(body.get("nick", ""))[:12]
        state.nicks[device] = nick or "anon"
        await push_state()
        return {"ok": True, "nick": state.nicks[device]}

    @app.post("/api/photo")
    async def photo(device: str = Form(...), class_name: str = Form(...), file: UploadFile = File(...)):
        if not state.show.photos_open:
            return JSONResponse({"error": "The paddock is closed right now."}, status_code=400)
        data = await file.read()
        try:
            p = state.live.add(device, class_name, data)
        except LiveError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        thumb = LiveStore.thumbnail_b64(p)
        await state.hub.broadcast({"type": "photo", "class": class_name, "thumb": thumb,
                                   "nick": state.nicks.get(device, "anon"), "count": state.live.count()})
        if state.live.count() >= state.live.caps["total"]:
            state.show.close_gate()
        await push_state()
        return {"ok": True, "thumb": thumb, "count": state.live.count_for(device)}

    @app.post("/api/bet")
    async def bet(body: dict):
        device, horse = str(body.get("device", "")), str(body.get("horse", ""))
        if not state.show.bets_open:
            return JSONResponse({"error": "Bets are closed."}, status_code=400)
        if horse not in state.names:
            return JSONResponse({"error": "That horse is not in this race."}, status_code=400)
        state.bets[device] = horse
        await push_state()
        return {"ok": True, "horse": horse}

    @app.post("/api/sprint-result")
    async def sprint_result(body: dict):
        horse, ms = str(body.get("horse", "")), float(body.get("ms", 0))
        if horse in state.model_paths and ms > 0:
            state.sprint_ms[horse] = ms
            state.sprint_device = str(body.get("device", "phone"))
        if set(state.sprint_ms) >= set(state.model_paths):
            await finish_sprint()
        return {"ok": True, "have": len(state.sprint_ms)}

    @app.post("/api/sprint/laptop")
    async def sprint_laptop():
        for k, p in state.model_paths.items():
            state.sprint_ms[k] = bench_cpu(p)
        state.sprint_device = "laptop"
        await finish_sprint()
        return {"ok": True}

    @app.post("/api/replay")
    async def replay(body: dict | None = None):
        path = Path((body or {}).get("path") or (recorder.latest_race(settings.data_dir / "races") or ""))
        if not path.is_file():
            return JSONResponse({"error": "No recorded race to replay."}, status_code=400)
        state.events.clear()
        state.caller = Caller([h.key for h in state.stable], state.names, settings.laps, seed=settings.seed)
        threading.Thread(target=run_events, args=(recorder.replay(path),), daemon=True).start()
        return {"ok": True, "path": str(path)}

    @app.get("/models/{key}.onnx")
    def model_file(key: str):
        if key == "champion" and state.champion:
            key = state.champion[0].horse
        p = state.model_paths.get(key)
        if not p:
            return JSONResponse({"error": "not ready"}, status_code=404)
        return FileResponse(p, media_type="application/octet-stream", filename=f"{key}.onnx")

    # ---------- websocket ----------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, role: str = "phone"):
        await state.hub.connect(ws, role)
        try:
            await ws.send_text(json.dumps({"type": "state", **state_payload(state)}))
            if state.show.act in (Act.HORSES, Act.DERBY, Act.SPRINT, Act.VERDICT, Act.SHIP):
                await ws.send_text(json.dumps({"type": "horses", "cards": horse_cards(state)}))
            while True:
                await ws.receive_text()   # phones only listen; keep the socket alive
        except WebSocketDisconnect:
            pass
        finally:
            state.hub.disconnect(ws)
            if state.loop:
                await push_state()

    # ---------- helpers that touch the race ----------
    async def send_previews():
        photos = state.live.photos()
        if photos:
            x, _ = LiveStore.tensors(photos[:48])
        elif state.backup is not None:
            x = state.backup.x[:: max(1, len(state.backup.x) // 48)][:48]
        else:
            x = torch.zeros(0, 3, config.IMG, config.IMG, dtype=torch.uint8)
        pairs = augment.preview_pairs(x, n=48, seed=settings.seed)
        await state.hub.broadcast({"type": "previews", "pairs": pairs, "n_before": state.live.count(),
                                   "n_after": state.live.count() * 10})

    def load_data():
        train, test = state.live.split_by_device(settings.seed)
        live_tr = LiveStore.tensors(train) if train else None
        live_te = LiveStore.tensors(test) if test else None
        state.data = dataset.assemble(state.backup, live_tr, live_te, settings.seed, device=state.device)
        state.show.data_loaded = True
        state.hub.broadcast_threadsafe(state.loop, {"type": "state", **state_payload(state)})

    def start_race():
        state.events.clear()
        state.sprint_ms.clear()
        state.race = Race(state.stable, state.data, laps=settings.laps, batch=settings.batch,
                          lap_floor_s=settings.lap_floor_s, seed=settings.seed, device=state.device)
        state.caller = Caller([h.key for h in state.stable], state.names, settings.laps, seed=settings.seed)
        threading.Thread(target=run_events, args=(state.race.run(), True), daemon=True).start()

    def run_events(event_iter, export_after: bool = False):
        for ev in event_iter:
            state.events.append(ev)
            state.race_standings = scoring.race_standings(state.events)
            lines = state.caller.on_event(ev) if state.caller else []
            state.hub.broadcast_threadsafe(state.loop, {
                "type": "lap", "event": ev.to_dict(), "lines": lines,
                "standings": [s.to_dict() for s in state.race_standings],
            })
        if export_after and state.race is not None:
            models_dir = settings.data_dir / "models"
            for key, model in state.race.models.items():
                state.model_paths[key] = export_onnx(model, models_dir / f"{key}.onnx")
        state.hub.broadcast_threadsafe(state.loop, {
            "type": "race_done", "standings": [s.to_dict() for s in state.race_standings],
            "models": [{"key": k, "url": f"/models/{k}.onnx"} for k in state.model_paths],
        })

    async def finish_sprint():
        state.sprint_standings = scoring.sprint_standings(state.sprint_ms, device=state.sprint_device)
        state.champion = scoring.champion_table(state.race_standings, state.sprint_standings)
        bets = scoring.settle_bets(state.bets, state.champion[0].horse, scoring.race_winner(state.race_standings))
        await state.hub.broadcast({
            "type": "sprint", "device": state.sprint_device,
            "sprint": [s.to_dict() for s in state.sprint_standings],
            "champion": [c.to_dict() for c in state.champion],
            "winners": {"champion": [state.nicks.get(d, d) for d in bets.champion_winners],
                        "race": [state.nicks.get(d, d) for d in bets.race_winners]},
            "bet_results": {d: (h == state.champion[0].horse) for d, h in state.bets.items()},
        })

    async def finish_verdict():
        if not state.champion and state.race_standings:
            state.sprint_standings = []
            state.champion = scoring.champion_table(state.race_standings, [])
        state.facts = compute_facts(state.events, state.stats, state.race_standings, state.sprint_standings, state.champion)
        n_train = state.data.n_train if state.data else 0
        state.verdicts = make_verdicts(state.facts, state.names, n_train)
        recorder.write_race(settings.data_dir / "races" / f"{time.strftime('%Y%m%d-%H%M%S')}.json", state.events,
                            state.race_standings, state.sprint_standings, state.champion, state.verdicts,
                            {"seed": settings.seed, "n_train": n_train, "device": state.device})
        await state.hub.broadcast({"type": "verdict", "verdicts": state.verdicts,
                                   "facts": {k: f.to_dict() for k, f in state.facts.items()}})

    async def ship():
        if state.champion:
            src = state.model_paths.get(state.champion[0].horse)
            if src:
                shutil.copy(src, settings.data_dir / "models" / "champion.onnx")
        await state.hub.broadcast({"type": "ship", "champion": state.champion[0].horse if state.champion else None,
                                   "url": "/models/champion.onnx", "classes": config.CLASSES})

    return app
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_show.py tests/test_app.py`
Expected: `6 passed`. The full-run test trains two horses for one lap on CPU; allow about 20 seconds.

- [ ] **Step 7: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/derby/server neural-derby/tests/test_show.py neural-derby/tests/test_app.py
git commit -m "Add the show state machine, websocket hub, and the derby server"
```

---
### Task 12: Stadium page (the arcade cabinet)

**Files:**
- Create: `neural-derby/tools/vendor_assets.py`, `neural-derby/web/stadium/index.html`, `neural-derby/web/stadium/stadium.css`, `neural-derby/web/stadium/sprites.js`, `neural-derby/web/stadium/audio.js`, `neural-derby/web/stadium/stadium.js`, `neural-derby/.claude/launch.json` (at the git root: `C:\dev\study\Edge-Ai\.claude\launch.json`)
- Modify: `neural-derby/derby/server/app.py` (add `GET /api/join-url`)

**Interfaces:**
- Consumes: websocket messages `state, photo, previews, horses, lap, race_done, sprint, verdict, ship` exactly as `app.py` sends them; `POST /api/advance`, `POST /api/gate/close`, `POST /api/sprint/laptop`, `POST /api/replay`; `GET /api/qr.png`; `GET /api/join-url` (added here) returning `{"url": "http://<ip>:8000/p"}`.
- Produces: a page that renders only from messages and holds no training logic. Keyboard: Space = advance, G = close gate, L = laptop sprint, R = replay last race, M = mute, Left/Right = page through verdict cards.
- `window.NeuralDerbyAudio` from `audio.js` with `unlock(), gun(), chime(), roar(), jingle()`; `window.NeuralDerbySprites` from `sprites.js` with `drawHorse(ctx, x, y, scale, colour, frame)` and `FRAMES = 2`.

- [ ] **Step 1: Add the join-url route to app.py**

In `neural-derby/derby/server/app.py`, directly below the `qr_png` route, add:

```python
    @app.get("/api/join-url")
    def join_url():
        return {"url": settings.public_url or f"http://{lan_ip()}:{config.PORTS['http']}/p"}
```

Run: `pytest tests/test_app.py` to confirm nothing broke. Expected: `4 passed`.

- [ ] **Step 2: Write the asset vendoring tool and run it once**

`neural-derby/tools/vendor_assets.py`:

```python
"""Download the pixel font and onnxruntime-web once (internet needed), into web/vendor/. The show never needs internet."""
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf"
FONT_LICENCE = "https://github.com/google/fonts/raw/main/ofl/pressstart2p/OFL.txt"
ORT_VERSION = "1.20.1"
ORT_TGZ = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    vendor = config.WEB_DIR / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "PressStart2P-Regular.ttf").write_bytes(get(FONT_URL))
    (vendor / "PressStart2P-OFL.txt").write_bytes(get(FONT_LICENCE))
    print("font ok")
    ort_dir = vendor / "onnxruntime-web"
    ort_dir.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(get(ORT_TGZ)), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.name.startswith("package/dist/") and (name == "ort.min.js" or name.startswith("ort-wasm-simd-threaded")):
                (ort_dir / name).write_bytes(tar.extractfile(member).read())
                print("  ", name)
            if member.name == "package/LICENSE":
                (ort_dir / "LICENSE").write_bytes(tar.extractfile(member).read())
    print("onnxruntime-web ok")


if __name__ == "__main__":
    main()
```

Run: `python tools/vendor_assets.py`
Expected: `font ok`, then at least `ort.min.js`, `ort-wasm-simd-threaded.wasm`, `ort-wasm-simd-threaded.mjs`, then `onnxruntime-web ok`. These files are committed (they are part of the show).

- [ ] **Step 3: Write sprites.js**

`neural-derby/web/stadium/sprites.js`:

```javascript
// Pixel-art horses drawn from strings, so there are no image files to lose.
// '#' body colour, 'e' eye (black), '.' empty. 16 wide x 12 tall, two gallop frames.
(function () {
  const A = [
    "..........###...",
    ".........#e###..",
    ".........####...",
    "....#########...",
    "...###########..",
    "..############..",
    "..############..",
    "..##..###..##...",
    ".##...##...##...",
    "##....#.....##..",
    "......#......#..",
    "................",
  ];
  const B = [
    "..........###...",
    ".........#e###..",
    ".........####...",
    "....#########...",
    "...###########..",
    "..############..",
    "..############..",
    "...##.###.##....",
    "...##.##...##...",
    "...#...#....##..",
    "..#....#.....#..",
    "................",
  ];
  const FRAMES = [A, B];
  function drawHorse(ctx, x, y, scale, colour, frame) {
    const map = FRAMES[frame % FRAMES.length];
    for (let r = 0; r < map.length; r++) {
      for (let c = 0; c < map[r].length; c++) {
        const ch = map[r][c];
        if (ch === ".") continue;
        ctx.fillStyle = ch === "e" ? "#000" : colour;
        ctx.fillRect(x + c * scale, y + r * scale, scale, scale);
      }
    }
  }
  window.NeuralDerbySprites = { drawHorse, FRAMES: FRAMES.length, W: 16, H: 12 };
})();
```

- [ ] **Step 4: Write audio.js**

`neural-derby/web/stadium/audio.js`:

```javascript
// 8-bit sounds from WebAudio. Nothing is loaded from disk.
(function () {
  let ctx = null;
  let muted = false;
  function unlock() { if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)(); if (ctx.state === "suspended") ctx.resume(); }
  function tone(freq, dur, type, gain) {
    if (!ctx || muted) return;
    const o = ctx.createOscillator(); const g = ctx.createGain();
    o.type = type || "square"; o.frequency.value = freq;
    g.gain.setValueAtTime(gain || 0.08, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
    o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + dur);
  }
  function noise(dur, gain, lowpass) {
    if (!ctx || muted) return;
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
    const s = ctx.createBufferSource(); s.buffer = buf;
    const f = ctx.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = lowpass || 2000;
    const g = ctx.createGain(); g.gain.value = gain || 0.2;
    s.connect(f).connect(g).connect(ctx.destination); s.start();
  }
  function gun() { noise(0.25, 0.4, 1200); }
  function chime() { tone(880, 0.08); setTimeout(() => tone(1320, 0.1), 70); }
  function roar() { noise(1.8, 0.25, 900); }
  function jingle() { [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => tone(f, 0.12), i * 90)); }
  function toggleMute() { muted = !muted; return muted; }
  window.NeuralDerbyAudio = { unlock, gun, chime, roar, jingle, toggleMute };
})();
```

- [ ] **Step 5: Write index.html**

`neural-derby/web/stadium/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neural Derby</title>
<meta name="viewport" content="width=1920">
<link rel="stylesheet" href="stadium.css">
</head>
<body>
<div class="scanlines"></div>
<header>
  <div class="title">NEURAL<span>DERBY</span></div>
  <div class="act" id="act">LOBBY</div>
  <div class="counters"><span id="c-phones">0 PHONES</span><span id="c-photos">0 PHOTOS</span><span id="c-bets">0 BETS</span></div>
</header>

<main>
  <section id="s-LOBBY" class="screen">
    <div class="hero">NEURAL DERBY</div>
    <div class="sub">SEVEN BRAINS. YOUR PHOTOS. ONE WINNER.</div>
    <div class="blink">PRESS SPACE TO OPEN THE PADDOCK</div>
  </section>

  <section id="s-PADDOCK" class="screen">
    <div class="paddock">
      <div class="qr-box">
        <div class="label">SCAN. PICK AN OBJECT. SHOOT IT.</div>
        <img id="qr" alt="QR code to join">
        <div class="url" id="join-url">...</div>
        <div class="big" id="photo-count">0 / 120</div>
        <div class="blink" id="paddock-hint">WAITING FOR THE FIRST PHOTO</div>
      </div>
      <div class="wall" id="wall"></div>
    </div>
  </section>

  <section id="s-MULTIPLY" class="screen">
    <div class="wall big-wall" id="aug-wall"></div>
    <div class="caption" id="aug-caption">MULTIPLYING...</div>
  </section>

  <section id="s-HORSES" class="screen">
    <div class="blink top-banner" id="bets-banner">PLACE YOUR BETS</div>
    <div class="cards" id="cards"></div>
  </section>

  <section id="s-DERBY" class="screen">
    <div class="derby">
      <div class="track-wrap"><canvas id="track" width="1380" height="880"></canvas><div class="lap" id="lap">LAP 0 / 20</div></div>
      <div class="tote"><div class="tote-title">TOTE</div><ol id="tote"></ol></div>
    </div>
    <div class="ticker" id="ticker"></div>
  </section>

  <section id="s-SPRINT" class="screen">
    <div class="sprint">
      <div class="bars" id="bars"></div>
      <div class="podium" id="podium"></div>
    </div>
    <div class="winners" id="winners"></div>
  </section>

  <section id="s-VERDICT" class="screen">
    <div class="verdict" id="verdict"></div>
    <div class="hint">LEFT / RIGHT TO PAGE THROUGH THE HORSES</div>
  </section>

  <section id="s-SHIP" class="screen">
    <div class="ship">
      <div class="qr-box">
        <div class="label">THE CHAMPION IS ON YOUR PHONE</div>
        <img id="qr2" alt="QR code to join">
        <div class="url" id="join-url2">...</div>
      </div>
      <div class="ship-right">
        <div class="hero small" id="ship-champ">...</div>
        <div class="sub">POINT YOUR PHONE AT YOUR DESK</div>
        <div class="veteran" id="veteran"></div>
      </div>
    </div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script src="sprites.js"></script>
<script src="audio.js"></script>
<script src="stadium.js"></script>
</body>
</html>
```

- [ ] **Step 6: Write stadium.css**

`neural-derby/web/stadium/stadium.css`:

```css
@font-face { font-family: "PS2P"; src: url("/vendor/PressStart2P-Regular.ttf") format("truetype"); }
:root {
  --bg: #0b0f2a; --panel: #121a3f; --line: #2a3a80; --amber: #ffb000; --text: #f4f4f4; --dim: #8fa0ff;
  --border: 4px;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: "PS2P", monospace; overflow: hidden; }
body { width: 1920px; height: 1080px; }
.scanlines { pointer-events: none; position: fixed; inset: 0; z-index: 50;
  background: repeating-linear-gradient(0deg, rgba(0,0,0,0.18) 0 2px, transparent 2px 4px); }
header { height: 90px; display: flex; align-items: center; justify-content: space-between; padding: 0 40px;
  border-bottom: var(--border) solid var(--line); background: var(--panel); }
.title { font-size: 34px; letter-spacing: 2px; } .title span { color: var(--amber); margin-left: 18px; }
.act { font-size: 26px; color: var(--amber); }
.counters span { font-size: 16px; margin-left: 36px; color: var(--dim); }
main { height: 990px; position: relative; }
.screen { display: none; position: absolute; inset: 0; padding: 40px; }
.screen.on { display: block; }
.hero { font-size: 110px; text-align: center; margin-top: 220px; color: var(--amber); text-shadow: 8px 8px 0 #000; }
.hero.small { font-size: 64px; margin-top: 60px; }
.sub { font-size: 24px; text-align: center; margin-top: 40px; color: var(--dim); }
.blink { animation: blink 1s steps(2, start) infinite; font-size: 22px; text-align: center; margin-top: 60px; }
@keyframes blink { to { visibility: hidden; } }
.top-banner { margin: 0 0 24px; font-size: 32px; color: var(--amber); }
.paddock, .ship { display: flex; gap: 40px; height: 100%; }
.qr-box { width: 520px; background: var(--panel); border: var(--border) solid var(--line); padding: 30px; text-align: center; }
.qr-box img { width: 400px; height: 400px; image-rendering: pixelated; background: #fff; padding: 12px; margin: 20px 0; }
.label { font-size: 16px; line-height: 1.8; color: var(--dim); }
.url { font-size: 22px; color: var(--amber); word-break: break-all; line-height: 1.6; }
.big { font-size: 56px; margin-top: 30px; }
.wall { flex: 1; display: grid; grid-template-columns: repeat(12, 96px); grid-auto-rows: 96px; gap: 8px; align-content: start; overflow: hidden; }
.big-wall { grid-template-columns: repeat(18, 96px); height: 860px; }
.tile { position: relative; width: 96px; height: 96px; border: 3px solid var(--line); background: #000; image-rendering: pixelated; animation: pop 0.25s steps(3, end); }
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
.tile .tag { position: absolute; left: 0; right: 0; bottom: 0; font-size: 9px; background: rgba(0,0,0,0.75); padding: 3px; color: var(--amber); white-space: nowrap; overflow: hidden; }
.tile.flip { animation: flip 0.4s steps(4, end); }
@keyframes pop { from { transform: scale(0.2); } to { transform: scale(1); } }
@keyframes flip { 0% { transform: rotateY(0); } 50% { transform: rotateY(90deg); } 100% { transform: rotateY(0); } }
.caption { position: absolute; left: 0; right: 0; bottom: 40px; text-align: center; font-size: 36px; color: var(--amber); text-shadow: 6px 6px 0 #000; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; }
.card { background: var(--panel); border: var(--border) solid var(--c, var(--line)); padding: 18px; min-height: 420px; position: relative; }
.card .name { font-size: 22px; color: var(--c); margin-bottom: 10px; }
.card canvas { display: block; margin: 6px 0 10px; image-rendering: pixelated; }
.card .block { font-size: 11px; color: var(--dim); margin-bottom: 10px; line-height: 1.6; }
.card .tag { font-size: 11px; line-height: 1.8; min-height: 80px; }
.stat { font-size: 10px; display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.stat .bar { flex: 1; height: 12px; background: #000; border: 2px solid var(--line); }
.stat .bar i { display: block; height: 100%; background: var(--c); }
.card .odds { position: absolute; right: 14px; top: 14px; font-size: 12px; color: var(--amber); }
.card .nums { font-size: 10px; color: var(--dim); margin-top: 8px; line-height: 1.8; }
.derby { display: flex; gap: 30px; height: 880px; }
.track-wrap { position: relative; background: #0d2a16; border: var(--border) solid #1f5a30; }
#track { display: block; }
.lap { position: absolute; top: 14px; right: 20px; font-size: 24px; color: var(--amber); text-shadow: 4px 4px 0 #000; }
.tote { width: 430px; background: #000; border: var(--border) solid var(--amber); padding: 18px; }
.tote-title { color: var(--amber); font-size: 22px; text-align: center; margin-bottom: 18px; }
.tote ol { list-style: none; margin: 0; padding: 0; }
.tote li { display: flex; align-items: center; gap: 10px; font-size: 16px; color: var(--amber); padding: 10px 0; border-bottom: 2px dashed #332; }
.tote li .sw { width: 18px; height: 18px; } .tote li .acc { margin-left: auto; }
.tote li.lead { background: #221; }
.ticker { position: absolute; left: 40px; right: 40px; bottom: 20px; height: 70px; background: #000; border: var(--border) solid var(--line); padding: 14px 20px; font-size: 20px; color: var(--amber); white-space: nowrap; overflow: hidden; }
.sprint { display: flex; gap: 40px; height: 700px; }
.bars { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 22px; }
.brow { display: flex; align-items: center; gap: 16px; font-size: 16px; }
.brow .nm { width: 260px; } .brow .bb { flex: 1; height: 36px; background: #000; border: 3px solid var(--line); }
.brow .bb i { display: block; height: 100%; transition: width 0.8s steps(12, end); }
.brow .ms { width: 140px; text-align: right; color: var(--amber); }
.podium { width: 620px; display: flex; align-items: flex-end; justify-content: center; gap: 16px; }
.pod { width: 180px; text-align: center; font-size: 14px; }
.pod .box { background: var(--c); color: #000; padding: 10px 4px; font-size: 12px; }
.pod.p1 .box { height: 260px; } .pod.p2 .box { height: 180px; } .pod.p3 .box { height: 120px; }
.winners { margin-top: 30px; font-size: 18px; line-height: 2; color: var(--dim); text-align: center; }
.winners b { color: var(--amber); }
.verdict { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; height: 880px; }
.vcard { background: var(--panel); border: var(--border) solid var(--c); padding: 26px; font-size: 14px; line-height: 1.9; overflow: hidden; }
.vcard .name { font-size: 30px; color: var(--c); margin-bottom: 12px; }
.vcard p { margin: 0 0 10px; }
.vtable { font-size: 10px; width: 100%; border-collapse: collapse; margin-top: 10px; }
.vtable td, .vtable th { border: 1px solid var(--line); padding: 4px 6px; text-align: left; }
.hint { position: absolute; bottom: 20px; left: 0; right: 0; text-align: center; font-size: 12px; color: var(--dim); }
.ship-right { flex: 1; }
.veteran { margin-top: 60px; font-size: 18px; line-height: 2; color: var(--dim); border: var(--border) dashed var(--line); padding: 20px; min-height: 160px; }
.toast { position: fixed; top: 110px; right: 40px; background: #a00; color: #fff; padding: 14px 20px; font-size: 14px; display: none; z-index: 60; }
```

- [ ] **Step 7: Write stadium.js**

`neural-derby/web/stadium/stadium.js`:

```javascript
// The stadium renders what the server tells it. It never trains anything.
(function () {
  const S = window.NeuralDerbySprites, A = window.NeuralDerbyAudio;
  const $ = (id) => document.getElementById(id);
  const LAPS_TOTAL = 20;
  let state = { act: "LOBBY" }, cards = [], cardByKey = {}, horses = [];
  let lanes = {};          // key -> {x, target, frame, colour, name}
  let lapNow = 0, verdictPage = 0, verdictData = null, factsData = null, muted = false;

  // ---------- helpers ----------
  const pct = (f) => Math.round(f * 100) + "%";
  const fmt = (n) => n.toLocaleString("en-US");
  function show(act) { document.querySelectorAll(".screen").forEach(s => s.classList.toggle("on", s.id === "s-" + act)); $("act").textContent = act; }
  function toast(msg) { const t = $("toast"); t.textContent = msg; t.style.display = "block"; setTimeout(() => t.style.display = "none", 2500); }
  function post(url, body) { return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }).then(r => r.json()); }
  function say(text) { if (muted || !window.speechSynthesis) return; const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.pitch = 1.1; speechSynthesis.speak(u); }
  function spriteCanvas(colour, scale) { const c = document.createElement("canvas"); c.width = S.W * scale; c.height = S.H * scale; S.drawHorse(c.getContext("2d"), 0, 0, scale, colour, 0); return c; }

  // ---------- state ----------
  function onState(m) {
    state = m; show(m.act);
    $("c-phones").textContent = m.phones + " PHONES"; $("c-photos").textContent = m.photos + " PHOTOS"; $("c-bets").textContent = m.bets + " BETS";
    $("photo-count").textContent = m.photos + " / 120";
    if (m.photos > 0) $("paddock-hint").textContent = m.photos_open ? "KEEP THEM COMING" : "GATES CLOSED";
    $("bets-banner").textContent = m.bets_open ? "PLACE YOUR BETS" : (m.data_loaded ? "GATES CLOSED. PRESS SPACE TO RACE" : "GATES CLOSED. LOADING PHOTOS...");
    if (m.act === "SHIP" && m.champion && cardByKey[m.champion]) $("ship-champ").textContent = cardByKey[m.champion].name;
  }
  fetch("/api/join-url").then(r => r.json()).then(j => { $("join-url").textContent = j.url; $("join-url2").textContent = j.url; });
  $("qr").src = "/api/qr.png"; $("qr2").src = "/api/qr.png";

  // ---------- paddock ----------
  function onPhoto(m) {
    const wall = $("wall");
    const t = document.createElement("div"); t.className = "tile";
    t.innerHTML = `<img src="data:image/jpeg;base64,${m.thumb}"><div class="tag">${m.class.toUpperCase()} · ${m.nick}</div>`;
    wall.prepend(t); while (wall.children.length > 96) wall.lastChild.remove();
    A.chime();
  }

  // ---------- multiply ----------
  function onPreviews(m) {
    const wall = $("aug-wall"); wall.innerHTML = "";
    m.pairs.forEach((p, i) => {
      const t = document.createElement("div"); t.className = "tile";
      t.innerHTML = `<img src="data:image/png;base64,${p[0]}">`; wall.appendChild(t);
      setTimeout(() => { t.classList.add("flip"); t.querySelector("img").src = "data:image/png;base64," + p[1]; }, 600 + i * 120);
    });
    const before = m.n_before || m.pairs.length, after = m.n_after || before * 10;
    $("aug-caption").textContent = "MULTIPLYING...";
    setTimeout(() => { $("aug-caption").textContent = `${fmt(before)} PHOTOS JUST BECAME ${fmt(after)}`; A.jingle(); }, 600 + m.pairs.length * 120);
  }

  // ---------- horses ----------
  function onHorses(m) {
    cards = m.cards; cardByKey = {}; horses = cards.map(c => c.key);
    const maxP = Math.max(...cards.map(c => c.params)), maxM = Math.max(...cards.map(c => c.macs)), maxD = Math.max(...cards.map(c => c.depth));
    const el = $("cards"); el.innerHTML = "";
    cards.forEach(c => {
      cardByKey[c.key] = c;
      const d = document.createElement("div"); d.className = "card"; d.style.setProperty("--c", c.colour);
      const form = c.form.races ? `WINS ${c.form.race_wins}/${c.form.races}` : "NO FORM";
      d.innerHTML = `<div class="odds">${form}</div><div class="name">${c.name}</div><div class="block">${c.block.toUpperCase()}</div>
        <div class="tag">${c.tagline}</div>
        <div class="stat">WEIGHT<div class="bar"><i style="width:${100 * c.params / maxP}%"></i></div></div>
        <div class="stat">WORK<div class="bar"><i style="width:${100 * c.macs / maxM}%"></i></div></div>
        <div class="stat">DEPTH<div class="bar"><i style="width:${100 * c.depth / maxD}%"></i></div></div>
        <div class="nums">${fmt(c.params)} PARAMS · ${fmt(c.macs)} MULTIPLIES · ${c.depth} LAYERS</div>`;
      d.insertBefore(spriteCanvas(c.colour, 5), d.querySelector(".block"));
      el.appendChild(d);
    });
    lanes = {}; cards.forEach(c => lanes[c.key] = { x: 0, target: 0, frame: 0, colour: c.colour, name: c.name });
    lapNow = 0; $("lap").textContent = `LAP 0 / ${LAPS_TOTAL}`; $("tote").innerHTML = ""; $("ticker").textContent = "";
  }

  // ---------- derby ----------
  const canvas = $("track"), ctx = canvas.getContext("2d");
  function drawTrack() {
    const W = canvas.width, H = canvas.height, n = Math.max(horses.length, 1), laneH = Math.floor((H - 40) / n), left = 260, right = W - 60, trackW = right - left;
    ctx.fillStyle = "#0d2a16"; ctx.fillRect(0, 0, W, H);
    ctx.font = "12px PS2P"; ctx.textBaseline = "middle";
    [0.25, 0.5, 0.75, 1.0].forEach(f => { const x = left + f * trackW; ctx.fillStyle = f === 1 ? "#fff" : "#1f5a30"; ctx.fillRect(x, 20, f === 1 ? 6 : 3, H - 40); ctx.fillStyle = "#8fa0ff"; ctx.fillText(pct(f), x - 20, 12); });
    for (let i = 0; i < 2 * (H - 40) / 24; i++) { ctx.fillStyle = i % 2 ? "#000" : "#fff"; ctx.fillRect(right, 20 + i * 12, 6, 12); }
    horses.forEach((k, i) => {
      const L = lanes[k]; if (!L) return;
      const y = 20 + i * laneH;
      ctx.fillStyle = i % 2 ? "#0f3119" : "#0d2a16"; ctx.fillRect(0, y, W, laneH);
      ctx.fillStyle = L.colour; ctx.fillText(L.name, 16, y + laneH / 2);
      L.x += (L.target - L.x) * 0.08;
      const scale = Math.max(3, Math.floor(laneH / 20));
      S.drawHorse(ctx, left + L.x * trackW - S.W * scale, y + laneH / 2 - S.H * scale / 2, scale, L.colour, L.frame);
      ctx.fillStyle = "#fff"; ctx.fillText(pct(L.target), left + L.x * trackW + 10, y + laneH / 2);
    });
  }
  setInterval(() => { Object.values(lanes).forEach(L => { if (Math.abs(L.target - L.x) > 0.002) L.frame = (L.frame + 1) % S.FRAMES; }); }, 150);
  (function loop() { if (state.act === "DERBY") drawTrack(); requestAnimationFrame(loop); })();

  function onLap(m) {
    const e = m.event, L = lanes[e.horse]; if (L) L.target = e.test_acc;
    if (e.lap !== lapNow) { lapNow = e.lap; $("lap").textContent = `LAP ${e.lap} / ${LAPS_TOTAL}`; if (e.lap === 1) A.gun(); else A.chime(); }
    const tote = $("tote"); tote.innerHTML = "";
    m.standings.forEach((s, i) => { const li = document.createElement("li"); if (i === 0) li.className = "lead";
      li.innerHTML = `<span>${s.rank}</span><span class="sw" style="background:${lanes[s.horse] ? lanes[s.horse].colour : "#fff"}"></span><span>${lanes[s.horse] ? lanes[s.horse].name : s.horse}</span><span class="acc">${pct(s.final_acc)}</span>`; tote.appendChild(li); });
    m.lines.forEach(line => { $("ticker").textContent = line; say(line); });
  }
  function onRaceDone(m) { A.roar(); $("ticker").textContent = "CHECKERED FLAG. PRESS SPACE FOR THE SPRINT."; }

  // ---------- sprint ----------
  function onSprint(m) {
    const bars = $("bars"); bars.innerHTML = ""; const maxMs = Math.max(...m.sprint.map(s => s.ms));
    m.sprint.forEach(s => { const c = cardByKey[s.horse] || { name: s.horse, colour: "#fff" };
      const row = document.createElement("div"); row.className = "brow";
      row.innerHTML = `<div class="nm" style="color:${c.colour}">${c.name}</div><div class="bb"><i style="width:0;background:${c.colour}"></i></div><div class="ms">${s.ms.toFixed(0)} ms</div>`;
      bars.appendChild(row); setTimeout(() => row.querySelector("i").style.width = (100 * s.ms / maxMs) + "%", 50); });
    const pod = $("podium"); pod.innerHTML = "";
    [1, 0, 2].forEach(i => { const c = m.champion[i]; if (!c) return; const card = cardByKey[c.horse] || { name: c.horse, colour: "#fff" };
      const d = document.createElement("div"); d.className = "pod p" + (i + 1); d.style.setProperty("--c", card.colour);
      d.innerHTML = `<div style="color:${card.colour};margin-bottom:8px">${card.name}</div><div class="box">${i === 0 ? "CHAMPION" : (i + 1) + (i === 1 ? "ND" : "RD")}<br>RACE ${c.race_rank} + SPRINT ${c.sprint_rank}</div>`;
      pod.appendChild(d); });
    const w = m.winners;
    $("winners").innerHTML = `TIMED ON: <b>${m.device.toUpperCase()}</b><br>PICKED THE CHAMPION: <b>${w.champion.length ? w.champion.join(", ") : "NOBODY"}</b><br>PICKED THE RACE WINNER: <b>${w.race.length ? w.race.join(", ") : "NOBODY"}</b>`;
    A.jingle(); say(`The champion is ${(cardByKey[m.champion[0].horse] || {}).name || m.champion[0].horse}.`);
  }

  // ---------- verdict ----------
  function renderVerdict() {
    if (!verdictData) return;
    const keys = horses.length ? horses : Object.keys(verdictData);
    const el = $("verdict"); el.innerHTML = "";
    [keys[verdictPage * 2], keys[verdictPage * 2 + 1]].forEach(k => {
      if (!k) return; const c = cardByKey[k] || { name: k, colour: "#fff", table: [] };
      const d = document.createElement("div"); d.className = "vcard"; d.style.setProperty("--c", c.colour);
      const rows = (c.table || []).slice(0, 10).map(r => `<tr><td>${r.layer}</td><td>${r.type}</td><td>${r.out_shape.join("x")}</td><td>${fmt(r.params)}</td></tr>`).join("");
      d.innerHTML = `<div class="name">${c.name}</div>` + verdictData[k].map(s => `<p>${s}</p>`).join("") +
        `<table class="vtable"><tr><th>LAYER</th><th>TYPE</th><th>OUT</th><th>PARAMS</th></tr>${rows}</table>`;
      el.appendChild(d);
    });
  }
  function onVerdict(m) { verdictData = m.verdicts; factsData = m.facts; verdictPage = 0; renderVerdict(); }

  // ---------- ship ----------
  function onShip(m) { if (m.champion && cardByKey[m.champion]) $("ship-champ").textContent = cardByKey[m.champion].name; A.jingle(); }
  function onVeteran(m) { $("veteran").innerHTML = `THE VETERAN (PRETRAINED MOBILENET): <b>${pct(m.acc)}</b> AFTER ONE LAP<br>BUT <b>${m.ms.toFixed(0)} ms</b> PER PHOTO ON ${m.device.toUpperCase()} VS THE CHAMPION'S <b>${m.champion_ms.toFixed(0)} ms</b><br>IT HAD SEEN A MILLION PHOTOS BEFORE TODAY. THAT IS TRANSFER LEARNING. TOO HEAVY FOR THE PHONE: THAT IS EDGE AI.`; }

  // ---------- socket ----------
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?role=stadium`);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data);
      ({ state: onState, photo: onPhoto, previews: onPreviews, horses: onHorses, lap: onLap, race_done: onRaceDone, sprint: onSprint, verdict: onVerdict, ship: onShip, veteran: onVeteran }[m.type] || (() => {}))(m); };
    ws.onclose = () => setTimeout(connect, 3000);
  }
  connect();

  // ---------- keys ----------
  document.addEventListener("keydown", async (e) => {
    A.unlock();
    if (e.code === "Space") { e.preventDefault(); const r = await post("/api/advance"); if (r.error) toast(r.error); }
    else if (e.key === "g" || e.key === "G") post("/api/gate/close");
    else if (e.key === "l" || e.key === "L") post("/api/sprint/laptop");
    else if (e.key === "r" || e.key === "R") { const r = await post("/api/replay"); if (r.error) toast(r.error); }
    else if (e.key === "m" || e.key === "M") { muted = A.toggleMute(); toast(muted ? "MUTED" : "SOUND ON"); }
    else if (e.key === "ArrowRight" && state.act === "VERDICT") { verdictPage = Math.min(verdictPage + 1, Math.ceil(horses.length / 2) - 1); renderVerdict(); }
    else if (e.key === "ArrowLeft" && state.act === "VERDICT") { verdictPage = Math.max(verdictPage - 1, 0); renderVerdict(); }
  });
})();
```

- [ ] **Step 8: Create the launch config and look at every act in the browser pane**

`C:\dev\study\Edge-Ai\.claude\launch.json`:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "derby",
      "runtimeExecutable": "C:\\dev\\study\\Edge-Ai\\neural-derby\\.venv\\Scripts\\python.exe",
      "runtimeArgs": ["-m", "uvicorn", "derby.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"],
      "cwd": "C:\\dev\\study\\Edge-Ai\\neural-derby",
      "port": 8000
    }
  ]
}
```

Start it with `preview_start` using name `derby`, then open `http://localhost:8000/stadium/` and `http://localhost:8000/control/` (the control page arrives in Task 13; until then use keys on the stadium). Walk the acts with Space and screenshot each at 1920x1080:

- LOBBY: hero text centred, blinking prompt.
- PADDOCK: QR renders, the URL reads `http://<lan ip>:8000/p`, the wall is empty with "WAITING FOR THE FIRST PHOTO" blinking. Upload a photo with curl to see a tile pop in:
  `curl -F device=dev_0001 -F class_name=pen -F "file=@data/backup/pen/0000.jpg" http://localhost:8000/api/photo`
- MULTIPLY: tiles flip to augmented versions; caption updates.
- HORSES: seven cards, sprites coloured per horse, bars and numbers legible; banner says loading then "PRESS SPACE TO RACE".
- DERBY: seven lanes, horses move right as laps come in, tote updates, ticker shows lines and the voice speaks. The finish line and markers are visible.
- SPRINT: press L for laptop timing; bars animate; podium shows three; winners text.
- VERDICT: two cards per page, arrows page through; layer tables render.
- SHIP: champion name, QR.

Expected: every screen readable at arm's length from the screenshot with no clipped text. Fix any overlap by reducing the font size in `stadium.css` for that element.

- [ ] **Step 9: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add .claude/launch.json neural-derby/tools/vendor_assets.py neural-derby/web/stadium neural-derby/web/vendor neural-derby/derby/server/app.py
git commit -m "Add the arcade stadium page, vendored font and runtime, and the launch config"
```

---

### Task 13: Phone and control pages

**Files:**
- Create: `neural-derby/web/phone/index.html`, `neural-derby/web/phone/phone.css`, `neural-derby/web/phone/phone.js`, `neural-derby/web/control/index.html`

**Interfaces:**
- Consumes: `POST /api/join`, `POST /api/photo` (multipart `device, class_name, file`), `POST /api/bet`, `POST /api/sprint-result` (`{horse, ms, device}`), `GET /models/<key>.onnx`, `GET /models/champion.onnx`, websocket `/ws?role=phone` or `role=team` (with `?team=1` in the page URL), messages `state, horses, lap, sprint, ship, sprint_request`.
- Produces: the phone page in snapshot mode for the crowd and live-video mode when opened over HTTPS with `?team=1` (Task 15 adds the HTTPS listener; the code path is written here). `window.NeuralDerbyInfer` with `load(url) -> Promise<session>`, `classify(session, canvas) -> Promise<[{label, p}]>`, `timeModel(url, n) -> Promise<ms>`.
- The device token is a random 16-character string stored in `localStorage` under `derby_device`; the nickname under `derby_nick`.

- [ ] **Step 1: Write index.html**

`neural-derby/web/phone/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Neural Derby</title>
<link rel="stylesheet" href="phone.css">
</head>
<body>
<header><span>NEURAL</span> DERBY</header>
<div class="banner" id="banner">CONNECTING...</div>

<section id="p-join" class="screen on">
  <h2>YOUR NAME</h2>
  <input id="nick" maxlength="12" placeholder="12 letters max" autocomplete="off">
  <button id="join-btn" class="big">JOIN THE ROOM</button>
</section>

<section id="p-wait" class="screen">
  <h2 id="wait-title">HOLD ON</h2>
  <p id="wait-text">The paddock opens soon. Look at the big screen.</p>
</section>

<section id="p-capture" class="screen">
  <h2>PICK WHAT YOU WILL SHOOT</h2>
  <div class="grid" id="class-grid"></div>
  <input type="file" id="shot" accept="image/*" capture="environment" hidden>
  <div id="preview-box" hidden>
    <img id="preview" alt="your photo">
    <div class="row"><button id="retake">RETAKE</button><button id="send" class="big">SEND</button></div>
  </div>
  <p class="small" id="sent-count">0 of 3 sent</p>
</section>

<section id="p-bet" class="screen">
  <h2>PLACE YOUR BET</h2>
  <div id="bet-list"></div>
</section>

<section id="p-race" class="screen">
  <h2 id="race-title">LAP 0</h2>
  <ol id="mini"></ol>
  <p class="small" id="your-horse"></p>
</section>

<section id="p-result" class="screen">
  <h2 id="result-title">...</h2>
  <p id="result-text"></p>
</section>

<section id="p-ship" class="screen">
  <h2 id="ship-title">THE CHAMPION IS HERE</h2>
  <p class="small" id="ship-status">downloading the model...</p>
  <input type="file" id="shot2" accept="image/*" capture="environment" hidden>
  <button id="classify-btn" class="big" disabled>POINT AT YOUR DESK</button>
  <video id="cam" playsinline muted hidden></video>
  <canvas id="work" width="96" height="96" hidden></canvas>
  <div id="bars"></div>
  <p class="small" id="infer-ms"></p>
</section>

<script src="/vendor/onnxruntime-web/ort.min.js"></script>
<script src="phone.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write phone.css**

`neural-derby/web/phone/phone.css`:

```css
@font-face { font-family: "PS2P"; src: url("/vendor/PressStart2P-Regular.ttf") format("truetype"); }
:root { --bg: #0b0f2a; --panel: #121a3f; --line: #2a3a80; --amber: #ffb000; --text: #f4f4f4; --dim: #8fa0ff; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: "PS2P", monospace; font-size: 12px; line-height: 1.7; }
header { padding: 14px 16px; border-bottom: 4px solid var(--line); background: var(--panel); font-size: 16px; }
header span { color: var(--amber); }
.banner { background: #a00; color: #fff; padding: 8px 16px; font-size: 10px; }
.banner.ok { background: #0a4; }
.screen { display: none; padding: 16px; } .screen.on { display: block; }
h2 { font-size: 14px; color: var(--amber); margin: 8px 0 16px; }
input[type=text], input:not([type]) { width: 100%; font: inherit; padding: 14px; background: #000; color: var(--text); border: 4px solid var(--line); margin-bottom: 16px; }
button { font: inherit; width: 100%; padding: 16px 10px; background: var(--panel); color: var(--text); border: 4px solid var(--line); margin-bottom: 12px; }
button.big { background: var(--amber); color: #000; border-color: var(--amber); }
button:disabled { opacity: 0.4; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.grid button { margin: 0; font-size: 13px; min-height: 80px; }
.grid button.sel { border-color: var(--amber); color: var(--amber); }
#preview { width: 100%; border: 4px solid var(--line); margin: 14px 0; }
.row { display: flex; gap: 10px; } .row button { flex: 1; }
.small { color: var(--dim); font-size: 10px; }
.bet { display: flex; gap: 12px; align-items: center; padding: 12px; border: 4px solid var(--c); margin-bottom: 10px; background: var(--panel); }
.bet .nm { color: var(--c); font-size: 12px; } .bet .tg { font-size: 9px; color: var(--dim); line-height: 1.6; }
.bet.sel { background: #332; }
ol#mini { list-style: none; padding: 0; margin: 0; }
ol#mini li { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 2px dashed var(--line); }
#bars .b { margin: 10px 0; } #bars .b .lab { display: flex; justify-content: space-between; font-size: 11px; }
#bars .b .bar { height: 18px; background: #000; border: 3px solid var(--line); } #bars .b .bar i { display: block; height: 100%; background: var(--amber); }
#cam { width: 100%; border: 4px solid var(--line); margin-bottom: 10px; }
```

- [ ] **Step 3: Write phone.js**

`neural-derby/web/phone/phone.js`:

```javascript
(function () {
  const $ = (id) => document.getElementById(id);
  const TEAM = new URLSearchParams(location.search).get("team") === "1";
  const CLASSES = ["bottle", "pen", "book", "phone", "backpack", "shoe"];
  const device = localStorage.getItem("derby_device") || (() => { const d = Array.from(crypto.getRandomValues(new Uint8Array(8)), b => b.toString(16).padStart(2, "0")).join(""); localStorage.setItem("derby_device", d); return d; })();
  let nick = localStorage.getItem("derby_nick") || "";
  let state = { act: "LOBBY" }, cards = [], myBet = localStorage.getItem("derby_bet") || null, sent = 0, chosenClass = null, shotFile = null, session = null;

  function show(id) { document.querySelectorAll(".screen").forEach(s => s.classList.toggle("on", s.id === id)); }
  function banner(text, ok) { const b = $("banner"); b.textContent = text; b.className = "banner" + (ok ? " ok" : ""); }
  function post(url, body) { return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json()); }

  // ---------- routing by act ----------
  function route() {
    if (!nick) { show("p-join"); return; }
    const a = state.act;
    if (a === "LOBBY") { $("wait-title").textContent = "HOLD ON"; $("wait-text").textContent = "The paddock opens soon. Look at the big screen."; show("p-wait"); }
    else if (a === "PADDOCK") { if (state.photos_open && sent < 3) show("p-capture"); else { $("wait-title").textContent = sent >= 3 ? "THANK YOU" : "GATES CLOSED"; $("wait-text").textContent = "Your photos are in. Betting opens next."; show("p-wait"); } }
    else if (a === "MULTIPLY") { $("wait-title").textContent = "AUGMENTING"; $("wait-text").textContent = "Your photos are being spun, darkened, and multiplied."; show("p-wait"); }
    else if (a === "HORSES") { if (state.bets_open) show("p-bet"); else { $("wait-title").textContent = myBet ? "BET LOCKED" : "GATES CLOSED"; $("wait-text").textContent = myBet ? "You picked " + nameOf(myBet) + ". Watch the big screen." : "No bet this time. Watch the race!"; show("p-wait"); } }
    else if (a === "DERBY") show("p-race");
    else if (a === "SPRINT" || a === "VERDICT") { if (!$("result-title").textContent.trim() || $("result-title").textContent === "...") { $("result-title").textContent = "TIMING ON THE PHONE"; $("result-text").textContent = "The sprint decides the champion."; } show("p-result"); }
    else if (a === "SHIP") show("p-ship");
  }
  function nameOf(key) { const c = cards.find(c => c.key === key); return c ? c.name : key.toUpperCase(); }

  // ---------- join ----------
  $("nick").value = nick;
  $("join-btn").onclick = async () => { nick = $("nick").value.trim().slice(0, 12) || "anon"; localStorage.setItem("derby_nick", nick); await post("/api/join", { device, nick }); route(); };

  // ---------- capture ----------
  const grid = $("class-grid");
  CLASSES.forEach(c => { const b = document.createElement("button"); b.textContent = c.toUpperCase(); b.onclick = () => { chosenClass = c; grid.querySelectorAll("button").forEach(x => x.classList.toggle("sel", x === b)); $("shot").click(); }; grid.appendChild(b); });
  $("shot").onchange = () => { shotFile = $("shot").files[0]; if (!shotFile) return; $("preview").src = URL.createObjectURL(shotFile); $("preview-box").hidden = false; };
  $("retake").onclick = () => { $("preview-box").hidden = true; $("shot").value = ""; $("shot").click(); };
  $("send").onclick = async () => {
    if (!shotFile || !chosenClass) return;
    const fd = new FormData(); fd.append("device", device); fd.append("class_name", chosenClass); fd.append("file", shotFile, "photo.jpg");
    $("send").disabled = true;
    const r = await fetch("/api/photo", { method: "POST", body: fd }).then(r => r.json()).catch(() => ({ error: "No connection. Try again." }));
    $("send").disabled = false;
    if (r.error) { banner(r.error); return; }
    sent = r.count; $("sent-count").textContent = `${sent} of 3 sent`; banner("SENT! " + (3 - sent) + " left", true);
    $("preview-box").hidden = true; $("shot").value = ""; shotFile = null;
    if (sent >= 3) route();
  };

  // ---------- bet ----------
  function renderBets() {
    const el = $("bet-list"); el.innerHTML = "";
    cards.forEach(c => { const d = document.createElement("div"); d.className = "bet" + (myBet === c.key ? " sel" : ""); d.style.setProperty("--c", c.colour);
      d.innerHTML = `<div><div class="nm">${c.name}</div><div class="tg">${c.tagline}</div><div class="tg">${c.params.toLocaleString()} params · ${c.form.races ? "wins " + c.form.race_wins + "/" + c.form.races : "no form"}</div></div>`;
      d.onclick = async () => { const r = await post("/api/bet", { device, horse: c.key }); if (r.error) { banner(r.error); return; } myBet = c.key; localStorage.setItem("derby_bet", myBet); banner("BET LOCKED ON " + c.name, true); renderBets(); };
      el.appendChild(d); });
  }

  // ---------- race ----------
  function onLap(m) {
    $("race-title").textContent = "LAP " + m.event.lap;
    const ol = $("mini"); ol.innerHTML = "";
    m.standings.forEach(s => { const li = document.createElement("li"); li.innerHTML = `<span style="color:${(cards.find(c => c.key === s.horse) || {}).colour || "#fff"}">${s.rank}. ${nameOf(s.horse)}</span><span>${Math.round(s.final_acc * 100)}%</span>`; ol.appendChild(li); });
    if (myBet) { const s = m.standings.find(s => s.horse === myBet); if (s) $("your-horse").textContent = `YOUR HORSE ${nameOf(myBet)} IS ${s.rank}${["st", "nd", "rd"][s.rank - 1] || "th"}`; }
  }
  function onSprint(m) {
    const won = m.bet_results && m.bet_results[device];
    $("result-title").textContent = myBet ? (won ? "YOU WON!" : "BETTER LUCK NEXT RACE") : "THE CHAMPION";
    $("result-text").textContent = `Champion: ${nameOf(m.champion[0].horse)}. Timed on the ${m.device}.` + (myBet ? ` You picked ${nameOf(myBet)}.` : "");
    show("p-result");
  }

  // ---------- inference on the phone ----------
  const Infer = {
    async load(url) { ort.env.wasm.wasmPaths = "/vendor/onnxruntime-web/"; ort.env.wasm.numThreads = 1; return ort.InferenceSession.create(url, { executionProviders: ["wasm"] }); },
    tensorFromCanvas(canvas) {
      const { data } = canvas.getContext("2d").getImageData(0, 0, 96, 96);
      const f = new Float32Array(3 * 96 * 96);
      for (let i = 0; i < 96 * 96; i++) { for (let c = 0; c < 3; c++) f[c * 96 * 96 + i] = (data[i * 4 + c] / 255 - 0.5) / 0.5; }
      return new ort.Tensor("float32", f, [1, 3, 96, 96]);
    },
    drawSquare(source, sw, sh, canvas) { const s = Math.min(sw, sh); canvas.getContext("2d").drawImage(source, (sw - s) / 2, (sh - s) / 2, s, s, 0, 0, 96, 96); },
    async classify(sess, canvas) {
      const t0 = performance.now(); const out = await sess.run({ image: Infer.tensorFromCanvas(canvas) }); const ms = performance.now() - t0;
      const l = Array.from(out.logits.data); const m = Math.max(...l); const e = l.map(v => Math.exp(v - m)); const z = e.reduce((a, b) => a + b, 0);
      return { probs: CLASSES.map((label, i) => ({ label, p: e[i] / z })).sort((a, b) => b.p - a.p), ms };
    },
    async timeModel(url, n) {
      const sess = await Infer.load(url); const t = new ort.Tensor("float32", new Float32Array(3 * 96 * 96), [1, 3, 96, 96]);
      for (let i = 0; i < 3; i++) await sess.run({ image: t });
      const times = []; for (let i = 0; i < n; i++) { const a = performance.now(); await sess.run({ image: t }); times.push(performance.now() - a); }
      times.sort((a, b) => a - b); return times[Math.floor(times.length / 2)];
    },
  };
  window.NeuralDerbyInfer = Infer;

  function showProbs(r) {
    const el = $("bars"); el.innerHTML = "";
    r.probs.slice(0, 3).forEach(p => { const d = document.createElement("div"); d.className = "b"; d.innerHTML = `<div class="lab"><span>${p.label.toUpperCase()}</span><span>${Math.round(p.p * 100)}%</span></div><div class="bar"><i style="width:${p.p * 100}%"></i></div>`; el.appendChild(d); });
    $("infer-ms").textContent = `${r.ms.toFixed(0)} ms on this phone, no laptop involved`;
  }
  async function onShip(m) {
    show("p-ship"); $("ship-status").textContent = "downloading " + nameOf(m.champion) + "...";
    try { session = await Infer.load(m.url); } catch (e) { $("ship-status").textContent = "could not load the model: " + e; return; }
    $("ship-status").textContent = nameOf(m.champion) + " is on your phone."; $("classify-btn").disabled = false;
    if (TEAM && navigator.mediaDevices && location.protocol === "https:") startLive(); else $("classify-btn").onclick = () => $("shot2").click();
  }
  $("shot2").onchange = async () => {
    const f = $("shot2").files[0]; if (!f || !session) return;
    const img = new Image(); img.onload = async () => { Infer.drawSquare(img, img.naturalWidth, img.naturalHeight, $("work")); showProbs(await Infer.classify(session, $("work"))); URL.revokeObjectURL(img.src); };
    img.src = URL.createObjectURL(f);
  };
  async function startLive() {
    const v = $("cam"); v.hidden = false; $("classify-btn").hidden = true;
    v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 640 } }); await v.play();
    (async function tick() { if (!session) return; Infer.drawSquare(v, v.videoWidth, v.videoHeight, $("work")); showProbs(await Infer.classify(session, $("work"))); setTimeout(tick, 150); })();
  }

  // ---------- team phone: sprint timing ----------
  async function onSprintRequest(m) {
    $("result-title").textContent = "TIMING THE HORSES"; show("p-result");
    for (const model of m.models) {
      $("result-text").textContent = "timing " + nameOf(model.key) + "...";
      try { const ms = await Infer.timeModel(model.url, 30); await post("/api/sprint-result", { horse: model.key, ms, device: "phone" }); }
      catch (e) { $("result-text").textContent = "failed on " + model.key + ": " + e; }
    }
    $("result-text").textContent = "all horses timed.";
  }

  // ---------- socket ----------
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?role=${TEAM ? "team" : "phone"}`);
    ws.onopen = () => banner(TEAM ? "TEAM PHONE CONNECTED" : "CONNECTED", true);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data);
      if (m.type === "state") { state = m; route(); }
      else if (m.type === "horses") { cards = m.cards; renderBets(); }
      else if (m.type === "lap") onLap(m);
      else if (m.type === "sprint") onSprint(m);
      else if (m.type === "ship") onShip(m);
      else if (m.type === "sprint_request" && TEAM) onSprintRequest(m); };
    ws.onclose = () => { banner("RECONNECTING..."); setTimeout(connect, 3000); };
  }
  connect();
})();
```

- [ ] **Step 4: Write the control page**

`neural-derby/web/control/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Derby control</title>
<style>
  body { font-family: Consolas, monospace; background: #111; color: #eee; padding: 16px; font-size: 14px; }
  button { font: inherit; padding: 10px 14px; margin: 4px 6px 4px 0; background: #333; color: #fff; border: 1px solid #666; }
  button.warn { background: #622; }
  pre { background: #000; padding: 10px; }
  #log div { border-bottom: 1px solid #333; padding: 4px 0; }
</style>
</head>
<body>
<h3>Neural Derby control</h3>
<div>Join URL: <b id="url">...</b></div>
<pre id="state">connecting...</pre>
<div>
  <button id="advance">Advance (Space)</button>
  <button id="gate">Close gate (G)</button>
  <button id="laptop" class="warn">Laptop Sprint timing (L)</button>
  <button id="replay" class="warn">Replay last race (R)</button>
  <button id="veteran">Run the Veteran</button>
</div>
<h4>Events</h4>
<div id="log"></div>
<script>
  const $ = (id) => document.getElementById(id);
  const post = (u, b) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) }).then(r => r.json()).then(j => log(JSON.stringify(j)));
  function log(t) { const d = document.createElement("div"); d.textContent = new Date().toLocaleTimeString() + " " + t; $("log").prepend(d); }
  fetch("/api/join-url").then(r => r.json()).then(j => $("url").textContent = j.url);
  $("advance").onclick = () => post("/api/advance"); $("gate").onclick = () => post("/api/gate/close");
  $("laptop").onclick = () => post("/api/sprint/laptop"); $("replay").onclick = () => post("/api/replay"); $("veteran").onclick = () => post("/api/veteran");
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?role=control`);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.type === "state") $("state").textContent = JSON.stringify(m, null, 1); else log(m.type + (m.event ? ` ${m.event.horse} lap ${m.event.lap} ${Math.round(m.event.test_acc * 100)}%` : "")); };
    ws.onclose = () => setTimeout(connect, 2000);
  }
  connect();
</script>
</body>
</html>
```

- [ ] **Step 5: Verify on the phone preset in the browser pane**

With the `derby` server running (Task 12 Step 8), open `http://localhost:8000/phone/` in the browser pane with the mobile preset (375 wide). Walk it alongside the stadium:

- JOIN: type a name, tap JOIN. Expected: the HOLD ON screen.
- Press Space on the stadium (PADDOCK). Expected: the six buttons appear; tapping one opens a file picker (the pane has no camera; pick any JPEG from `data/backup/`). Preview shows; SEND posts; counter says `1 of 3 sent`; the stadium wall gets the tile.
- Advance to HORSES. Expected: seven bet rows; tapping one shows "BET LOCKED ON ..." and highlights it.
- Advance to DERBY (wait for `data_loaded`). Expected: LAP counter and the mini standings update; "YOUR HORSE ... IS 3rd".
- Advance to SPRINT, press L on the stadium. Expected: the result screen says YOU WON or BETTER LUCK.
- Advance through VERDICT to SHIP. Expected: "... is on your phone", the POINT AT YOUR DESK button enables, picking a photo shows three bars and a millisecond figure. Check the browser console for ONNX errors; if `wasmPaths` cannot be found, the vendor folder is missing the `.wasm` file from Task 12 Step 2.
- Open `http://localhost:8000/control/`. Expected: state JSON updates live; buttons work.

Also open the phone page in the desktop preset at 1024 wide to confirm nothing breaks, then reset the preset.

- [ ] **Step 6: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/web/phone neural-derby/web/control
git commit -m "Add the phone page with on-device inference and the operator control page"
```

---
### Task 14: Practice races, balancing, form guide, and the Veteran

**Files:**
- Create: `neural-derby/tools/practice_races.py`, `neural-derby/derby/horses/veteran.py`, `neural-derby/tests/test_veteran.py`
- Modify: `neural-derby/derby/server/app.py` (add `POST /api/veteran`), `neural-derby/derby/horses/stable.py` (widths and `DEEP_STAGES` as the balancing step decides)

**Interfaces:**
- Produces: `veteran.build_veteran(pretrained: bool = True) -> nn.Module` (torchvision MobileNetV3-small with a new 6-way head); `veteran.VETERAN: Horse` with key `veteran`; `POST /api/veteran` which trains it for one lap on the loaded race data, exports it, times it and the champion on the laptop, and broadcasts `{"type": "veteran", "acc", "ms", "champion_ms", "device": "laptop"}`.
- Produces: `data/form_guide.json` (committed) and `data/races/practice-*.json` (git-ignored).

- [ ] **Step 1: Write the failing veteran test**

`neural-derby/tests/test_veteran.py`:

```python
import torch

from derby.horses.veteran import VETERAN, build_veteran


def test_veteran_builds_without_weights_and_outputs_six():
    m = build_veteran(pretrained=False).eval()
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, 6)
    assert VETERAN.key == "veteran" and VETERAN.name == "THE VETERAN"
```

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/test_veteran.py`
Expected: `ModuleNotFoundError: derby.horses.veteran`

- [ ] **Step 3: Write veteran.py**

`neural-derby/derby/horses/veteran.py`:

```python
"""The epilogue horse: a pretrained MobileNetV3-small (Module 2, transfer learning). Not in the race."""
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from derby import config
from derby.horses.stable import Horse


def build_veteran(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = mobilenet_v3_small(weights=weights)   # first call with weights downloads about 10 MB once (internet)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(config.CLASSES))
    return m


VETERAN = Horse("veteran", "THE VETERAN",
                "Has seen a million photos before today. Pretrained, then fine-tuned for one lap.",
                "#ffffff", "pretrained MobileNetV3-small", build_veteran)
```

- [ ] **Step 4: Run the test, then pull the weights once**

Run: `pytest tests/test_veteran.py` → `1 passed`.
Run once with internet: `python -c "from derby.horses.veteran import build_veteran; build_veteran(True); print('weights cached')"`.

- [ ] **Step 5: Add the veteran route to app.py**

In `neural-derby/derby/server/app.py`, add the import near the others:

```python
from derby.horses.veteran import VETERAN
```

and this route next to `/api/sprint/laptop`:

```python
    @app.post("/api/veteran")
    async def veteran():
        if state.data is None or not state.champion:
            return JSONResponse({"error": "Run the race and the sprint first."}, status_code=400)

        def work():
            race = Race([VETERAN], state.data, laps=1, batch=settings.batch, lap_floor_s=0.0,
                        seed=settings.seed, lr=1e-3, device=state.device)
            ev = list(race.run())[-1]
            path = export_onnx(race.models["veteran"], settings.data_dir / "models" / "veteran.onnx")
            ms = bench_cpu(path)
            champ_ms = bench_cpu(state.model_paths[state.champion[0].horse])
            state.hub.broadcast_threadsafe(state.loop, {"type": "veteran", "acc": ev.test_acc, "ms": ms,
                                                        "champion_ms": champ_ms, "device": "laptop"})

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}
```

Run: `pytest tests/test_app.py` → still `4 passed`.

- [ ] **Step 6: Write the practice-race tool**

`neural-derby/tools/practice_races.py`:

```python
"""Run N headless races on the backup set. Writes data/races/practice-<seed>.json and data/form_guide.json.

Usage: python tools/practice_races.py --n 20
Prints the win table you use to balance the stable (Step 7) and that becomes the betting odds.
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402
from derby.commentary.verdict import compute_facts  # noqa: E402
from derby.data import backup, dataset  # noqa: E402
from derby.export.bench import bench_cpu  # noqa: E402
from derby.export.onnx_export import export_onnx  # noqa: E402
from derby.horses import stable as stable_mod  # noqa: E402
from derby.horses.stats import horse_stats  # noqa: E402
from derby.race import recorder, scoring  # noqa: E402
from derby.race.trainer import Race  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--laps", type=int, default=config.LAPS)
    ap.add_argument("--seed-start", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable = stable_mod.enabled_stable()
    stats = {h.key: horse_stats(h) for h in stable}
    loaded = backup.load()
    races_dir = config.DATA_DIR / "races"
    tmp_models = config.DATA_DIR / "models" / "practice"
    paths = []
    acc = defaultdict(list)
    conv = defaultdict(list)
    for i in range(args.n):
        seed = args.seed_start + i
        data = dataset.assemble(loaded, None, None, seed, device=device)
        race = Race(stable, data, laps=args.laps, lap_floor_s=0.0, seed=seed, device=device)
        t = time.perf_counter()
        events = list(race.run())
        race_st = scoring.race_standings(events)
        ms = {k: bench_cpu(export_onnx(m, tmp_models / f"{k}.onnx")) for k, m in race.models.items()}
        sprint_st = scoring.sprint_standings(ms, device="laptop")
        champ = scoring.champion_table(race_st, sprint_st)
        facts = compute_facts(events, stats, race_st, sprint_st, champ)
        for k, f in facts.items():
            acc[k].append(f.final_acc)
            conv[k].append(f.convergence_lap)
        p = recorder.write_race(races_dir / f"practice-{seed}.json", events, race_st, sprint_st, champ, {}, {"seed": seed, "practice": True})
        paths.append(p)
        print(f"race {i + 1}/{args.n} seed {seed}: winner {race_st[0].horse} ({race_st[0].final_acc:.2f}), "
              f"champion {champ[0].horse}, {time.perf_counter() - t:.0f}s")

    guide = recorder.form_guide(paths)
    (config.DATA_DIR / "form_guide.json").write_text(__import__("json").dumps(guide, indent=1), encoding="utf-8")
    print("\nhorse           race wins  champ wins  mean acc  mean stride lap")
    for h in stable:
        g = guide[h.key]
        print(f"{h.key:14s} {g['race_wins']:9d} {g['champion_wins']:11d} {sum(acc[h.key]) / len(acc[h.key]):9.3f} "
              f"{sum(conv[h.key]) / len(conv[h.key]):15.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run it, balance, and repeat until the table is fair**

Run: `python tools/practice_races.py --n 20`
Expected: twenty lines then the table. The balancing rule from the spec: no horse wins the Race more than about 7 of 20, and the Sprint order is a true speed order (FEATHERWEIGHT fastest).

How to adjust, one knob at a time, then re-run:
- A horse wins too often: reduce its widths by about a quarter in `stable.py` (for example TEXTBOOK 24/48/96 to 16/32/64). Never change its block type.
- A horse never wins and sits far below the others: increase its widths by about a quarter.
- STAIRS and ELEVATOR show the same stride lap: increase depth with `DEEP_STAGES = [(24, 1), (48, 2), (96, 3)]` and re-run; if STAIRS then finishes far below everyone, step back one notch. Update the STAIRS tagline's floor count to match `stats.depth(build_stairs()) - 1`.
- THE INFLUENCER is allowed to lose; it is the comedy lane. It should still beat chance (above 0.17).

Each adjustment is a commit with the table pasted in the message. Stop when the rule holds. Keep the final `data/form_guide.json`.

- [ ] **Step 8: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/tools/practice_races.py neural-derby/derby/horses/veteran.py neural-derby/derby/horses/stable.py neural-derby/derby/server/app.py neural-derby/tests/test_veteran.py neural-derby/data/form_guide.json
git commit -m "Add practice races with the form guide, the balanced stable, and the Veteran epilogue"
```

---

### Task 15: HTTPS team-phone listener and the Sprint round-trip

**Files:**
- Create: `neural-derby/tools/make_cert.py`, `neural-derby/derby/server/main.py`
- Modify: `neural-derby/tests/test_app.py` (add the sprint round-trip test)

**Interfaces:**
- Produces: `data/cert.pem` and `data/key.pem` (git-ignored) with the LAN IP and `localhost` as subject alternative names; `python -m derby.server.main` serving HTTP on 8000 and, when the certificate exists, HTTPS on 8443 from the same app and event loop.

- [ ] **Step 1: Add the sprint round-trip test**

Append to `neural-derby/tests/test_app.py`:

```python
def test_sprint_results_produce_champion_and_bet_settlement(client: TestClient):
    with client.websocket_connect("/ws?role=stadium") as ws:
        ws.receive_json()
        for _ in range(3):
            client.post("/api/advance")
        client.post("/api/join", json={"device": "dev_0001", "nick": "Tejas"})
        client.post("/api/bet", json={"device": "dev_0001", "horse": "featherweight"})
        import time
        for _ in range(200):
            if client.get("/api/state").json()["data_loaded"]:
                break
            time.sleep(0.05)
        client.post("/api/advance")                                  # DERBY
        while ws.receive_json()["type"] != "race_done":
            pass
        client.post("/api/advance")                                  # SPRINT (sprint_request goes to team/control only)
        client.post("/api/sprint-result", json={"horse": "textbook", "ms": 20.0, "device": "phone"})
        r = client.post("/api/sprint-result", json={"horse": "featherweight", "ms": 10.0, "device": "phone"})
        assert r.json()["have"] == 2
        msg = ws.receive_json()
        while msg["type"] != "sprint":
            msg = ws.receive_json()
        assert msg["device"] == "phone"
        assert msg["sprint"][0]["horse"] == "featherweight"
        assert msg["champion"][0]["rank"] == 1
        assert "Tejas" in msg["winners"]["champion"] or "Tejas" in msg["winners"]["race"] or msg["winners"] == {"champion": [], "race": []}
        client.post("/api/advance")                                  # VERDICT
        msg = ws.receive_json()
        while msg["type"] != "verdict":
            msg = ws.receive_json()
        assert set(msg["verdicts"]) == {"textbook", "featherweight"}
        client.post("/api/advance")                                  # SHIP
        msg = ws.receive_json()
        while msg["type"] != "ship":
            msg = ws.receive_json()
        assert client.get("/models/champion.onnx").status_code == 200
```

The `winners` assertion accepts either outcome because which horse wins the Race depends on training randomness; the important checks are the Sprint ordering, that a champion exists, and that settlement ran.

Run: `pytest tests/test_app.py` → `5 passed`.

- [ ] **Step 2: Write make_cert.py**

`neural-derby/tools/make_cert.py`:

```python
"""Make a self-signed certificate so the TEAM phone can use live video (browsers need HTTPS for the camera).

Usage: python tools/make_cert.py            (writes data/cert.pem and data/key.pem)
On the team phone, open https://<laptop ip>:8443/phone/?team=1 once and accept the warning.
"""
import datetime
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402
from derby.server.app import lan_ip  # noqa: E402


def main():
    ip = lan_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Neural Derby")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
            .sign(key, hashes.SHA256()))
    (config.DATA_DIR / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    (config.DATA_DIR / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"wrote data/cert.pem and data/key.pem for {ip} and localhost")


if __name__ == "__main__":
    main()
```

Run: `python tools/make_cert.py` → prints the IP it was made for. Re-run whenever the laptop's IP changes (new network).

- [ ] **Step 3: Write main.py**

`neural-derby/derby/server/main.py`:

```python
"""Start the show: HTTP on 8000 for the room, HTTPS on 8443 for the team phone, one app, one event loop."""
import asyncio

import uvicorn

from derby import config
from derby.server.app import create_app


async def serve():
    app = create_app()
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.PORTS["http"], log_level="info"))]
    cert, key = config.DATA_DIR / "cert.pem", config.DATA_DIR / "key.pem"
    if cert.exists() and key.exists():
        servers.append(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.PORTS["https"], log_level="info",
                                                     ssl_certfile=str(cert), ssl_keyfile=str(key))))
    else:
        print("no data/cert.pem: HTTPS listener off (team phone live video unavailable). Run tools/make_cert.py.")
    await asyncio.gather(*(s.serve() for s in servers))


def run():
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Prove the round-trip on a real Android phone**

1. Join the laptop and the phone to the chosen network (spec section 9 order: campus Wi-Fi test first, then an Android hotspot, then a router; Windows hotspot last and only for up to 8 devices).
2. `python tools/make_cert.py`, then `python -m derby.server.main`. Expected log: two `Uvicorn running on` lines, ports 8000 and 8443.
3. On the phone: `http://<ip>:8000/p` loads the phone page (crowd mode). Take a photo with the object button; the tile appears on the stadium.
4. On the phone: `https://<ip>:8443/phone/?team=1`, accept the warning. Banner must read `TEAM PHONE CONNECTED`.
5. Run the show through SPRINT. Expected: the phone shows "timing TEXTBOOK..." through all horses, the stadium's Sprint bars say `TIMED ON: PHONE`. Write the seven millisecond figures into the commit message.
6. Advance to SHIP. Expected: the team phone shows live video with a prediction that changes as you point at a bottle, a pen, a shoe. A crowd phone (plain http) shows the POINT AT YOUR DESK button and classifies a snapshot.
7. Turn the laptop's internet off and repeat step 3. Expected: identical behaviour (no external resource is loaded).

- [ ] **Step 5: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/tools/make_cert.py neural-derby/derby/server/main.py neural-derby/tests/test_app.py
git commit -m "Add the HTTPS team-phone listener and prove the phone Sprint round-trip"
```

---

### Task 16: Learning docs, demo script, rehearsal

**Files:**
- Create: `neural-derby/docs/learning/00-overview.md`, `neural-derby/docs/learning/01-race-engine.md`, `neural-derby/docs/learning/02-the-stable.md`, `neural-derby/docs/learning/03-the-grandstand.md`, `neural-derby/docs/learning/qa-drill.md`, `neural-derby/docs/demo-script.md`
- Modify: `neural-derby/docs/setup.md` (add the run-of-show keys)

**Interfaces:**
- Consumes: the final numbers from Task 3 Step 7 (params, multiplies, depth per horse), the final table from Task 14 Step 7, and the phone millisecond figures from Task 15 Step 4. Every number in the docs comes from one of those runs; none are invented.
- Produces: documents a 15-year-old can read, one chapter per role, a question drill, and a minute-by-minute script.

Writing rules for every file: short sentences, everyday words, each technical word explained in one line the first time it appears, no acronyms without expansion, numbers only from the runs above.

- [ ] **Step 1: Write 00-overview.md**

Sections, in this order, each with the content named:

1. **What the room sees** (one paragraph per act, taken from spec section 3, in present tense).
2. **The one-sentence version** ("Everyone takes photos of their stuff. Seven tiny AI brains race to learn what's in the photos. You bet on one. The winner gets sent to your phone.").
3. **Which act proves which module**, as a table: Act 1 and 2 = Module 1 (we chose classification) and Module 4 (augmentation); Act 3 = Module 3 (the blocks); Act 4 = Modules 2 and 5 (training loop, Adam, loss); Act 5 = Module 7 (speed versus accuracy on a phone); Act 6 = Module 2 (transfer learning via the Veteran) and Module 7 (edge deployment).
4. **Glossary** with one line each: photo, label, class, dataset, epoch (a lap), batch, parameter (a weight the network learns), accuracy, loss, optimiser (Adam), augmentation, convolution, pooling, skip connection, attention, depthwise convolution, inception block, transformer, multiply-accumulate, ONNX (a file format for a trained network that other programs can run), on-device (the phone does the thinking itself), edge AI.
5. **How a lap works, in words** (eight numbered steps from shuffle to evaluate).
6. **Why this is honest** (nothing is scripted: the caller and verdicts are rules over measured numbers; the form guide is from recorded practice races).

- [ ] **Step 2: Write 01-race-engine.md (Ranvir's chapter)**

Sections: the lock-step idea and why it is fair; a walk through `Race._train_one_lap` line by line in words; where augmentation happens and why it is on the GPU; evaluation and the three accuracies (all, internet, room); pacing and why laps have a floor; events to websocket to stadium (data flow from spec section 11); recording and replay as the safety net; the Race, Sprint, Champion rules with a worked example using the real standings from one practice race; what breaks and what the fallback is (spec section 12). End with ten likely questions and crisp answers, including: "why Adam for every horse", "why the same shuffle order", "what is the loss function", "what happens if two horses tie", "what is in the JSON file".

- [ ] **Step 3: Write 02-the-stable.md (Tejas's chapter)**

Sections: the shared head and why only the body differs; one subsection per horse with the block in words, a small ASCII block diagram, the real parameter, multiply, and depth numbers from Task 3 Step 7, the tagline, and the honest "when it wins, when it loses" from the practice table; STAIRS versus ELEVATOR as the headline contrast with the stride-lap numbers; the balanced form guide table from Task 14; a worked parameter-count example in ma'am's exam style (image 200x200x3, five conv layers with 16, 16, 16, 32, 32 filters of 3x3: show the multiplication for each layer and the total; then the follow-ups: maxpool versus stride both have zero parameters, so parameters are equal and only compute differs; channel attention adds two small layers while depthwise conv removes most of the 3x3 cost; changing all filters to 1x1 keeps channel mixing but loses neighbourhoods). End with ten likely questions, including "how did you count multiplies", "why does the transformer lose", "what does the sigmoid do in attention", "what would you change to make SPOTLIGHT win".

- [ ] **Step 4: Write 03-the-grandstand.md (Harshit's chapter)**

Sections: the phone page flow screen by screen; why the photo button works on plain http but live video needs https; the device token and why no personal data is stored; caps and the per-device split and why a person's photos never sit on both sides; the six augmentations with one sentence each on what it simulates (tilted phone, bad light, shaky hand, clutter) and the exact ranges used; the network options with the 8-device Windows hotspot limit and the test you ran; ONNX export and the match check; how the phone runs the model (onnxruntime-web, WebAssembly, single thread) and the preprocessing it must mirror; the Sprint timing method (median of 30, warm-up of 3) and the real millisecond table from Task 15; the Veteran and what it proves. End with ten likely questions, including "why not a native app", "why median not mean", "what is WebAssembly", "why 96 pixels".

- [ ] **Step 5: Write qa-drill.md and demo-script.md**

`qa-drill.md`: 30 questions grouped by person (10 each), each with a two-to-four sentence answer and the name of the file or screen that proves it. Include the five ma'am asks in the study map: task type, data source, which pretrained network, size versus speed, train-from-scratch versus fine-tune versus transfer.

`demo-script.md`: a table with columns minute, act, who speaks, what they say (one or two sentences, verbatim), what the operator presses, fallback if it fails. Total under 12 minutes. Opening line belongs to Ranvir; the horse cards to Tejas; the phone and Sprint to Harshit; the verdicts to Tejas; the Veteran and close to Ranvir.

Also append to `docs/setup.md`:

```markdown
## Keys on the stadium page
Space = next act. G = close the gate. L = laptop Sprint timing (fallback). R = replay the last recorded race (fallback). M = mute. Left/Right = page the verdict cards.
```

- [ ] **Step 6: Read-aloud check and rehearsal**

Read each chapter aloud to someone outside the team (or to a teammate who did not write it). Every sentence they stop on gets rewritten. Then the full rehearsal with three phones, timing each act against the script; adjust `LAP_FLOOR_S` in `config.py` if the Derby runs long or short. Tick the prep checklist in spec section 15.

- [ ] **Step 7: Commit**

```powershell
cd C:\dev\study\Edge-Ai
git add neural-derby/docs
git commit -m "Add the learning chapters, question drill, demo script, and run-of-show keys"
```

---

## Self-review notes

- Spec coverage: acts 1 to 6 (Tasks 11 to 13), seven horses (Tasks 2 and 3), data and split rules (Tasks 4 to 6), race rules and scoring (Tasks 7 and 8), caller and verdicts (Task 9), export and phone inference (Tasks 10 and 13), network and HTTPS (Task 15), UI direction and forgotten states (Tasks 12 and 13), failure handling (replay, laptop timing, gates, caps: Tasks 8, 11, 13), tests (every task), roles and docs (Task 16), prep checklist (Tasks 4, 7, 12, 14, 15, 16). The spec's "room code" is implemented as the short `/p` path.
- Type consistency: accuracies are fractions everywhere; `LapEvent` fields are identical in trainer, scoring, recorder, caller, verdict, and the JavaScript (`event.test_acc`); horse keys are the seven lowercase strings throughout; `Settings` and `AppState` names match between `app.py`, `main.py`, and the tests.
- Known judgement calls left to execution: exact widths after balancing (Task 14), the STAIRS tagline floor count after balancing, and the augmentation ranges if the contact sheet looks too strong (Task 6 Step 7).
