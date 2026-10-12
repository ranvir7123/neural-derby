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
