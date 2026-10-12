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
