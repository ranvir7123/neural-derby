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
