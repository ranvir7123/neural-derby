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
