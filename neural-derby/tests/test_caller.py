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
