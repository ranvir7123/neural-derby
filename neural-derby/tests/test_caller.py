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


def said(lines):
    return " ".join(lines).lower()


def test_surge_fires_at_exactly_eight_points_but_not_below():
    on = feed(Caller(["a"], NAMES, laps_total=10, seed=0), [ev("a", 1, 0.40), ev("a", 2, 0.48)])
    assert "surge" in said(on[1])
    below = feed(Caller(["a"], NAMES, laps_total=10, seed=0), [ev("a", 1, 0.40), ev("a", 2, 0.479)])
    assert "surge" not in said(below[1])


def test_stall_needs_a_spread_under_one_point_so_exactly_one_point_is_no_wall():
    exactly_one = feed(Caller(["a"], NAMES, laps_total=10, seed=0), [ev("a", 1, 0.40), ev("a", 2, 0.40), ev("a", 3, 0.41)])
    assert "wall" not in said(exactly_one[2])
    just_under = feed(Caller(["a"], NAMES, laps_total=10, seed=0), [ev("a", 1, 0.40), ev("a", 2, 0.40), ev("a", 3, 0.409)])
    assert "wall" in said(just_under[2])


def test_memorising_fires_at_exactly_twenty_points_but_not_below():
    on = feed(Caller(["a"], NAMES, laps_total=10, seed=0), [ev("a", 1, 0.80, train_acc=1.00)])
    assert "memoris" in said(on[0])
    below = feed(Caller(["a"], NAMES, laps_total=10, seed=0), [ev("a", 1, 0.81, train_acc=1.00)])
    assert "memoris" not in said(below[0])


def finish_lines(first_acc, second_acc):
    """Run a two-horse, two-lap race and return what the caller said when the last horse crossed the line."""
    c = Caller(["a", "b"], NAMES, laps_total=2, seed=0)
    out = feed(c, [ev("a", 1, 0.5), ev("b", 1, 0.4), ev("a", 2, first_acc), ev("b", 2, second_acc)])
    return said(out[3])


def test_photo_finish_is_under_two_points_so_exactly_two_points_is_a_clear_win():
    clear = finish_lines(0.70, 0.68)                    # exactly 2 points
    assert "wins" in clear and "photo finish" not in clear
    assert "bravo second, 2 points back at 68 percent" in clear      # the clear win states its margin
    photo = finish_lines(0.80, 0.781)                   # 1.9 points
    assert "wins" in photo and "photo finish" in photo


def test_start_line_counts_the_horses_that_are_really_racing():
    found = False
    for seed in range(21):
        first = Caller(["a", "b"], NAMES, laps_total=4, seed=seed).on_event(ev("a", 1, 0.5))[0]
        if "brains" in first:
            found = True
            assert "2 brains" in first
    assert found
