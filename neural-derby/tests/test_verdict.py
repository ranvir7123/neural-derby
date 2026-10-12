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


def lines_for(events, ms, horse):
    return V.verdicts(build(events, ms), NAMES, 600)[horse]


def test_memorising_claim_at_exactly_twenty_points_but_not_at_nineteen():
    exactly = lines_for(curve("textbook", [0.8], trains=[1.0]), {"textbook": 5}, "textbook")
    assert any("memoris" in s for s in exactly)
    nineteen = lines_for(curve("textbook", [0.8], trains=[0.99]), {"textbook": 5}, "textbook")
    assert not any("memoris" in s for s in nineteen)


def test_room_is_harder_at_exactly_ten_points_but_not_at_nine():
    at_ten = lines_for([ev("textbook", 1, 0.7, live=0.6)], {"textbook": 5}, "textbook")
    assert any("harder" in s for s in at_ten)
    at_nine = lines_for([ev("textbook", 1, 0.7, live=0.61)], {"textbook": 5}, "textbook")
    assert not any("harder" in s for s in at_nine)


def test_elevator_slower_than_stairs_is_not_called_the_same_pace():
    events = curve("stairs", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) + curve("elevator", [0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    v = V.verdicts(build(events, {"stairs": 10, "elevator": 10}), NAMES, n_train=600)
    assert any("reached its stride on lap 1" in s and "took until lap 6" in s and "plain stairs set the pace" in s
               for s in v["stairs"])
    assert any("did not pay off" in s and "lap 6 against STAIRS's lap 1" in s for s in v["elevator"])
    for key in ("stairs", "elevator"):
        assert not any("did not change the pace" in s or "about the same pace" in s for s in v[key])


def test_elevator_same_pace_sentence_still_used_when_laps_are_close():
    close = curve("stairs", [0.6, 0.8, 0.9, 0.9]) + curve("elevator", [0.6, 0.85, 0.9, 0.9])
    v = V.verdicts(build(close, {"stairs": 10, "elevator": 10}), NAMES, n_train=600)
    assert any("about the same pace" in s for s in v["elevator"])
    assert not any("did not pay off" in s or "paid off" in s for s in v["elevator"])


def test_small_sprint_times_keep_a_decimal_and_big_ones_do_not():
    events = curve("featherweight", [0.7, 0.8]) + curve("textbook", [0.7, 0.85])
    small = lines_for(events, {"featherweight": 1.2, "textbook": 1.4}, "featherweight")
    assert any("1.2 ms against 1.4 ms" in s for s in small)
    assert "(1.2 ms)" in small[0]
    big = lines_for(events, {"featherweight": 25.0, "textbook": 30.0}, "featherweight")
    assert any("25 ms against 30 ms" in s for s in big)
    assert "(25 ms)" in big[0]
    slower = lines_for(events, {"featherweight": 1.4, "textbook": 1.2}, "featherweight")
    assert any("(1.4 ms vs 1.2 ms)" in s for s in slower)


def test_influencer_is_only_blamed_for_too_little_data_when_it_trails_textbook():
    trails = curve("influencer", [0.5, 0.6]) + curve("textbook", [0.7, 0.85])
    below = lines_for(trails, {"influencer": 5, "textbook": 5}, "influencer")
    assert any("not enough" in s for s in below)
    wins = curve("influencer", [0.7, 0.9]) + curve("textbook", [0.5, 0.6])
    above = lines_for(wins, {"influencer": 5, "textbook": 5}, "influencer")
    assert not any("not enough" in s for s in above)
    assert any("kept up with the convolution horses anyway" in s for s in above)


def test_influencer_with_no_textbook_to_compare_makes_no_claim_either_way():
    alone = lines_for(curve("influencer", [0.5, 0.6]), {"influencer": 5}, "influencer")
    assert not any("not enough" in s or "kept up" in s for s in alone)
