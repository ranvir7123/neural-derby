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


def test_champion_tie_on_rank_sum_goes_to_higher_accuracy():
    # b has the better final accuracy (race rank 1), a is the faster sprinter (sprint rank 1).
    # Both add up to 3, and "b" sorts after "a" by name, so only the accuracy rule can put b first.
    race = scoring.race_standings([ev("a", 1, 0.6), ev("b", 1, 0.7)])
    sprint = scoring.sprint_standings({"a": 10.0, "b": 20.0})
    champ = scoring.champion_table(race, sprint)
    assert [(c.horse, c.total, c.rank) for c in champ] == [("b", 3, 1), ("a", 3, 2)]
