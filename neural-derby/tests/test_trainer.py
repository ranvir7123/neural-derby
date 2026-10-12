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
