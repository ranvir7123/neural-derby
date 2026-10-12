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
