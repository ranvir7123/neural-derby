"""Why each horse finished where it did, said only when the numbers back it up."""
from collections import defaultdict
from dataclasses import asdict, dataclass

from derby.race.scoring import ChampionStanding, SprintStanding, Standing
from derby.race.trainer import LapEvent

LAP_GAP = 3            # laps of difference before we call a slow start
MEMORISE_GAP = 0.20    # train minus test before we call memorising
ROOM_GAP = 0.10        # internet minus room accuracy before we call the room's photos harder
CONVERGED_FRAC = 0.9   # a horse "reached its stride" at 90 percent of its own final accuracy


@dataclass(frozen=True)
class Facts:
    horse: str
    params: int
    macs: int
    depth: int
    final_acc: float
    acc_internet: float | None
    acc_live: float | None
    convergence_lap: int
    max_gap: float
    sprint_ms: float | None
    race_rank: int
    sprint_rank: int
    champion_rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def pct(x: float) -> str:
    return f"{round(x * 100)} percent"


def compute_facts(events: list[LapEvent], stats_by_horse: dict[str, dict], race: list[Standing],
                  sprint: list[SprintStanding], champion: list[ChampionStanding]) -> dict[str, Facts]:
    by: dict[str, list[LapEvent]] = defaultdict(list)
    for e in events:
        by[e.horse].append(e)
    race_rank = {s.horse: s.rank for s in race}
    sprint_rank = {s.horse: s.rank for s in sprint}
    sprint_ms = {s.horse: s.ms for s in sprint}
    champ_rank = {c.horse: c.rank for c in champion}
    out = {}
    for horse, evs in by.items():
        evs = sorted(evs, key=lambda e: e.lap)
        last = evs[-1]
        target = last.test_acc * CONVERGED_FRAC
        convergence_lap = next(e.lap for e in evs if e.test_acc >= target - 1e-9)
        max_gap = max(e.train_acc - e.test_acc for e in evs)
        st = stats_by_horse.get(horse, {"params": 0, "macs": 0, "depth": 0})
        out[horse] = Facts(
            horse, st["params"], st["macs"], st["depth"], last.test_acc, last.test_acc_internet, last.test_acc_live,
            convergence_lap, max_gap, sprint_ms.get(horse), race_rank.get(horse, 0),
            sprint_rank.get(horse, len(race) + 1), champ_rank.get(horse, 0),
        )
    return out


def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def verdicts(facts: dict[str, Facts], names: dict[str, str], n_train: int) -> dict[str, list[str]]:
    def nm(k):
        return names.get(k, k.upper())

    out: dict[str, list[str]] = {}
    tb = facts.get("textbook")
    for key, f in facts.items():
        lines: list[str] = []
        ms = f"{f.sprint_ms:.0f} ms" if f.sprint_ms is not None else "no time"
        lines.append(f"Finished {ordinal(f.race_rank)} in the Derby at {pct(f.final_acc)}, "
                     f"{ordinal(f.sprint_rank)} in the Sprint ({ms}), {ordinal(f.champion_rank)} overall.")
        lines.append(f"{f.params:,} parameters, {f.macs:,} multiplies per photo, {f.depth} weight layers.")

        if key == "stairs" and "elevator" in facts:
            e = facts["elevator"]
            if f.convergence_lap - e.convergence_lap >= LAP_GAP:
                lines.append(f"{nm(key)} reached its stride on lap {f.convergence_lap}; {nm('elevator')} got there on lap "
                             f"{e.convergence_lap}. Same depth, one shortcut.")
            else:
                lines.append(f"At this depth the shortcuts did not change the pace (lap {f.convergence_lap} vs lap "
                             f"{e.convergence_lap}). Deeper buildings need elevators more.")
        if key == "elevator" and "stairs" in facts:
            s = facts["stairs"]
            if s.convergence_lap - f.convergence_lap >= LAP_GAP:
                lines.append(f"The shortcuts paid off: stride on lap {f.convergence_lap} against {nm('stairs')}'s lap "
                             f"{s.convergence_lap}, with exactly the same parameters.")
            else:
                lines.append(f"Same parameters as {nm('stairs')} and about the same pace this time "
                             f"(lap {f.convergence_lap} vs lap {s.convergence_lap}).")
        if key == "featherweight" and tb is not None:
            if f.sprint_ms is not None and tb.sprint_ms is not None and f.sprint_ms < tb.sprint_ms:
                lines.append(f"{f.macs:,} multiplies against {nm('textbook')}'s {tb.macs:,}. That is why it ran the Sprint in "
                             f"{f.sprint_ms:.0f} ms against {tb.sprint_ms:.0f} ms.")
            elif f.sprint_ms is not None and tb.sprint_ms is not None:
                lines.append(f"{f.macs:,} multiplies against {nm('textbook')}'s {tb.macs:,}, yet it was not faster on this device "
                             f"({f.sprint_ms:.0f} ms vs {tb.sprint_ms:.0f} ms). Fewer multiplies only help when the runtime "
                             f"is tuned for depthwise layers.")
        if key == "spotlight" and tb is not None:
            delta = f.final_acc - tb.final_acc
            sign = "+" if delta >= 0 else "-"
            lines.append(f"Attention added {f.params - tb.params:,} parameters over {nm('textbook')} and changed accuracy by "
                         f"{sign}{abs(round(delta * 100))} points.")
        if key == "trifocals" and tb is not None:
            lines.append(f"Three lens sizes cost {f.macs:,} multiplies against {nm('textbook')}'s {tb.macs:,}; "
                         f"accuracy {pct(f.final_acc)} against {pct(tb.final_acc)}.")
        if key == "influencer":
            lines.append(f"{pct(f.final_acc)} with {n_train:,} photos. Transformers have no built-in idea that nearby pixels "
                         f"belong together; they must learn it from data, and there was not enough.")
        if f.max_gap >= MEMORISE_GAP:
            lines.append(f"It memorised: at its worst, {pct(f.max_gap)} more on photos it had seen than on new ones.")
        if f.acc_internet is not None and f.acc_live is not None:
            line = f"On internet photos {pct(f.acc_internet)}; on the room's photos {pct(f.acc_live)}."
            if f.acc_internet - f.acc_live >= ROOM_GAP:
                line += " The room's photos are harder: different light, different backgrounds."
            lines.append(line)
        out[key] = lines
    return out
