"""The race-caller. Every line comes from a rule on real numbers; only the wording is random."""
import random

from derby.race.scoring import race_standings
from derby.race.trainer import LapEvent

SURGE_POINTS = 0.08
STALL_POINTS = 0.01
MEMORISE_POINTS = 0.20
PHOTO_FINISH_POINTS = 0.02
EPS = 1e-9  # decimal fractions are not exact in floating point (1.0 - 0.8 is 0.19999999999999996), so thresholds get a tiny allowance

# Every phrasing of a rule contains that rule's keyword (lead, overtakes, surges, wall, memorising, halfway,
# last lap, wins, photo finish) so the ticker, the voice, and the tests all agree on what fired.
START = ["And they're off!", "The gates open. They're off!", "{n} brains, one dataset. They're off!"]
FIRST_LEADER = ["{h} takes the early lead at {acc}.", "First out of the gate: {h} leads at {acc}."]
LEAD_CHANGE = ["{y} overtakes {x}!", "New leader! {y} overtakes {x}.", "{x} loses the lead as {y} overtakes!"]
SURGE = ["{h} surges forward, up to {acc}!", "Big lap for {h}, it surges to {acc}!", "{h} finds another gear and surges to {acc}."]
STALL = ["{h} has hit a wall at {acc}.", "{h} is stuck at a wall, {acc}. Three laps, no progress.", "Nothing new from {h}, a wall at {acc}."]
MEMORISING = ["{h} is memorising the photos instead of learning. {train} on photos it has seen, {test} on new ones.",
              "Careful, {h}: {train} in training but only {test} on new photos. That is memorising."]
HALFWAY = ["Halfway there. {a} leads at {acc}, {b} second, {c} third.", "Halfway, lap {lap} of {total}: {a} in front at {acc}, then {b} and {c}."]
HALFWAY_TWO = ["Halfway there. {a} leads at {acc}, {b} chasing.", "Halfway, lap {lap} of {total}: {a} in front at {acc}, {b} behind."]
FINAL_LAP = ["Last lap! {a} by {margin} over {b}.", "Last lap. {a} leads {b} by {margin}."]
FINISH = ["{a} wins the Derby at {acc}!", "It's {a}! {a} wins the Derby at {acc}!"]
PHOTO = ["Photo finish! {a} by just {margin} over {b}.", "Too close to call... photo finish says {a} by {margin}!"]
CLEAR = ["{b} second, {margin} back at {bacc}."]


def pct(fraction: float) -> str:
    return f"{round(fraction * 100)} percent"


def points(diff: float) -> str:
    p = round(diff * 100)
    return f"{p} point" if p == 1 else f"{p} points"


class Caller:
    def __init__(self, horses: list[str], names: dict[str, str], laps_total: int, seed: int | None = None):
        self.horses = list(horses)
        self.names = names
        self.laps_total = laps_total
        self.rng = random.Random(seed)
        self.history: dict[str, dict[int, LapEvent]] = {h: {} for h in horses}
        self.leader: str | None = None
        self.started = False
        self.memorised: set[str] = set()
        self.last_stall_lap: dict[str, int] = {}

    def name(self, key: str) -> str:
        return self.names.get(key, key.upper())

    def say(self, options: list[str], **kw) -> str:
        return self.rng.choice(options).format(**kw)

    def on_event(self, ev: LapEvent) -> list[str]:
        lines: list[str] = []
        if not self.started:
            self.started = True
            lines.append(self.say(START, n=len(self.horses)))
        self.history.setdefault(ev.horse, {})[ev.lap] = ev
        lines += self._horse_rules(ev)
        if all(ev.lap in self.history.get(h, {}) for h in self.horses):
            lines += self._lap_rules(ev.lap)
        return lines

    def _horse_rules(self, ev: LapEvent) -> list[str]:
        lines = []
        h = self.name(ev.horse)
        hist = self.history[ev.horse]
        prev = hist.get(ev.lap - 1)
        if prev is not None and ev.test_acc - prev.test_acc >= SURGE_POINTS - EPS:
            lines.append(self.say(SURGE, h=h, acc=pct(ev.test_acc)))
        three = [hist.get(lap) for lap in (ev.lap - 2, ev.lap - 1, ev.lap)]
        if all(e is not None for e in three) and ev.lap < self.laps_total:
            accs = [e.test_acc for e in three]
            recently = ev.lap - self.last_stall_lap.get(ev.horse, -10) < 3
            if max(accs) - min(accs) < STALL_POINTS - EPS and not recently:
                self.last_stall_lap[ev.horse] = ev.lap
                lines.append(self.say(STALL, h=h, acc=pct(ev.test_acc)))
        if ev.train_acc - ev.test_acc >= MEMORISE_POINTS - EPS and ev.horse not in self.memorised:
            self.memorised.add(ev.horse)
            lines.append(self.say(MEMORISING, h=h, train=pct(ev.train_acc), test=pct(ev.test_acc)))
        return lines

    def _lap_rules(self, lap: int) -> list[str]:
        lines = []
        events = [e for h in self.horses for e in self.history[h].values() if e.lap <= lap]
        st = race_standings(events)
        leader = st[0].horse
        if self.leader is None:
            lines.append(self.say(FIRST_LEADER, h=self.name(leader), acc=pct(st[0].final_acc)))
        elif leader != self.leader:
            lines.append(self.say(LEAD_CHANGE, y=self.name(leader), x=self.name(self.leader)))
        self.leader = leader
        second = st[1] if len(st) > 1 else None
        margin = (st[0].final_acc - second.final_acc) if second else 0.0
        if lap == self.laps_total // 2 and lap not in (self.laps_total, self.laps_total - 1):
            if len(st) >= 3:
                lines.append(self.say(HALFWAY, a=self.name(st[0].horse), acc=pct(st[0].final_acc),
                                      b=self.name(st[1].horse), c=self.name(st[2].horse), lap=lap, total=self.laps_total))
            elif second:
                lines.append(self.say(HALFWAY_TWO, a=self.name(st[0].horse), acc=pct(st[0].final_acc),
                                      b=self.name(second.horse), lap=lap, total=self.laps_total))
        if lap == self.laps_total - 1 and second:
            lines.append(self.say(FINAL_LAP, a=self.name(leader), b=self.name(second.horse), margin=points(margin)))
        if lap == self.laps_total:
            lines.append(self.say(FINISH, a=self.name(leader), acc=pct(st[0].final_acc)))
            if second:
                if margin < PHOTO_FINISH_POINTS - EPS:
                    lines.append(self.say(PHOTO, a=self.name(leader), b=self.name(second.horse), margin=points(margin)))
                else:
                    lines.append(self.say(CLEAR, b=self.name(second.horse), margin=points(margin), bacc=pct(second.final_acc)))
        return lines
