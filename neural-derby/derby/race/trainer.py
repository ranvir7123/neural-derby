"""The race engine. One lap = one epoch for every horse, in lock-step, with the same shuffle order.

The trainer knows nothing about the web. It yields LapEvents; whoever calls it decides what to do with them.
"""
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from derby import config
from derby.data.augment import augment, normalize, to_float
from derby.data.dataset import RaceData
from derby.horses.stable import Horse


@dataclass(frozen=True)
class LapEvent:
    horse: str
    lap: int
    train_acc: float
    test_acc: float
    test_acc_internet: float | None
    test_acc_live: float | None
    loss: float
    ms: float      # wall time this horse spent on this lap
    t: float       # seconds since the race started

    def to_dict(self) -> dict:
        return asdict(self)


class Race:
    def __init__(self, stable: list[Horse], data: RaceData, laps: int = config.LAPS, batch: int = config.BATCH,
                 lap_floor_s: float = config.LAP_FLOOR_S, seed: int = config.SEED, lr: float = config.LR,
                 device=None):
        self.stable = stable
        self.data = data
        self.laps = laps
        self.batch = batch
        self.lap_floor_s = lap_floor_s
        self.seed = seed
        self.device = torch.device(device) if device is not None else data.train_x.device
        self.models: dict[str, nn.Module] = {}
        self.opts: dict[str, torch.optim.Optimizer] = {}
        for h in stable:
            torch.manual_seed(seed + zlib.crc32(h.key.encode()))
            m = h.build().to(self.device)
            self.models[h.key] = m
            self.opts[h.key] = torch.optim.Adam(m.parameters(), lr=lr)
        self.events: list[LapEvent] = []

    def run(self) -> Iterator[LapEvent]:
        t0 = time.perf_counter()
        n = self.data.n_train
        aug_gen = torch.Generator(device=self.device)
        for lap in range(1, self.laps + 1):
            lap_start = time.perf_counter()
            order = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed * 1000 + lap)).to(self.device)
            for i, h in enumerate(self.stable):
                aug_gen.manual_seed(self.seed * 100_000 + lap * 100 + i)
                ev = self._train_one_lap(h.key, order, aug_gen, lap, t0)
                self.events.append(ev)
                yield ev
            if lap < self.laps:
                remaining = self.lap_floor_s - (time.perf_counter() - lap_start)
                if remaining > 0:
                    time.sleep(remaining)

    def _train_one_lap(self, key: str, order: torch.Tensor, gen: torch.Generator, lap: int, t0: float) -> LapEvent:
        m, opt = self.models[key], self.opts[key]
        m.train()
        start = time.perf_counter()
        n = self.data.n_train
        correct = 0
        loss_sum = 0.0
        for i in range(0, n, self.batch):
            idx = order[i:i + self.batch]
            x = normalize(augment(self.data.train_x[idx], gen))
            y = self.data.train_y[idx]
            logits = m(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += int((logits.argmax(1) == y).sum().item())
            loss_sum += float(loss.item()) * int(idx.shape[0])
        test = self.evaluate(key)
        now = time.perf_counter()
        return LapEvent(key, lap, correct / n, test["all"], test["internet"], test["live"],
                        loss_sum / n, (now - start) * 1000.0, now - t0)

    @torch.no_grad()
    def evaluate(self, key: str) -> dict:
        m = self.models[key]
        m.eval()
        preds = []
        for i in range(0, self.data.n_test, 256):
            x = normalize(to_float(self.data.test_x[i:i + 256]))
            preds.append(m(x).argmax(1))
        hit = torch.cat(preds) == self.data.test_y
        live = self.data.test_is_live

        def acc(mask):
            return float(hit[mask].float().mean().item()) if bool(mask.any()) else None

        return {"all": float(hit.float().mean().item()), "internet": acc(~live), "live": acc(live)}
