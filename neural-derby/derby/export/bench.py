"""Laptop-side timing, the fallback when the team phone is not reachable for the Sprint."""
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from derby import config


def bench_cpu(path: Path, n: int = 30, img: int = config.IMG) -> float:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # one core, like a phone browser without threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img, img).astype(np.float32)
    for _ in range(3):
        sess.run(["logits"], {"image": x})
    times = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(["logits"], {"image": x})
        times.append((time.perf_counter() - t) * 1000.0)
    return float(statistics.median(times))
