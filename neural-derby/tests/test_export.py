from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from derby.export.bench import bench_cpu
from derby.export.onnx_export import ExportMismatch, export_onnx
from derby.horses import stable


def test_every_horse_exports_and_matches(tmp_path: Path):
    x = torch.randn(2, 3, 96, 96)
    for h in stable.STABLE:
        m = h.build().eval()
        p = export_onnx(m, tmp_path / f"{h.key}.onnx")
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        out = sess.run(["logits"], {"image": x.numpy()})[0]
        with torch.no_grad():
            ref = m(x).numpy()
        assert np.abs(out - ref).max() < 1e-4, h.key
        assert sess.get_inputs()[0].name == "image"


def test_export_leaves_original_model_on_its_device_and_mode(tmp_path: Path):
    m = stable.build_textbook().train()
    export_onnx(m, tmp_path / "t.onnx")
    assert m.training


def test_export_raises_when_onnx_disagrees_with_torch(tmp_path: Path):
    class Liar(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 96 * 96, 6)

        def forward(self, x):
            out = self.fc(x.flatten(1))
            # The exporter traces the model, so the saved file gets the extra +1, but the plain torch run does not.
            if torch.jit.is_tracing():
                out = out + 1.0
            return out

    with pytest.raises(ExportMismatch):
        export_onnx(Liar(), tmp_path / "bad.onnx")


def test_bench_cpu_returns_positive_ms(tmp_path: Path):
    p = export_onnx(stable.build_featherweight(), tmp_path / "f.onnx")
    ms = bench_cpu(p, n=5)
    assert ms > 0
