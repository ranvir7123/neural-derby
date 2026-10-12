"""Turn a trained horse into an ONNX file the phone can run, and prove it gives the same answers."""
import copy
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from derby import config


class ExportMismatch(RuntimeError):
    pass


def export_onnx(model: nn.Module, path: Path, img: int = config.IMG) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, 3, img, img)
    kwargs = dict(input_names=["image"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(m, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(m, dummy, str(path), **kwargs)
    x = torch.randn(2, 3, img, img)
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"image": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    if diff > 1e-4:
        raise ExportMismatch(f"onnx output differs from torch by {diff}")
    return path
