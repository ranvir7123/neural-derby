"""Download the pixel font and onnxruntime-web once (internet needed), into web/vendor/. The show never needs internet."""
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf"
FONT_LICENCE = "https://github.com/google/fonts/raw/main/ofl/pressstart2p/OFL.txt"
ORT_VERSION = "1.20.1"
ORT_TGZ = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
# The npm package has no LICENSE file inside, so take it from the same release on GitHub.
ORT_LICENCE = f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{ORT_VERSION}/LICENSE"
# Only the plain WebAssembly build the phone uses. The ".jsep" files are the much bigger WebGPU build.
ORT_FILES = {"ort.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    vendor = config.WEB_DIR / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "PressStart2P-Regular.ttf").write_bytes(get(FONT_URL))
    (vendor / "PressStart2P-OFL.txt").write_bytes(get(FONT_LICENCE))
    print("font ok")
    ort_dir = vendor / "onnxruntime-web"
    ort_dir.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(get(ORT_TGZ)), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.name.startswith("package/dist/") and name in ORT_FILES:
                (ort_dir / name).write_bytes(tar.extractfile(member).read())
                print("  ", name)
    (ort_dir / "LICENSE").write_bytes(get(ORT_LICENCE))
    print("   LICENSE")
    print("onnxruntime-web ok")


if __name__ == "__main__":
    main()
