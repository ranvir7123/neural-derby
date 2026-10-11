# Setup (from clone to a running show)

1. Install Python 3.12 (the course version).
2. From `neural-derby/`:
   - `py -3.12 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
   - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   - `pip install -r requirements.txt`
   - `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.
3. One-time downloads (internet needed, done before the demo):
   - `python tools/vendor_assets.py` (font and onnxruntime-web into `web/vendor/`)
   - `python tools/build_backup_dataset.py` (about 0.5 GB of photos into `data/backup/`)
4. Tests: `pytest`.
5. Run the show: `python -m derby.server.main`. Open `http://localhost:8000/stadium/` on the projector, `http://localhost:8000/control/` on a second window, and the QR on the stadium page on phones.
