import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from derby import config
from derby.data.backup import Loaded
from derby.server.app import Settings, create_app

# only two horses, so the CPU race in the tests stays fast; missing keys would default to enabled
TWO = {k: k in ("textbook", "featherweight") for k in config.HORSES_ENABLED}


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (30, 60, 90)).save(buf, "JPEG")
    return buf.getvalue()


def fake_backup(n=48):
    x = torch.randint(0, 256, (n, 3, 96, 96), dtype=torch.uint8)
    return Loaded(x, torch.arange(n) % 6, [None] * n)


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, laps=1, batch=16, lap_floor_s=0.0, seed=1,
                        enabled=TWO, load_backup=False, device="cpu")
    app = create_app(settings)
    app.state.derby.backup = fake_backup()
    with TestClient(app) as c:
        yield c


def test_state_and_join(client: TestClient):
    assert client.get("/api/state").json()["act"] == "LOBBY"
    r = client.post("/api/join", json={"device": "dev_0001", "nick": "Ranvir"})
    assert r.status_code == 200 and r.json()["ok"]


def test_photo_only_when_paddock_open_and_caps_apply(client: TestClient):
    files = {"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}
    r = client.post("/api/photo", data={"device": "dev_0001", "class_name": "pen"}, files=files)
    assert r.status_code == 400 and "closed" in r.json()["error"].lower()
    client.post("/api/advance")                       # PADDOCK
    for _ in range(3):
        r = client.post("/api/photo", data={"device": "dev_0001", "class_name": "pen"}, files=files)
        assert r.status_code == 200 and r.json()["thumb"]
    r = client.post("/api/photo", data={"device": "dev_0001", "class_name": "pen"}, files=files)
    assert r.status_code == 400 and "3 photos" in r.json()["error"]
    assert client.get("/api/state").json()["photos"] == 3


def test_bets_only_in_horses_act(client: TestClient):
    r = client.post("/api/bet", json={"device": "dev_0001", "horse": "textbook"})
    assert r.status_code == 400
    for _ in range(3):
        client.post("/api/advance")                   # PADDOCK, MULTIPLY, HORSES
    assert client.get("/api/state").json()["act"] == "HORSES"
    assert client.post("/api/bet", json={"device": "dev_0001", "horse": "textbook"}).status_code == 200
    assert client.post("/api/bet", json={"device": "dev_0001", "horse": "unicorn"}).status_code == 400


def test_full_run_broadcasts_lap_events_and_finishes(client: TestClient):
    with client.websocket_connect("/ws?role=stadium") as ws:
        first = ws.receive_json()
        assert first["type"] == "state"
        for _ in range(3):
            client.post("/api/advance")
            msg = ws.receive_json()
            while msg["type"] not in ("state",):
                msg = ws.receive_json()
        # HORSES: server loads data in the background; wait for data_loaded
        for _ in range(200):
            if client.get("/api/state").json()["data_loaded"]:
                break
            import time; time.sleep(0.05)
        assert client.post("/api/advance").status_code == 200      # DERBY
        seen = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "lap":
                seen.append(msg["event"]["horse"])
            if msg["type"] == "race_done":
                break
        assert seen == ["textbook", "featherweight"]
        assert client.get("/api/state").json()["act"] == "DERBY"
        assert client.get("/models/textbook.onnx").status_code == 200


def test_load_failure_is_reported_not_silent(tmp_path: Path):
    # no backup is set, so dataset.assemble fails inside the loader thread
    settings = Settings(data_dir=tmp_path, laps=1, batch=16, lap_floor_s=0.0, seed=1,
                        enabled=TWO, load_backup=False, device="cpu")
    with TestClient(create_app(settings)) as c:
        with c.websocket_connect("/ws?role=stadium") as ws:
            assert ws.receive_json()["type"] == "state"
            for _ in range(3):
                assert c.post("/api/advance").status_code == 200   # PADDOCK, MULTIPLY, HORSES
            msg = ws.receive_json()
            while msg["type"] != "error":
                msg = ws.receive_json()
            assert msg["where"] == "load"
            assert c.get("/api/state").json()["data_loaded"] is False
            r = c.post("/api/advance")                             # DERBY stays locked
            assert r.status_code == 400 and "loading" in r.json()["error"]
