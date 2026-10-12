import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from derby import config
from derby.data.backup import Loaded
from derby.race import recorder
from derby.race.scoring import Standing
from derby.race.trainer import LapEvent
from derby.server.app import Settings, create_app, finite_or_none
from derby.server.show import Act

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
                        enabled=TWO, load_backup=False, device="cpu", local_only_ops=False, team_token="tok")
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
        assert ws.receive_json() == {"type": "hello", "role": "stadium"}
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
        assert client.app.state.derby.race_running is False
        # a page that connects late, or reconnects, still gets the one-off message of the current act
        with client.websocket_connect("/ws?role=stadium") as late:
            assert [late.receive_json()["type"] for _ in range(4)] == ["hello", "state", "horses", "race_done"]
        with client.websocket_connect("/ws?role=phone") as phone:
            assert [phone.receive_json()["type"] for _ in range(4)] == ["hello", "state", "horses", "race_done"]
        assert client.post("/api/advance").json()["act"] == "SPRINT"
        with client.websocket_connect("/ws?role=team&token=tok") as team:
            got = [team.receive_json()["type"] for _ in range(5)]
            assert got == ["hello", "state", "horses", "race_done", "sprint_request"]
        with client.websocket_connect("/ws?role=phone") as phone:
            assert [phone.receive_json()["type"] for _ in range(4)] == ["hello", "state", "horses", "race_done"]
            client.post("/api/join", json={"device": "dev_0002", "nick": "late"})   # sends one more state
            assert phone.receive_json()["type"] == "state"                          # no sprint_request for phones


def test_load_failure_is_reported_not_silent(tmp_path: Path):
    # no backup is set, so dataset.assemble fails inside the loader thread
    settings = Settings(data_dir=tmp_path, laps=1, batch=16, lap_floor_s=0.0, seed=1,
                        enabled=TWO, load_backup=False, device="cpu", local_only_ops=False)
    with TestClient(create_app(settings)) as c:
        with c.websocket_connect("/ws?role=stadium") as ws:
            assert ws.receive_json()["type"] == "hello"
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


def test_operator_endpoints_refuse_phones_when_local_only(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, laps=1, batch=16, lap_floor_s=0.0, seed=1,
                        enabled=TWO, load_backup=False, device="cpu")          # local_only_ops is on by default
    app = create_app(settings)
    with TestClient(app, client=("10.0.0.5", 1234)) as phone:
        r = phone.post("/api/advance")
        assert r.status_code == 403 and "laptop" in r.json()["error"]
        assert phone.get("/api/team-url").status_code == 403
    with TestClient(app, client=("127.0.0.1", 50000)) as laptop:
        assert laptop.post("/api/advance").status_code == 200


def test_replay_rejects_paths_outside_the_races_folder(client: TestClient, tmp_path: Path):
    assert client.post("/api/replay", json={}).status_code == 400            # nothing recorded yet
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    for body in ({"name": "../../pytest.ini"}, {"name": "C:/Windows/win.ini"},
                 {"name": "../outside.json"}, {"path": str(outside)}):
        assert client.post("/api/replay", json=body).status_code == 400, body
    recorder.write_race(tmp_path / "races" / "r1.json", [], [], [], [], {}, {})
    with client.websocket_connect("/ws?role=stadium") as ws:
        ws.receive_json()
        r = client.post("/api/replay", json={"name": "r1.json"})
        assert r.status_code == 200 and r.json() == {"ok": True, "name": "r1.json"}   # never the full path
        while ws.receive_json()["type"] != "race_done":                      # let the replay thread finish
            pass


def test_team_role_and_sprint_results_need_the_token(client: TestClient):
    r = client.post("/api/sprint-result", json={"horse": "textbook", "ms": 5, "token": "wrong"})
    assert r.status_code == 403
    r = client.post("/api/sprint-result", json={"horse": "textbook", "ms": 5, "token": "tok"})
    assert r.status_code == 400                                              # right token, but no Sprint yet
    assert "team=tok" in client.get("/api/team-url").json()["url"]
    hub = client.app.state.derby.hub
    with client.websocket_connect("/ws?role=team&token=wrong") as ws:
        ws.receive_json()
        assert client.get("/api/state").json()["phones"] == 1               # downgraded to a phone
        assert hub.count("team") == 0
        with client.websocket_connect("/ws?role=team&token=tok") as team:
            team.receive_json()
            assert hub.count("team") == 1


def test_first_message_says_which_role_the_server_accepted(client: TestClient):
    # a team phone left open across a restart keeps an old token; it must be able to tell it was downgraded
    with client.websocket_connect("/ws?role=team&token=wrong") as ws:
        assert ws.receive_json() == {"type": "hello", "role": "phone"}
        assert ws.receive_json()["type"] == "state"
    with client.websocket_connect("/ws?role=team&token=tok") as ws:
        assert ws.receive_json() == {"type": "hello", "role": "team"}
        assert ws.receive_json()["type"] == "state"


def strict_json(text: str):
    """Parse like a browser does: NaN and Infinity are not JSON."""
    def refuse(token):
        raise ValueError(f"{token} is not valid JSON")
    return json.loads(text, parse_constant=refuse)


def test_running_race_blocks_advance_and_replay(client: TestClient, tmp_path: Path):
    derby = client.app.state.derby
    recorder.write_race(tmp_path / "races" / "r1.json", [], [], [], [], {}, {})   # a race replay could start
    derby.show.act, derby.show.data_loaded = Act.DERBY, True
    derby.race_running = True
    r = client.post("/api/advance")
    assert r.status_code == 400 and r.json()["error"] == "The race is still running. Wait for the checkered flag."
    r = client.post("/api/replay", json={})
    assert r.status_code == 400 and r.json()["error"] == "A race is already running."
    assert derby.show.act == Act.DERBY
    derby.race_running = False
    assert client.post("/api/advance").json()["act"] == "SPRINT"


def test_sprint_results_only_while_the_sprint_is_open(client: TestClient):
    derby = client.app.state.derby
    body = {"horse": "textbook", "ms": 5.0, "token": "tok"}
    r = client.post("/api/sprint-result", json=body)                       # the show is still in LOBBY
    assert r.status_code == 400 and r.json()["error"] == "The Sprint is not running."
    derby.show.act = Act.SPRINT                                             # a finished race, set up by hand
    derby.model_paths = {"textbook": "t.onnx", "featherweight": "f.onnx"}
    derby.race_standings = [Standing("textbook", 0.6, 1, 1), Standing("featherweight", 0.5, 1, 2)]
    for bad in (None, "fast", "NaN", "inf", 10 ** 400):
        r = client.post("/api/sprint-result", json={**body, "ms": bad})
        assert r.status_code == 400 and r.json()["error"] == "Bad timing value.", bad
    assert client.post("/api/sprint-result", json=body).json() == {"ok": True, "have": 1}
    assert client.post("/api/sprint-result", json={**body, "horse": "featherweight", "ms": 3.0}).json()["have"] == 2
    assert derby.sprint_done is True                                        # both times are in: decided
    late = client.post("/api/sprint-result", json={**body, "ms": 1.0})
    assert late.status_code == 400 and late.json()["error"] == "The Sprint is already decided."
    r = client.post("/api/sprint/laptop")
    assert r.status_code == 400 and r.json()["error"] == "The Sprint is already decided."
    assert derby.sprint_ms == {"textbook": 5.0, "featherweight": 3.0}     # nothing was overwritten


def test_lap_events_never_send_nan_to_the_browser():
    ev = LapEvent("textbook", 1, 0.5, 0.4, 0.4, None, float("nan"), 12.0, 1.0)
    with pytest.raises(ValueError):
        strict_json(json.dumps(ev.to_dict()))                              # a bare NaN breaks the page
    parsed = strict_json(json.dumps(finite_or_none(ev.to_dict())))
    assert parsed["loss"] is None and parsed["train_acc"] == 0.5 and parsed["test_acc_live"] is None
