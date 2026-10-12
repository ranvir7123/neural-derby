"""The one server: routes for phones and the operator, the websocket hub, and the race thread."""
import asyncio
import json
import logging
import math
import secrets
import shutil
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from derby import config
from derby.commentary.caller import Caller
from derby.commentary.verdict import compute_facts, verdicts as make_verdicts
from derby.data import augment, backup as backup_mod, dataset
from derby.data.live import LiveError, LiveStore
from derby.export.bench import bench_cpu
from derby.export.onnx_export import export_onnx
from derby.horses import stable as stable_mod
from derby.horses.stats import horse_stats
from derby.race import recorder, scoring
from derby.race.trainer import LapEvent, Race
from derby.server.hub import Hub
from derby.server.show import Act, GateError, Show

log = logging.getLogger("derby")

# One-off messages that a page connecting late (or reconnecting) still needs, by act, in sending order.
RESEND_BY_ACT = {
    Act.MULTIPLY: ("previews",),
    Act.DERBY: ("race_done",),
    Act.SPRINT: ("race_done", "sprint_request", "sprint"),
    Act.VERDICT: ("sprint", "verdict"),
    Act.SHIP: ("sprint", "verdict", "ship", "veteran"),
}
# The ones that belong to a single race: a new race or a replay forgets them.
RACE_MESSAGES = ("race_done", "sprint_request", "sprint", "verdict", "ship", "veteran")


def finite_or_none(d: dict) -> dict:
    """JSON has no NaN or infinity, and the browser's JSON.parse rejects them, so send null instead."""
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in d.items()}


@dataclass
class Settings:
    data_dir: Path = config.DATA_DIR
    web_dir: Path = config.WEB_DIR
    laps: int = config.LAPS
    batch: int = config.BATCH
    lap_floor_s: float = config.LAP_FLOOR_S
    seed: int = config.SEED
    enabled: dict = field(default_factory=lambda: dict(config.HORSES_ENABLED))
    load_backup: bool = True
    device: str | None = None     # None = cuda if available else cpu
    public_url: str | None = None  # what the QR points at; None = detect LAN ip
    local_only_ops: bool = True    # operator controls answer only the laptop itself
    team_token: str | None = None  # secret in the team phone link; None = make a new one at start


@dataclass
class AppState:
    settings: Settings
    show: Show
    hub: Hub
    live: LiveStore
    stable: list
    stats: dict
    names: dict
    nicks: dict = field(default_factory=dict)
    bets: dict = field(default_factory=dict)
    backup: backup_mod.Loaded | None = None
    data: dataset.RaceData | None = None
    race: Race | None = None
    caller: Caller | None = None
    events: list = field(default_factory=list)
    race_standings: list = field(default_factory=list)
    sprint_ms: dict = field(default_factory=dict)
    sprint_device: str = "phone"
    sprint_standings: list = field(default_factory=list)
    champion: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    model_paths: dict = field(default_factory=dict)
    form_guide: dict = field(default_factory=dict)
    team_token: str = ""
    race_running: bool = False     # a race or replay thread is still sending laps
    sprint_done: bool = False      # the Sprint is decided; late times are refused
    last: dict = field(default_factory=dict)   # newest one-off message of each type, with its roles
    loop: asyncio.AbstractEventLoop | None = None

    @property
    def device(self) -> str:
        if self.settings.device:
            return self.settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; this just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def horse_cards(state: AppState) -> list[dict]:
    cards = []
    for h in state.stable:
        st = state.stats[h.key]
        fg = state.form_guide.get(h.key, {"race_wins": 0, "champion_wins": 0, "races": 0})
        cards.append({"key": h.key, "name": h.name, "tagline": h.tagline, "colour": h.colour, "block": h.block_name,
                      "params": st["params"], "macs": st["macs"], "depth": st["depth"], "table": st["table"],
                      "form": fg})
    return cards


def state_payload(state: AppState) -> dict:
    d = state.show.to_dict()
    d.update({"photos": state.live.count(), "bets": len(state.bets), "phones": state.hub.count("phone"),
              "horses": [h.key for h in state.stable], "n_live_train": state.data.n_live_train if state.data else 0,
              "n_live_test": state.data.n_live_test if state.data else 0,
              "champion": state.champion[0].horse if state.champion else None})
    return d


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    stable = stable_mod.enabled_stable(settings.enabled)
    state = AppState(settings=settings, show=Show(), hub=Hub(),
                     live=LiveStore(settings.data_dir / "live"), stable=stable,
                     stats={h.key: horse_stats(h) for h in stable}, names={h.key: h.name for h in stable})
    fg_path = settings.data_dir / "form_guide.json"
    if fg_path.exists():
        state.form_guide = json.loads(fg_path.read_text(encoding="utf-8"))
    # An empty token would let any phone claim the team role, so treat "" like None.
    state.team_token = settings.team_token or secrets.token_urlsafe(6)

    def operator_allowed(request: Request) -> bool:
        """Operator controls (advance, gates, Sprint on the laptop, replay) only answer the laptop itself."""
        if not settings.local_only_ops or request.client is None:
            return True
        return request.client.host in {"127.0.0.1", "::1", "localhost"}

    def operator_refused() -> JSONResponse:
        return JSONResponse({"error": "Operator controls only work on the laptop."}, status_code=403)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.loop = asyncio.get_running_loop()
        if settings.load_backup:
            state.backup = backup_mod.load(settings.data_dir / "backup")
        yield

    app = FastAPI(title="Neural Derby", lifespan=lifespan)
    app.state.derby = state

    async def push_state():
        await state.hub.broadcast({"type": "state", **state_payload(state)})

    async def send_and_remember(msg: dict, roles: set[str] | None = None) -> None:
        """Send a one-off message and keep it, so a page that connects later still gets it."""
        state.last[msg["type"]] = (msg, roles)
        await state.hub.broadcast(msg, roles)

    def forget_race() -> None:
        """A new race or a replay starts clean: no Sprint times, no decided Sprint, no old race messages."""
        state.sprint_ms.clear()
        state.sprint_done = False
        for kind in RACE_MESSAGES:
            state.last.pop(kind, None)

    # ---------- pages ----------
    if (settings.web_dir / "stadium").is_dir():
        app.mount("/stadium", StaticFiles(directory=settings.web_dir / "stadium", html=True), name="stadium")
        app.mount("/phone", StaticFiles(directory=settings.web_dir / "phone", html=True), name="phone")
        app.mount("/control", StaticFiles(directory=settings.web_dir / "control", html=True), name="control")
        app.mount("/vendor", StaticFiles(directory=settings.web_dir / "vendor"), name="vendor")

    @app.get("/p")
    def short_link():
        return RedirectResponse("/phone/")

    @app.get("/api/qr.png")
    def qr_png():
        import qrcode
        url = settings.public_url or f"http://{lan_ip()}:{config.PORTS['http']}/p"
        img = qrcode.make(url)
        import io
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"X-Join-Url": url})

    # ---------- state and operator ----------
    @app.get("/api/state")
    def get_state():
        return state_payload(state)

    @app.post("/api/advance")
    async def advance(request: Request):
        if not operator_allowed(request):
            return operator_refused()
        if state.show.act == Act.DERBY and state.race_running:
            return JSONResponse({"error": "The race is still running. Wait for the checkered flag."}, status_code=400)
        try:
            act = state.show.advance()
        except GateError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        await push_state()
        if act == Act.MULTIPLY:
            await send_previews()
        elif act == Act.HORSES:
            await state.hub.broadcast({"type": "horses", "cards": horse_cards(state)})
            threading.Thread(target=load_data, daemon=True).start()
        elif act == Act.DERBY:
            start_race()
        elif act == Act.SPRINT:
            models = [{"key": k, "url": f"/models/{k}.onnx"} for k in state.model_paths]
            await send_and_remember({"type": "sprint_request", "models": models}, roles={"team", "control"})
        elif act == Act.VERDICT:
            await finish_verdict()
        elif act == Act.SHIP:
            await ship()
        return {"act": act.value}

    @app.post("/api/gate/close")
    async def close_gate(request: Request):
        if not operator_allowed(request):
            return operator_refused()
        state.show.close_gate()
        await push_state()
        return {"ok": True}

    # ---------- phones ----------
    @app.post("/api/join")
    async def join(body: dict):
        device, nick = str(body.get("device", "")), str(body.get("nick", ""))[:12]
        state.nicks[device] = nick or "anon"
        await push_state()
        return {"ok": True, "nick": state.nicks[device]}

    @app.post("/api/photo")
    async def photo(device: str = Form(...), class_name: str = Form(...), file: UploadFile = File(...)):
        if not state.show.photos_open:
            return JSONResponse({"error": "The paddock is closed right now."}, status_code=400)
        data = await file.read()
        try:
            p = state.live.add(device, class_name, data)
        except LiveError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        thumb = LiveStore.thumbnail_b64(p)
        await state.hub.broadcast({"type": "photo", "class": class_name, "thumb": thumb,
                                   "nick": state.nicks.get(device, "anon"), "count": state.live.count()})
        if state.live.count() >= state.live.caps["total"]:
            state.show.close_gate()
        await push_state()
        return {"ok": True, "thumb": thumb, "count": state.live.count_for(device)}

    @app.post("/api/bet")
    async def bet(body: dict):
        device, horse = str(body.get("device", "")), str(body.get("horse", ""))
        if not state.show.bets_open:
            return JSONResponse({"error": "Bets are closed."}, status_code=400)
        if horse not in state.names:
            return JSONResponse({"error": "That horse is not in this race."}, status_code=400)
        state.bets[device] = horse
        await push_state()
        return {"ok": True, "horse": horse}

    @app.post("/api/sprint-result")
    async def sprint_result(body: dict):
        if str(body.get("token", "")) != state.team_token:
            return JSONResponse({"error": "Only the team phone can send Sprint times."}, status_code=403)
        if state.show.act != Act.SPRINT:
            return JSONResponse({"error": "The Sprint is not running."}, status_code=400)
        if state.sprint_done:
            return JSONResponse({"error": "The Sprint is already decided."}, status_code=400)
        if not state.model_paths or not state.race_standings:
            return JSONResponse({"error": "Run the race first."}, status_code=400)
        try:
            ms = float(body.get("ms", 0))
        except (TypeError, ValueError, OverflowError):   # null, text, or a number too big for a float
            ms = math.nan
        if not math.isfinite(ms):
            return JSONResponse({"error": "Bad timing value."}, status_code=400)
        horse = str(body.get("horse", ""))
        if horse in state.model_paths and ms > 0:
            state.sprint_ms[horse] = ms
            state.sprint_device = str(body.get("device", "phone"))
        if set(state.sprint_ms) >= set(state.model_paths):
            await finish_sprint()
        return {"ok": True, "have": len(state.sprint_ms)}

    @app.post("/api/sprint/laptop")
    async def sprint_laptop(request: Request):
        if not operator_allowed(request):
            return operator_refused()
        if state.sprint_done:
            return JSONResponse({"error": "The Sprint is already decided."}, status_code=400)
        if not state.model_paths or not state.race_standings:
            return JSONResponse({"error": "Run the race first."}, status_code=400)
        for k, p in state.model_paths.items():
            state.sprint_ms[k] = bench_cpu(p)
        state.sprint_device = "laptop"
        await finish_sprint()
        return {"ok": True}

    @app.post("/api/replay")
    async def replay(request: Request, body: dict | None = None):
        """Replay a recorded race. Only a plain file name inside the races folder is accepted,
        never a path, so a request cannot reach any other file on the laptop. No name = the latest race."""
        if not operator_allowed(request):
            return operator_refused()
        if state.race_running:
            return JSONResponse({"error": "A race is already running."}, status_code=400)
        races_dir = settings.data_dir / "races"
        name = Path(str((body or {}).get("name") or "")).name
        path = races_dir / name if name else recorder.latest_race(races_dir)
        if (path is None or path.suffix != ".json" or path.resolve().parent != races_dir.resolve()
                or not path.is_file()):
            return JSONResponse({"error": "No recorded race to replay."}, status_code=400)
        state.events.clear()
        forget_race()
        state.caller = Caller([h.key for h in state.stable], state.names, settings.laps, seed=settings.seed)
        state.race_running = True
        threading.Thread(target=run_events, args=(recorder.replay(path),), daemon=True).start()
        return {"ok": True, "name": path.name}

    @app.get("/api/team-url")
    def team_url(request: Request):
        """The link for the team phone that runs the Sprint. It carries the secret team token."""
        if not operator_allowed(request):
            return operator_refused()
        url = f"https://{lan_ip()}:{config.PORTS['https']}/phone/?team={state.team_token}"
        return {"url": url, "token": state.team_token}

    @app.get("/models/{key}.onnx")
    def model_file(key: str):
        if key == "champion" and state.champion:
            key = state.champion[0].horse
        p = state.model_paths.get(key)
        if not p:
            return JSONResponse({"error": "not ready"}, status_code=404)
        return FileResponse(p, media_type="application/octet-stream", filename=f"{key}.onnx")

    # ---------- websocket ----------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, role: str = "phone", token: str = ""):
        if role == "team" and token != state.team_token:
            role = "phone"   # a wrong or missing token makes it an ordinary phone, quietly
        await state.hub.connect(ws, role)
        try:
            await ws.send_text(json.dumps({"type": "state", **state_payload(state)}))
            if state.show.act in (Act.HORSES, Act.DERBY, Act.SPRINT, Act.VERDICT, Act.SHIP):
                await ws.send_text(json.dumps({"type": "horses", "cards": horse_cards(state)}))
            for kind in RESEND_BY_ACT.get(state.show.act, ()):
                kept = state.last.get(kind)
                if kept is not None and (kept[1] is None or role in kept[1]):
                    await ws.send_text(json.dumps(kept[0]))
            while True:
                await ws.receive_text()   # phones only listen; keep the socket alive
        except WebSocketDisconnect:
            pass
        finally:
            state.hub.disconnect(ws)
            if state.loop:
                await push_state()

    # ---------- helpers that touch the race ----------
    async def send_previews():
        photos = state.live.photos()
        if photos:
            x, _ = LiveStore.tensors(photos[:48])
        elif state.backup is not None:
            x = state.backup.x[:: max(1, len(state.backup.x) // 48)][:48]
        else:
            x = torch.zeros(0, 3, config.IMG, config.IMG, dtype=torch.uint8)
        pairs = augment.preview_pairs(x, n=48, seed=settings.seed) if x.shape[0] else []   # nothing to show yet
        await send_and_remember({"type": "previews", "pairs": pairs, "n_before": state.live.count(),
                                 "n_after": state.live.count() * 10}, roles={"stadium", "control"})

    def load_data():
        # Runs in a worker thread. A failure must reach the pages, or DERBY stays locked with no reason given.
        try:
            train, test = state.live.split_by_device(settings.seed)
            live_tr = LiveStore.tensors(train) if train else None
            live_te = LiveStore.tensors(test) if test else None
            state.data = dataset.assemble(state.backup, live_tr, live_te, settings.seed, device=state.device)
        except Exception as e:
            log.exception("loading the race data failed")
            state.hub.broadcast_threadsafe(state.loop, {"type": "error", "where": "load", "message": str(e)},
                                           roles={"stadium", "control"})   # raw error text is for the operator only
            return
        state.show.data_loaded = True
        state.hub.broadcast_threadsafe(state.loop, {"type": "state", **state_payload(state)})

    def start_race():
        state.events.clear()
        forget_race()
        state.race = Race(state.stable, state.data, laps=settings.laps, batch=settings.batch,
                          lap_floor_s=settings.lap_floor_s, seed=settings.seed, device=state.device)
        state.caller = Caller([h.key for h in state.stable], state.names, settings.laps, seed=settings.seed)
        state.race_running = True
        threading.Thread(target=run_events, args=(state.race.run(), True), daemon=True).start()

    def run_events(event_iter, export_after: bool = False):
        # Runs in a worker thread. A failure (training or export) is reported, and race_done is not sent.
        try:
            try:
                for ev in event_iter:
                    state.events.append(ev)
                    state.race_standings = scoring.race_standings(state.events)
                    lines = state.caller.on_event(ev) if state.caller else []
                    state.hub.broadcast_threadsafe(state.loop, {
                        "type": "lap", "event": finite_or_none(ev.to_dict()), "lines": lines,
                        "standings": [s.to_dict() for s in state.race_standings],
                    })
                if export_after and state.race is not None:
                    models_dir = settings.data_dir / "models"
                    for key, model in state.race.models.items():
                        state.model_paths[key] = export_onnx(model, models_dir / f"{key}.onnx")
            finally:
                state.race_running = False   # cleared before race_done or error goes out, so Space works on arrival
        except Exception as e:
            log.exception("the race thread failed")
            state.hub.broadcast_threadsafe(state.loop, {"type": "error", "where": "race", "message": str(e)},
                                           roles={"stadium", "control"})   # raw error text is for the operator only
            return
        done = {
            "type": "race_done", "standings": [s.to_dict() for s in state.race_standings],
            "models": [{"key": k, "url": f"/models/{k}.onnx"} for k in state.model_paths],
        }
        state.last["race_done"] = (done, None)
        state.hub.broadcast_threadsafe(state.loop, done)

    async def finish_sprint():
        state.sprint_done = True   # first, so a late phone time cannot slip in while this result is being sent
        state.sprint_standings = scoring.sprint_standings(state.sprint_ms, device=state.sprint_device)
        state.champion = scoring.champion_table(state.race_standings, state.sprint_standings)
        bets = scoring.settle_bets(state.bets, state.champion[0].horse, scoring.race_winner(state.race_standings))
        await send_and_remember({
            "type": "sprint", "device": state.sprint_device,
            "sprint": [s.to_dict() for s in state.sprint_standings],
            "champion": [c.to_dict() for c in state.champion],
            "winners": {"champion": [state.nicks.get(d, d) for d in bets.champion_winners],
                        "race": [state.nicks.get(d, d) for d in bets.race_winners]},
            "bet_results": {d: (h == state.champion[0].horse) for d, h in state.bets.items()},
        })

    async def finish_verdict():
        if not state.champion and state.race_standings:
            state.sprint_standings = []
            state.champion = scoring.champion_table(state.race_standings, [])
        state.facts = compute_facts(state.events, state.stats, state.race_standings, state.sprint_standings, state.champion)
        n_train = state.data.n_train if state.data else 0
        state.verdicts = make_verdicts(state.facts, state.names, n_train)
        recorder.write_race(settings.data_dir / "races" / f"{time.strftime('%Y%m%d-%H%M%S')}.json", state.events,
                            state.race_standings, state.sprint_standings, state.champion, state.verdicts,
                            {"seed": settings.seed, "n_train": n_train, "device": state.device})
        await send_and_remember({"type": "verdict", "verdicts": state.verdicts,
                                 "facts": {k: f.to_dict() for k, f in state.facts.items()}})

    async def ship():
        if state.champion:
            src = state.model_paths.get(state.champion[0].horse)
            if src:
                shutil.copy(src, settings.data_dir / "models" / "champion.onnx")
        await send_and_remember({"type": "ship", "champion": state.champion[0].horse if state.champion else None,
                                 "url": "/models/champion.onnx", "classes": config.CLASSES})

    return app
