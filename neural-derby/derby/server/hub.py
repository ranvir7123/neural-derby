"""Fan-out of JSON messages to every connected page. Roles: stadium, phone, team, control."""
import asyncio
import json

from fastapi import WebSocket


class Hub:
    def __init__(self):
        self.clients: dict[WebSocket, str] = {}

    async def connect(self, ws: WebSocket, role: str) -> None:
        await ws.accept()
        self.clients[ws] = role

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.pop(ws, None)

    def count(self, role: str) -> int:
        return sum(r == role for r in self.clients.values())

    async def broadcast(self, msg: dict, roles: set[str] | None = None) -> None:
        text = json.dumps(msg)
        dead = []
        for ws, role in list(self.clients.items()):
            if roles is not None and role not in roles:
                continue
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_threadsafe(self, loop: asyncio.AbstractEventLoop, msg: dict, roles: set[str] | None = None) -> None:
        asyncio.run_coroutine_threadsafe(self.broadcast(msg, roles), loop)
