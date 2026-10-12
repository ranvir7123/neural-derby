"""Start the show: HTTP on 8000 for the room, HTTPS on 8443 for the team phone, one app, one event loop."""
import asyncio

import uvicorn

from derby import config
from derby.server.app import create_app


async def serve():
    app = create_app()
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.PORTS["http"], log_level="info"))]
    cert, key = config.DATA_DIR / "cert.pem", config.DATA_DIR / "key.pem"
    if cert.exists() and key.exists():
        servers.append(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.PORTS["https"], log_level="info",
                                                     lifespan="off", ssl_certfile=str(cert), ssl_keyfile=str(key))))
    else:
        print("no data/cert.pem: HTTPS listener off (team phone live video unavailable). Run tools/make_cert.py.")
    await asyncio.gather(*(s.serve() for s in servers))


def run():
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
