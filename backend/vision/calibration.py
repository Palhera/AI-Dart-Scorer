import asyncio
from fastapi import FastAPI


async def _calibrate_cameras(app: FastAPI) -> None:
    await asyncio.sleep(4)
    app.state.ready = True