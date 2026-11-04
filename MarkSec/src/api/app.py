from __future__ import annotations

import hmac
import logging
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .engine import Settings, ToolkitEngine

_WORKBENCH_HTML = Path(__file__).resolve().parent / 'static' / 'workbench.html'


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    prompt: str = Field(min_length=1, max_length=4096)
    watermark: Literal['KGW', 'Unigram'] = 'KGW'
    attack: Literal['LLMP'] = 'LLMP'
    max_new_tokens: int = Field(default=128, ge=16, le=256)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=16384)
    watermark: Literal['KGW', 'Unigram'] = 'KGW'


def create_app(engine: ToolkitEngine | None = None):
    engine = engine or ToolkitEngine(Settings.from_env())
    app = FastAPI(title='MarkSec toolkit API', version='0.1.0')
    app.state.engine = engine

    def authorize(authorization: str = Header(default='')):
        if engine.settings.api_token and not hmac.compare_digest(
            authorization, f'Bearer {engine.settings.api_token}',
        ):
            raise HTTPException(401, 'Invalid API token.')

    def execute(operation, **kwargs):
        if not engine.lock.acquire(blocking=False):
            raise HTTPException(429, 'Toolkit is busy. Retry after the current request.',
                                headers={'Retry-After': '5'})
        try:
            return operation(**kwargs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception('MarkSec model execution failed')
            raise HTTPException(503, 'Model execution failed. Check the backend model configuration and logs.') from exc
        finally:
            engine.lock.release()

    @app.get('/health')
    def health():
        return {'status': 'ok', 'mode': 'live', 'model_loaded': engine.model is not None}

    @app.get('/', include_in_schema=False)
    def workbench():
        # Built-in workbench page. It must only use relative paths (v1/...) so the
        # same file works standalone and under the hub's /proxy/{toolkit}/ reverse proxy.
        return FileResponse(_WORKBENCH_HTML, media_type='text/html')

    @app.get('/v1/capabilities', dependencies=[Depends(authorize)])
    def capabilities():
        return engine.capabilities()

    @app.post('/v1/run', dependencies=[Depends(authorize)])
    def run(request: RunRequest):
        return execute(engine.run, **request.model_dump(exclude={'attack'}))

    @app.post('/v1/detect', dependencies=[Depends(authorize)])
    def detect(request: DetectRequest):
        return execute(engine.detect, **request.model_dump())

    return app


app = create_app()
