"""FastAPI application: hub API + static frontend."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ._version import __version__
from .chapter_server import ChapterServerError, ChapterServerManager
from .config import HubConfig
from .devices import list_gpus, resolve_auto_device
from .model_source import validate_model_entry as check_model_entry
from .proxy import ProxyGateway
from .registry import ManifestError, Registry, scan_registry
from .runner import RunnerBusyError, RunnerError, RunManager
from .schema import ConfigError, dumps_yaml, validate_config
from .sessions import SessionPaths, list_sessions


def create_app(config: HubConfig | None = None) -> FastAPI:
    hub_config = config or _config_from_cwd()
    hub_config.workspace.mkdir(parents=True, exist_ok=True)

    try:
        registry: Registry = scan_registry(hub_config.iter_chapter_dirs())
    except ManifestError as exc:
        raise RuntimeError(f"章节注册失败: {exc}") from exc

    runner = RunManager(hub_config)
    chapter_servers = ChapterServerManager(hub_config)
    proxy_gateway = ProxyGateway()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        runner.shutdown()
        chapter_servers.stop_all()
        await proxy_gateway.aclose()

    app = FastAPI(
        title="TrustLLM Toolkit Hub",
        version=__version__,
        description="《可信大模型》教材工具包集成前端：公共参数、章节调度、YAML 生成与执行",
        lifespan=lifespan,
    )

    # -- helpers ---------------------------------------------------------------
    def _manifest_or_404(toolkit: str):
        if not registry.has(toolkit):
            raise HTTPException(status_code=404, detail=f"未注册的章节工具包: {toolkit}")
        manifest = registry.get(toolkit)
        if not manifest.enabled:
            raise HTTPException(status_code=409, detail=f"章节 {manifest.title} 尚未启用")
        return manifest

    def _session_response(session: str) -> dict[str, Any]:
        try:
            handle = runner.get(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        status = handle.status()
        paths = handle.paths
        status["result"] = paths.read_result()
        return status

    # -- hub info & chapters ------------------------------------------------------
    @app.get("/api/hub-info")
    def hub_info() -> dict[str, Any]:
        return {
            "version": __version__,
            "root": str(hub_config.root),
            "workspace": str(hub_config.workspace),
            "max_concurrent_runs": hub_config.max_concurrent_runs,
            "running": runner.active_count(),
        }

    @app.get("/api/chapters")
    def chapters() -> list[dict[str, Any]]:
        result = []
        for manifest in registry.all():
            server = chapter_servers.status(manifest) if manifest.frontend_server else None
            view = manifest.public_view(
                port=server.get("port") if server else None,
                server_state=server.get("state", "unsupported") if server else "unsupported",
            )
            result.append(view)
        return result

    # -- model validation -----------------------------------------------------------
    @app.get("/api/devices")
    def devices() -> dict[str, Any]:
        """GPU inventory for the device dropdown (empty when nvidia-smi is absent)."""
        return {"gpus": list_gpus(), "auto_device": resolve_auto_device()}

    @app.get("/api/models/local")
    def models_local() -> list[dict[str, Any]]:
        """Model directories found under the configured model_dirs (for the dropdown)."""
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for base_name in hub_config.get("model_dirs") or []:
            base = (hub_config.root / str(base_name)).resolve()
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if not child.is_dir() or str(child) in seen:
                    continue
                if (child / "config.json").is_file() and (child / "tokenizer_config.json").is_file():
                    seen.add(str(child))
                    found.append({"name": child.name, "path": str(child)})
        return found

    @app.post("/api/models/validate")
    def models_validate(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        entry = payload.get("entry")
        toolkit = payload.get("toolkit")
        role = payload.get("role")
        if not isinstance(entry, dict):
            raise HTTPException(status_code=400, detail="缺少 entry 模型配置")
        requires: list[str] = []
        if toolkit and role:
            try:
                manifest = _manifest_or_404(str(toolkit))
            except HTTPException:
                manifest = None
            if manifest is not None:
                spec = manifest.model_role(str(role))
                if spec:
                    accepts = spec.get("accepts") or []
                    if accepts and entry.get("source") not in accepts:
                        raise HTTPException(
                            status_code=400,
                            detail=f"角色 {role} 仅支持模型源 {accepts}，当前为 {entry.get('source')}",
                        )
                    requires = list(spec.get("requires") or [])
        try:
            return check_model_entry(entry, role=str(role or "default"), requires=requires)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- YAML composition & validation ------------------------------------------------
    @app.post("/api/yaml")
    def compose_yaml(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            normalized = validate_config(payload, base=hub_config.root)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not registry.has(normalized["toolkit"]):
            raise HTTPException(status_code=404, detail=f"未注册的章节工具包: {normalized['toolkit']}")
        manifest = registry.get(normalized["toolkit"])
        required_roles = {spec["role"] for spec in manifest.models if spec.get("required", True)}
        missing = required_roles - set(normalized["models"])
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"章节 {manifest.title} 需要模型角色 {sorted(required_roles)}，缺少 {sorted(missing)}",
            )
        return {"config": normalized, "yaml": dumps_yaml(normalized)}

    # -- runs ------------------------------------------------------------------------------
    @app.post("/api/runs")
    def start_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        raw = payload.get("config")
        if raw is None and isinstance(payload.get("yaml"), str):
            try:
                raw = yaml.safe_load(payload["yaml"])
            except yaml.YAMLError as exc:
                raise HTTPException(status_code=400, detail=f"YAML 解析失败: {exc}") from exc
        if raw is None:
            raise HTTPException(status_code=400, detail="缺少 config 或 yaml")
        try:
            normalized = validate_config(raw, base=hub_config.root)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not registry.has(normalized["toolkit"]):
            raise HTTPException(status_code=404, detail=f"未注册的章节工具包: {normalized['toolkit']}")
        manifest = registry.get(normalized["toolkit"])
        if not manifest.enabled:
            raise HTTPException(status_code=409, detail=f"章节 {manifest.title} 尚未启用")
        required_roles = {spec["role"] for spec in manifest.models if spec.get("required", True)}
        missing = required_roles - set(normalized["models"])
        if missing:
            raise HTTPException(status_code=400, detail=f"缺少模型角色: {sorted(missing)}")
        try:
            session = runner.start(manifest, normalized)
        except RunnerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RunnerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"session": session, "status": _session_response(session)}

    @app.get("/api/runs")
    def runs(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return list_sessions(hub_config.workspace, limit=limit)

    @app.get("/api/runs/{session}")
    def run_status(session: str) -> dict[str, Any]:
        return _session_response(session)

    @app.post("/api/runs/{session}/stop")
    def run_stop(session: str) -> dict[str, Any]:
        try:
            return runner.stop(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RunnerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{session}/logs")
    def run_logs(
        session: str,
        offset: int = Query(0, ge=0),
        wait: float = Query(0.0, ge=0.0, le=5.0),
    ) -> Response:
        import time as _time

        try:
            new_offset, text, state = runner.tail_log(session, offset)
            # Long-poll: wait briefly for the log to grow so the UI polls less.
            deadline = _time.monotonic() + wait
            while not text and state == "running" and _time.monotonic() < deadline:
                _time.sleep(0.25)
                new_offset, text, state = runner.tail_log(session, new_offset)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"offset": new_offset, "text": text, "state": state})

    @app.get("/api/runs/{session}/result")
    def run_result(session: str) -> dict[str, Any]:
        response = _session_response(session)
        if response.get("result") is None:
            raise HTTPException(status_code=404, detail="该会话尚无 result.json")
        return response

    @app.get("/api/runs/{session}/artifacts/{name}")
    def run_artifact(session: str, name: str) -> FileResponse:
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="非法的文件名")
        try:
            handle = runner.get(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = handle.paths.dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"产物不存在: {name}")
        return FileResponse(path, filename=name)

    # -- chapter frontend servers --------------------------------------------------------------
    @app.post("/api/chapters/{toolkit}/server/start")
    def chapter_server_start(toolkit: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        manifest = _manifest_or_404(toolkit)
        params = payload if isinstance(payload, dict) else {}
        try:
            return chapter_servers.start(manifest, params=params)
        except ChapterServerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ManifestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/chapters/{toolkit}/server/stop")
    def chapter_server_stop(toolkit: str) -> dict[str, Any]:
        manifest = _manifest_or_404(toolkit)
        stopped = chapter_servers.stop(manifest.toolkit)
        status = chapter_servers.status(manifest)
        status["stopped"] = stopped
        return status

    @app.get("/api/chapters/{toolkit}/server")
    def chapter_server_status(toolkit: str) -> dict[str, Any]:
        manifest = _manifest_or_404(toolkit)
        return chapter_servers.status(manifest)

    # -- same-origin reverse proxy to chapter frontends -----------------------------------
    @app.api_route(
        "/proxy/{toolkit}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy_chapter(toolkit: str, path: str, request: Request) -> Response:
        if not registry.has(toolkit):
            raise HTTPException(status_code=404, detail=f"未注册的章节工具包: {toolkit}")
        manifest = registry.get(toolkit)
        if not manifest.frontend_server:
            raise HTTPException(status_code=404, detail=f"{toolkit} 未声明前端服务")
        status = chapter_servers.status(manifest)
        if status.get("state") != "ready":
            return _proxy_unavailable(path)
        return await proxy_gateway.forward(request, int(status["port"]), path)

    @app.get("/proxy/{toolkit}")
    def proxy_chapter_root(toolkit: str) -> Response:
        # Missing trailing slash would break the frontends' relative paths.
        return RedirectResponse(f"/proxy/{toolkit}/", status_code=307)

    # -- static hub frontend (mounted last) ---------------------------------------------------------
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/", NoCacheStaticFiles(directory=str(static_dir), html=True), name="static")

    return app


class NoCacheStaticFiles(StaticFiles):
    """Serve hub frontend assets with revalidation headers.

    The hub UI iterates quickly; heuristic browser caching of index/chapter.html
    makes stale pages look like "the redesign didn't apply". With
    ``Cache-Control: no-cache`` browsers revalidate on every load, so ordinary
    refreshes pick up new files immediately.
    """

    extensions_no_cache = {".html", ".js", ".css"}

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        media = getattr(response, "media_type", "") or ""
        if any(media.startswith(prefix) for prefix in ("text/html", "text/css", "application/javascript", "text/javascript")):
            response.headers["Cache-Control"] = "no-cache"
        return response


def _config_from_cwd() -> HubConfig:
    from .config import load_config

    return load_config()


def _proxy_unavailable(path: str) -> Response:
    from .proxy import _unavailable_page

    return _unavailable_page(path)


__all__ = ["create_app"]
