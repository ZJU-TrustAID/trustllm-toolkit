"""Chapter registry: discover toolkits via their ``chapter.yaml`` manifests.

A chapter toolkit joins the hub by dropping a folder with a ``chapter.yaml``
into the repository — no hub code changes. The manifest declares how the hub
launches the chapter's own frontend server, how the hub executes a run from a
unified YAML, which model roles the chapter consumes and which parameters the
hub UI should render.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_TOOLKIT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_PARAM_TYPES = {"enum", "int", "float", "bool", "text"}
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")
_COMMAND_PLACEHOLDERS = {"conda", "python", "yaml", "output_dir", "port", "config"}


class ManifestError(ValueError):
    """Raised when a chapter.yaml is missing required structure."""


class ChapterManifest:
    def __init__(self, data: dict[str, Any], directory: Path) -> None:
        self.data = data
        self.directory = directory

    # -- identity ---------------------------------------------------------
    @property
    def toolkit(self) -> str:
        return str(self.data["toolkit"])

    @property
    def chapter(self) -> int | str:
        return self.data["chapter"]

    @property
    def title(self) -> str:
        return str(self.data.get("title") or self.toolkit)

    @property
    def brief(self) -> str:
        return str(self.data.get("brief") or "")

    @property
    def enabled(self) -> bool:
        return bool(self.data.get("enabled", True))

    # -- frontend ----------------------------------------------------------
    @property
    def frontend(self) -> dict[str, Any]:
        return self.data.get("frontend") or {}

    @property
    def frontend_mode(self) -> str:
        # "iframe" is the legacy alias of "proxy" (same-origin reverse proxy).
        raw = str(self.frontend.get("mode") or "proxy")
        return "proxy" if raw in {"proxy", "iframe"} else raw

    @property
    def frontend_autostart(self) -> bool:
        return bool(self.frontend.get("autostart", False))

    @property
    def frontend_server(self) -> dict[str, Any] | None:
        return self.frontend.get("server") or None

    @property
    def frontend_entry(self) -> str:
        return str(self.frontend.get("entry") or "/")

    @property
    def frontend_proxy_path(self) -> str:
        entry = self.frontend_entry
        return f"/proxy/{self.toolkit}/{entry.lstrip('/')}" if entry != "/" else f"/proxy/{self.toolkit}/"

    def frontend_port(self, default: int) -> int:
        server = self.frontend_server or {}
        return int(server.get("port", default))

    def frontend_url(self, port: int) -> str:
        if self.frontend_mode == "proxy":
            return self.frontend_proxy_path
        return f"http://127.0.0.1:{port}{self.frontend_entry}"

    # -- backend -----------------------------------------------------------
    @property
    def backend(self) -> dict[str, Any]:
        return self.data.get("backend") or {}

    @property
    def backend_command(self) -> list[str]:
        return [str(part) for part in self.backend.get("command") or []]

    @property
    def backend_timeout(self) -> int:
        return int(self.backend.get("timeout") or 3600)

    @property
    def backend_cwd(self) -> Path:
        return (self.directory / str(self.backend.get("cwd") or ".")).resolve()

    # -- models ------------------------------------------------------------
    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.data.get("models") or [])

    def model_role(self, role: str) -> dict[str, Any] | None:
        for spec in self.models:
            if spec.get("role") == role:
                return spec
        return None

    # -- params / results ---------------------------------------------------
    @property
    def params(self) -> list[dict[str, Any]]:
        return list(self.data.get("params") or [])

    @property
    def results(self) -> dict[str, Any]:
        return self.data.get("results") or {}

    # -- placeholder expansion ------------------------------------------------
    def fill_command(
        self,
        command: list[str],
        *,
        conda_bin: str,
        python_bin: str,
        yaml_path: str | None = None,
        output_dir: str | None = None,
        port: int | None = None,
        config_path: str | None = None,
    ) -> list[str]:
        values: dict[str, str] = {
            "conda": conda_bin,
            "python": python_bin,
            "port": str(port if port is not None else ""),
            "yaml": yaml_path or "",
            "output_dir": output_dir or "",
            "config": config_path or "",
        }
        filled = []
        for part in command:
            def _sub(match: re.Match[str]) -> str:
                key = match.group(1)
                value = values.get(key)
                if value is None or (value == "" and key not in ("conda", "python")):
                    raise ManifestError(f"命令占位符 {{{key}}} 缺少对应值")
                return value
            filled.append(_PLACEHOLDER_PATTERN.sub(_sub, part))
        return filled

    def render_config_template(
        self,
        template: dict[str, Any],
        *,
        port: int,
        port_key: str | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Render the frontend server's config template (nested mapping).

        Placeholders resolve against ``params`` with dotted paths such as
        ``models.guard.entry`` or ``common.device``; ``{port}`` is always
        available and is also injected at ``port_key`` when given.
        """

        def resolve(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, str):
                def _sub(match: re.Match[str]) -> str:
                    key = match.group(1)
                    if key == "port":
                        return str(port)
                    # 空字符串是合法值（可选角色未填时由 chapter_server 显式填 ""）；
                    # 缺 key 或 None 仍然报错，避免静默渲染空模型。
                    if key not in params or params[key] is None:
                        raise ManifestError(f"前端服务配置占符 {{{key}}} 缺少对应值")
                    return str(params[key])
                return _PLACEHOLDER_PATTERN.sub(_sub, value)
            return value

        rendered = resolve(template)
        if port_key:
            node: dict[str, Any] = rendered
            keys = port_key.split(".")
            for key in keys[:-1]:
                child = node.setdefault(key, {})
                if not isinstance(child, dict):
                    raise ManifestError(f"config_port_key 的路径 {port_key} 不是映射")
                node = child
            node[keys[-1]] = port
        return rendered

    def public_view(self, port: int | None = None, server_state: str = "unsupported") -> dict[str, Any]:
        frontend: dict[str, Any] = {"mode": self.frontend_mode, "autostart": self.frontend_autostart}
        if self.frontend_server:
            frontend["port"] = port
            frontend["url"] = self.frontend_url(port) if (port or self.frontend_mode == "proxy") else None
            frontend["state"] = server_state
        return {
            "chapter": self.chapter,
            "title": self.title,
            "toolkit": self.toolkit,
            "enabled": self.enabled,
            "brief": self.brief,
            "models": self.models,
            "params": self.params,
            "results": self.results,
            "frontend": frontend,
            "backend_timeout": self.backend_timeout,
        }


def _validate_param(spec: Any, index: int) -> dict[str, Any]:
    where = f"params[{index}]"
    if not isinstance(spec, dict) or not spec.get("key"):
        raise ManifestError(f"{where} 必须是含 key 的映射")
    ptype = spec.get("type") or "text"
    if ptype not in _PARAM_TYPES:
        raise ManifestError(f"{where}.type 必须是 {sorted(_PARAM_TYPES)} 之一")
    if ptype == "enum" and not spec.get("options"):
        raise ManifestError(f"{where} type=enum 需要 options 列表")
    return spec


def load_manifest(path: Path) -> ChapterManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError(f"{path} 必须是 YAML 映射")
    directory = path.resolve().parent

    for key in ("chapter", "title", "toolkit", "backend", "models"):
        if key not in data:
            raise ManifestError(f"{path} 缺少必填字段: {key}")
    if not isinstance(data["chapter"], (int, str)) or isinstance(data["chapter"], bool):
        raise ManifestError(f"{path}.chapter 必须是章号（整数）或范围（字符串）")
    if not _TOOLKIT_PATTERN.match(str(data["toolkit"])):
        raise ManifestError(f"{path}.toolkit 只能包含小写字母、数字、'-' 和 '_'")
    if not isinstance(data["title"], str) or not data["title"].strip():
        raise ManifestError(f"{path}.title 必须是非空字符串")

    backend = data["backend"]
    if not isinstance(backend, dict) or not backend.get("command"):
        raise ManifestError(f"{path}.backend.command 必须是非空字符串数组")
    for part in backend["command"]:
        if not isinstance(part, str) or not part:
            raise ManifestError(f"{path}.backend.command 必须是非空字符串数组")
        for key in _PLACEHOLDER_PATTERN.findall(part):
            if key not in _COMMAND_PLACEHOLDERS:
                raise ManifestError(f"{path}.backend.command 含未知占位符 {{{key}}}")

    frontend = data.get("frontend") or {}
    if not isinstance(frontend, dict):
        raise ManifestError(f"{path}.frontend 必须是映射")
    if str(frontend.get("mode") or "proxy") not in {"proxy", "iframe", "iframe-direct", "link"}:
        raise ManifestError(f"{path}.frontend.mode 必须是 'proxy'、'iframe-direct' 或 'link'")
    server = frontend.get("server")
    if server is not None:
        if not isinstance(server, dict) or not server.get("command"):
            raise ManifestError(f"{path}.frontend.server.command 必须是非空字符串数组")
        if not isinstance(server.get("port"), int):
            raise ManifestError(f"{path}.frontend.server.port 必须是整数")
        for part in server["command"]:
            for key in _PLACEHOLDER_PATTERN.findall(str(part)):
                if key not in _COMMAND_PLACEHOLDERS:
                    raise ManifestError(f"{path}.frontend.server.command 含未知占位符 {{{key}}}")
        if server.get("config") is not None:
            if not isinstance(server["config"], dict):
                raise ManifestError(f"{path}.frontend.server.config 必须是映射")
            if server.get("config_port_key") is not None and not isinstance(server["config_port_key"], str):
                raise ManifestError(f"{path}.frontend.server.config_port_key 必须是字符串")
        if server.get("env") is not None and not isinstance(server["env"], dict):
            raise ManifestError(f"{path}.frontend.server.env 必须是映射")

    models = data["models"]
    if not isinstance(models, list) or not models:
        raise ManifestError(f"{path}.models 必须是非空数组")
    roles: set[str] = set()
    for index, spec in enumerate(models):
        where = f"models[{index}]"
        if not isinstance(spec, dict) or not spec.get("role"):
            raise ManifestError(f"{where} 必须是含 role 的映射")
        if spec["role"] in roles:
            raise ManifestError(f"{where}.role 重复: {spec['role']}")
        roles.add(spec["role"])
        accepts = spec.get("accepts") or ["local_path", "hf"]
        if not isinstance(accepts, list) or not accepts:
            raise ManifestError(f"{where}.accepts 必须是非空数组")

    for index, spec in enumerate(data.get("params") or []):
        _validate_param(spec, index)

    results = data.get("results") or {}
    if not isinstance(results, dict):
        raise ManifestError(f"{path}.results 必须是映射")

    return ChapterManifest(data=data, directory=directory)


class Registry:
    """All chapter manifests found under the configured scan roots."""

    def __init__(self, manifests: list[ChapterManifest]) -> None:
        self._by_toolkit: dict[str, ChapterManifest] = {}
        for manifest in manifests:
            self._by_toolkit[manifest.toolkit] = manifest

    def get(self, toolkit: str) -> ChapterManifest:
        if toolkit not in self._by_toolkit:
            raise KeyError(f"未注册的章节工具包: {toolkit}")
        return self._by_toolkit[toolkit]

    def has(self, toolkit: str) -> bool:
        return toolkit in self._by_toolkit

    def all(self) -> list[ChapterManifest]:
        return sorted(self._by_toolkit.values(), key=lambda item: str(item.chapter))


def scan_registry(chapter_dirs: list[Path]) -> Registry:
    manifests: list[ChapterManifest] = []
    seen: set[str] = set()
    for directory in chapter_dirs:
        manifest_path = directory / "chapter.yaml"
        if not manifest_path.is_file():
            continue
        manifest = load_manifest(manifest_path)
        if manifest.toolkit in seen:
            raise ManifestError(f"toolkit 重复注册: {manifest.toolkit}（{directory}）")
        seen.add(manifest.toolkit)
        manifests.append(manifest)
    return Registry(manifests)


__all__ = [
    "ChapterManifest",
    "ManifestError",
    "Registry",
    "load_manifest",
    "scan_registry",
]
