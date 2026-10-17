from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from . import __version__


class PluginStore:
    """Transactional shared environments, selected by an atomic registry pointer."""
    def __init__(self, root: Path, python: Path, sdk_source: Path, runner=None):
        self.root, self.python, self.sdk_source = Path(root), Path(python), Path(sdk_source)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = self.root / "registry.json"
        self.cancel_event = threading.Event()
        self.runner = runner or self._run

    def _run(self, arguments, progress):
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        lines = []
        output = queue.Queue()
        def read():
            for line in process.stdout:
                output.put(line)
            output.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        deadline = time.monotonic() + 300
        try:
            while True:
                if self.cancel_event.is_set() or time.monotonic() > deadline:
                    raise RuntimeError("安装已取消或操作超时；现有环境保留")
                try:
                    line = output.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                lines.append(line)
                if progress:
                    progress(line.rstrip())
            if process.wait():
                raise RuntimeError("依赖安装失败：\n" + "".join(lines[-18:]))
            return "".join(lines)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            reader.join(timeout=1)
            process.stdout.close()

    def read(self):
        if not self.registry.exists():
            return {"environment": None, "plugins": {}, "lock": []}
        return json.loads(self.registry.read_text(encoding="utf-8"))

    def _commit(self, data):
        temporary = self.root / ("registry-" + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.registry)

    @staticmethod
    def executable(environment: Path):
        return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def current_python(self):
        state = self.read()
        if not state["environment"]:
            raise RuntimeError("尚未安装插件环境")
        return self.executable(self.root / state["environment"])

    @staticmethod
    def validate_manifest(value):
        for key in ("id", "name", "version", "entry", "sdk_version", "dependencies"):
            if key not in value:
                raise ValueError(f"插件清单缺少 {key}")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", value["id"]):
            raise ValueError("插件标识格式无效")
        if value["sdk_version"] != 2:
            raise ValueError("此插件 SDK 版本不受支持；需要声明几何轮廓定义的 SDK v2")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value["entry"]):
            raise ValueError("插件入口格式应为 module:Class")
        if not isinstance(value["dependencies"], list):
            raise ValueError("依赖必须是列表")
        for dependency in value["dependencies"]:
            Requirement(dependency)
        if value.get("kind", "analysis") not in ("analysis", "interactive"):
            raise ValueError("插件类型无效")
        if Version(value.get("minimum_host_version", "0")) > Version(__version__):
            raise ValueError("此插件需要更新版本的 beamic pro")
        return value

    @staticmethod
    def extract(package: Path, destination: Path):
        with zipfile.ZipFile(package) as archive:
            for item in archive.infolist():
                normalized = item.filename.replace("\\", "/")
                target = (destination / normalized).resolve()
                if not target.is_relative_to(destination.resolve()) or normalized.startswith("/") or ":" in normalized:
                    raise ValueError("插件包包含越界路径")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("插件包不能包含符号链接")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)

    def install(self, package: Path, progress=None):
        state = self.read()
        stage = self.root / "packages" / uuid.uuid4().hex
        environment = self.root / "environments" / uuid.uuid4().hex
        stage.mkdir(parents=True)
        environment.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.extract(Path(package), stage)
            manifest = self.validate_manifest(json.loads((stage / "manifest.json").read_text(encoding="utf-8")))
            compatible = {key: item for key, item in state["plugins"].items()
                          if item["manifest"].get("sdk_version") == 2}
            legacy = {**state.get("legacy_plugins", {}), **{key: item for key, item in state["plugins"].items()
                      if item["manifest"].get("sdk_version") != 2}}
            existing = compatible.get(manifest["id"])
            if existing:
                raise ValueError("同标识插件已安装，请先停用并卸载后再导入新版本")
            self.runner([str(self.python), "-m", "venv", str(environment)], progress)
            python = self.executable(environment)
            self.runner([str(python), "-m", "pip", "install", "--disable-pip-version-check", str(self.sdk_source)], progress)
            requirements = ["beamic-pro-sdk==2.1.0", *state["lock"]]
            requirements += manifest["dependencies"]
            # Existing pinned versions are explicit requirements, not silently upgraded.
            if requirements:
                self.runner([str(python), "-m", "pip", "install", "--disable-pip-version-check", *requirements], progress)
            self.runner([str(python), "-m", "pip", "check"], progress)
            plugins = dict(compatible)
            plugins[manifest["id"]] = {"manifest": manifest, "path": str(stage.relative_to(self.root))}
            for item in plugins.values():
                module, symbol = item["manifest"]["entry"].split(":")
                self.runner([str(python), "-c",
                             "import importlib,sys; sys.path.insert(0,sys.argv[1]); "
                             "getattr(importlib.import_module(sys.argv[2]),sys.argv[3])",
                             str(self.root / item["path"]), module, symbol], progress)
            frozen = self.runner([str(python), "-m", "pip", "freeze", "--exclude", "beamic-pro-sdk"], None)
            locks = [line for line in frozen.splitlines() if "==" in line and not line.startswith("#")]
            self._commit({"environment": str(environment.relative_to(self.root)), "plugins": plugins, "lock": locks,
                          "sdk_version": 2, "sdk_package_version": "2.1.0", "legacy_plugins": legacy,
                          "previous_environment": state["environment"]})
            return manifest
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(environment, ignore_errors=True)
            raise

    def uninstall(self, plugin_id):
        state = self.read()
        if plugin_id not in state["plugins"]:
            raise ValueError("插件未安装")
        removed = state["plugins"].pop(plugin_id)
        self._commit(state)
        shutil.rmtree(self.root / removed["path"], ignore_errors=True)
        # Keep the shared environment intact; removing dependencies could break other plugins.

