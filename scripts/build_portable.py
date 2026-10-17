"""Assemble a Windows x64 runtime bundle on any host; never run foreign binaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    subprocess.run([str(a) for a in arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use previously downloaded Windows runtime and wheels")
    parser.add_argument("--include-docs", action="store_true", help="Explicitly include local tutorials/reports; omitted from public builds")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    sdk_version = tomllib.loads((ROOT / "sdk" / "pyproject.toml").read_text())["project"]["version"]
    cache = ROOT / "build" / "portable-downloads"
    wheels = cache / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    runtime_package = cache / "python.3.12.10.nupkg"
    runtime_url = "https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg"
    if not runtime_package.exists():
        if args.offline:
            raise RuntimeError("Windows Python runtime is not cached")
        urllib.request.urlretrieve(runtime_url, runtime_package)
    index_options = ["--no-index", "--find-links", str(wheels)] if args.offline else []
    run(sys.executable, "-m", "pip", "download", "--dest", wheels, "--platform", "win_amd64",
        "--python-version", "312", "--implementation", "cp", "--abi", "cp312", "--only-binary=:all:",
        "-r", ROOT / "requirements-core.lock", *index_options)
    # Build from source copies: stale build caches and sync-conflict files must
    # never become Python modules in the distributed application or SDK wheel.
    source_stage = ROOT / "build" / "portable-source"
    if source_stage.exists():
        shutil.rmtree(source_stage)
    for name, source in (("app", ROOT), ("sdk", ROOT / "sdk")):
        target = source_stage / name
        target.mkdir(parents=True)
        shutil.copy2(source / "pyproject.toml", target)
        shutil.copytree(source / "src", target / "src", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", "*SFConflict*"))
        run(sys.executable, "-m", "build", target, "--no-isolation", "--wheel", "--outdir", ROOT / "build" / "local-wheels")
    destination = ROOT / "dist" / "beamic-pro-windows"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    runtime = destination / "runtime"
    runtime.mkdir()
    with zipfile.ZipFile(runtime_package) as package:
        for item in package.infolist():
            if item.filename.startswith("tools/") and not item.is_dir():
                target = runtime / item.filename.removeprefix("tools/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(item))
    for name in ("python.exe", "pythonw.exe", "Lib/venv/__init__.py", "Lib/ensurepip/__init__.py"):
        if not (runtime / name).exists():
            raise RuntimeError("Runtime does not contain required component: " + name)
    packages = runtime / "Lib" / "site-packages"
    packages.mkdir(parents=True, exist_ok=True)
    local_wheels = ROOT / "build" / "local-wheels"
    sdk_wheels = [p for p in local_wheels.glob(f"beamic_pro_sdk-{sdk_version}-*.whl") if "SFConflict" not in p.name]
    distributions = [p for p in [*wheels.glob("*.whl"), *local_wheels.glob(f"beamic_pro-{version}-*.whl"), *sdk_wheels]
                     if "SFConflict" not in p.name]
    for wheel in distributions:
        with zipfile.ZipFile(wheel) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                name = item.filename
                if ".data/" in name:
                    prefix, content = name.split(".data/", 1)
                    category, relative = content.split("/", 1)
                    if category in ("purelib", "platlib"):
                        target = packages / relative
                    elif category == "scripts":
                        target = runtime / "Scripts" / relative
                    else:
                        target = runtime / relative
                else:
                    target = packages / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    assets = destination / "assets"
    shutil.copytree(ROOT / "plugins", assets / "plugins", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*SFConflict*"))
    (assets / "wheels").mkdir()
    for wheel in sdk_wheels:
        shutil.copy2(wheel, assets / "wheels" / wheel.name)
    (assets / "bridge").mkdir()
    shutil.copy2(ROOT / "bridge" / "device-profile.example.json", assets / "bridge")
    shutil.copytree(ROOT / "bridge" / "bin" / "Release" / "net48", destination / "bridge", ignore=shutil.ignore_patterns("*SFConflict*"))
    for item in (("README.md", "docs") if args.include_docs else ("README.md",)):
        source, target = ROOT / item, destination / item
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("*SFConflict*")) if source.is_dir() else shutil.copy2(source, target)
    bootstrap = '''import os, sys\nfrom pathlib import Path\nroot = Path(__file__).resolve().parent\nos.environ["BEAMIC_PRO_BUNDLED"] = "1"\nos.environ["BEAMIC_PRO_ASSETS"] = str(root / "assets")\nos.environ["BEAMIC_PRO_APP_ROOT"] = str(root)\nos.environ["BEAMIC_PRO_PYTHON"] = str(root / "runtime" / "python.exe")\nos.environ["PYTHONUTF8"] = "1"\n'''
    (destination / "beamic-pro.pyw").write_text(bootstrap + 'from beamic_pro.app import main\nraise SystemExit(main())\n', encoding="utf-8")
    (destination / "preflight.py").write_text(bootstrap + 'from beamic_pro.preflight import main\nraise SystemExit(main())\n', encoding="utf-8")
    launch = '@echo off\r\nset PYTHONUTF8=1\r\nstart "" "%~dp0runtime\\pythonw.exe" "%~dp0beamic-pro.pyw" %*\r\n'
    (destination / "启动 beamic pro.cmd").write_bytes(launch.encode("utf-8"))
    (destination / "启动模拟光路.cmd").write_bytes(launch.replace('%*', '--demo %*').encode("utf-8"))
    (destination / "曝光预检.cmd").write_bytes(b'@echo off\r\nset PYTHONUTF8=1\r\n"%~dp0runtime\\python.exe" "%~dp0preflight.py" %*\r\n')
    (destination / "install.ps1").write_text((ROOT / "installer" / "install_portable.ps1").read_text(encoding="utf-8"), encoding="utf-8-sig")
    (destination / "安装到本机.cmd").write_bytes(b'@echo off\r\npowershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"\r\npause\r\n')
    (destination / "启动诊断.cmd").write_bytes(b'@echo off\r\nset PYTHONUTF8=1\r\n"%~dp0runtime\\python.exe" "%~dp0beamic-pro.pyw" %*\r\npause\r\n')
    metadata = {"version": version, "sdk_version": sdk_version, "platform": "Windows x64", "python": "3.12.10", "runtime_source": runtime_url,
                "built_on": sys.platform, "windows_execution_test": "pending", "hardware_acceptance": "pending",
                "downloads": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [runtime_package, *wheels.glob("*.whl")]}}
    (destination / "build-manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    output = ROOT / "dist" / f"beamic-pro-{version}-windows-x64.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in destination.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(destination.parent))
    print(output)
    print("Windows execution and hardware acceptance are pending; this is a packaged runtime, not a test result.")


if __name__ == "__main__":
    main()

