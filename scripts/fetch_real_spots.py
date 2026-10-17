"""Download the four public experimental sources; keep data outside the app package."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "build/real-spots"


def download(job):
    target = DATA / job["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(3):
        try:
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".part")
                request = Request(job["url"], headers={"User-Agent": "beamic-pro-research/0.4.1"})
                with urlopen(request, timeout=45) as source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                temporary.replace(target)
            data = target.read_bytes()
            if job.get("size") is not None and len(data) != job["size"]:
                raise ValueError("download size mismatch")
            if job.get("sha256") and hashlib.sha256(data).hexdigest() != job["sha256"]:
                raise ValueError("SHA256 mismatch")
            if job.get("md5") and hashlib.md5(data).hexdigest() != job["md5"]:
                raise ValueError("MD5 mismatch")
            if job.get("git_sha"):
                assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == job["git_sha"]
            if target.suffix == ".zip":
                with zipfile.ZipFile(target) as archive:
                    assert archive.testzip() is None
                    destination = target.with_suffix("")
                    for item in archive.infolist():
                        relative = Path(item.filename)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("unsafe archive path")
                    archive.extractall(destination)
            return {**job, "downloaded": True, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        except Exception as exception:
            error = repr(exception)
            if target.exists():
                target.unlink()
            time.sleep(attempt + 1)
    return {**job, "downloaded": False, "error": error}


def discover_jobs():
    DATA.mkdir(parents=True, exist_ok=True)
    metadata_urls = {
        "laserbeamsize-tree.json": "https://api.github.com/repos/scottprahl/laserbeamsize/git/trees/main?recursive=1",
        "oxford-notebook.json": "https://raw.githubusercontent.com/Dipolar-Quantum-Gases/nn-beam-profiling/master/Explore_the_Dataset.ipynb",
        "oxford-readme.txt": "https://ora.ox.ac.uk/objects/uuid:e7f9ee4c-5b07-469f-979c-73b8a28d7ec2/download_file?safe_filename=readme.txt&type_of_work=Dataset",
        "nist.json": "https://data.nist.gov/od/id/mds2-2233",
        "zenodo.json": "https://zenodo.org/api/records/15230870",
    }
    for name, url in metadata_urls.items():
        if not (DATA / name).exists():
            result = download({"source": "metadata", "path": name, "url": url})
            if not result["downloaded"]:
                raise RuntimeError(result)
    tree = json.loads((DATA / "laserbeamsize-tree.json").read_text())
    commit = tree["sha"]
    jobs = []
    for item in tree["tree"]:
        path = item["path"]
        name = Path(path).name
        if path.startswith("docs/images/") and (name.endswith(".pgm") or (name.startswith("k-") and name.endswith(".png") and "crop" not in name)):
            jobs.append({"source": "laserbeamsize", "path": "laserbeamsize/" + name,
                         "url": f"https://raw.githubusercontent.com/scottprahl/laserbeamsize/{commit}/{path}",
                         "git_sha": item["sha"], "size": item["size"]})
    jobs.append({"source": "laserbeamsize", "path": "laserbeamsize/LICENSE.txt",
                 "url": f"https://raw.githubusercontent.com/scottprahl/laserbeamsize/{commit}/LICENSE.txt"})
    jobs.append({"source": "oxford", "path": "oxford/experimental_data.zip",
                 "url": "https://ora.ox.ac.uk/objects/uuid:e7f9ee4c-5b07-469f-979c-73b8a28d7ec2/download_file?safe_filename=experimental_data.zip&type_of_work=Dataset"})
    nist = json.loads((DATA / "nist.json").read_text())
    item = next(x for x in nist["components"] if x.get("filepath") == "Build Metadata.zip")
    jobs.append({"source": "nist", "path": "nist/metadata.zip", "url": item["downloadURL"],
                 "size": item["size"], "sha256": item["checksum"]["hash"]})
    zenodo = json.loads((DATA / "zenodo.json").read_text())
    for item in zenodo["files"]:
        if "yolo" not in item["key"]:
            jobs.append({"source": "rgb", "path": "rgb/" + item["key"], "url": item["links"]["self"],
                         "size": item["size"], "md5": item["checksum"].removeprefix("md5:")})
    return jobs, commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, help="Replay exact URLs and hashes from an existing report manifest")
    arguments = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if arguments.manifest:
        previous = json.loads(arguments.manifest.read_text())
        previous = previous.get("download_manifest", previous)
        commit = previous["laserbeamsize_commit"]
        allowed = {"source", "path", "url", "git_sha", "size", "sha256", "md5"}
        jobs = [{k: v for k, v in item.items() if k in allowed} for item in previous["files"]]
    else:
        jobs, commit = discover_jobs()
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(download, job) for job in jobs]):
            result = future.result()
            results.append(result)
            print(json.dumps({"file": result["path"], "ok": result["downloaded"],
                              "bytes": result.get("bytes"), "error": result.get("error")}), flush=True)
    manifest = {"date": "2026-10-15", "laserbeamsize_commit": commit,
                "files": sorted(results, key=lambda x: x["path"])}
    (DATA / "download-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Downloaded", sum(x["downloaded"] for x in results), "/", len(results), flush=True)


if __name__ == "__main__":
    main()
