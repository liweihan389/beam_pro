import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/plugins"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in (root / "plugins").iterdir():
        if not (folder / "manifest.json").exists():
            continue
        destination = args.output / (folder.name + ".zip")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in folder.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and not any('SFConflict' in part for part in path.parts):
                    archive.write(path, path.relative_to(folder))
        print(destination)


if __name__ == "__main__":
    main()

