"""Package source, tests, documentation and the compiled bridge, excluding local state."""
import argparse
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-docs', action='store_true', help='Explicitly include local tutorials/reports; omitted from public builds')
    args = parser.parse_args()
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    output = ROOT / 'dist' / f'beamic-pro-{version}-source.zip'
    excluded = {'.venv', '.tools', 'build', 'dist', '__pycache__', '.pytest_cache', '.ruff_cache',
                '.local-state', '.build-venv', '.git', 'obj', 'tmp', 'bin'}
    if not args.include_docs:
        excluded.update({'docs', 'output'})
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in ROOT.rglob('*'):
            relative = path.relative_to(ROOT)
            if (path.is_file() and not path.is_symlink()
                    and not any(part in excluded or part.endswith('.egg-info') or 'SFConflict' in part for part in relative.parts)
                    and path.name not in {'.DS_Store', '.coverage'} and path.suffix != '.pyc'):
                archive.write(path, Path('beamic-pro') / relative)
    print(output)


if __name__ == '__main__':
    main()
