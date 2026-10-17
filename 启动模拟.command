#!/bin/zsh
cd -- "$(dirname -- "$0")"
exec .venv/bin/python -m beamic_pro.app --demo --data-dir .local-state

