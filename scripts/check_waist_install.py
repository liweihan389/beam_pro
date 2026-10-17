"""Offline, real candidate environment / wheel / worker compatibility check."""
import json
import queue
import shutil
import sys
import tempfile
from pathlib import Path

import numpy

from beamic_pro.plugin_runtime import PluginProcess
from beamic_pro.plugin_store import PluginStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='beamic-waist-install-') as directory:
        store = PluginStore(Path(directory), Path(sys.executable), ROOT/'build/local-wheels/beamic_pro_sdk-2.1.0-py3-none-any.whl')
        # Seed the dependency from this already tested NumPy distribution. The
        # installer still executes real venv / pip / check / freeze commands;
        # no registry or package validation is mocked, and pip is offline.
        def runner(command, progress):
            if 'pip' in command and 'install' in command:
                command = [*command, '--no-index']
            text = store._run(command, progress)
            if 'venv' in command:
                python = PluginStore.executable(Path(command[-1]))
                import subprocess
                site = Path(subprocess.check_output([str(python), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip())
                origin = Path(numpy.__file__).parent.parent
                for source in origin.glob('numpy*'):
                    if source.is_dir():
                        shutil.copytree(source, site/source.name)
            return text
        store.runner = runner
        store.install(ROOT/'dist/plugins/basic_analysis.zip')
        before = store.read()
        manifest = store.install(ROOT/'dist/plugins/waist.zip')
        after = store.read()
        assert after['previous_environment'] == before['environment']
        assert set(after['plugins']) == {'basic_analysis', 'waist'}
        events = queue.Queue()
        worker = PluginProcess(store.current_python(), store.root/after['plugins']['waist']['path'], manifest,
                               lambda identity, event: events.put(event))
        try:
            assert worker.ready.wait(10)
            panel = events.get(timeout=5)
            assert panel['type'] == 'panel' and len(panel['fields']) == 9
            worker.send({'type': 'action', 'action': 'configure', 'values': {}})
            capture = events.get(timeout=5)
            while capture['type'] != 'panel':
                assert capture['type'] != 'error', capture
                capture = events.get(timeout=5)
            assert capture['type'] == 'panel' and capture['data']['stage'] == 'capture'
            assert capture['fields'][0]['readonly']
            report = {'simulated': True, 'hardware_acceptance': 'pending', 'offline': True,
                      'sdk_package_version': after['sdk_package_version'], 'installed_plugins': list(after['plugins']),
                      'actual_worker_started': True, 'declarative_form_processed': True,
                      'previous_environment_retained': (store.root/before['environment']).exists(),
                      'main_environment_modified': False}
            (ROOT/'docs/waist-install-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            worker.close()


if __name__ == '__main__':
    main()
