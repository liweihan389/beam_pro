"""Explicit integration check of a real isolated SDK v2 plugin installation."""
import json
import queue
import sys
import time
import uuid
from pathlib import Path

from beamic_pro.fitting import fit_beam
from beamic_pro.plugin_runtime import PluginProcess
from beamic_pro.plugin_store import PluginStore
from beamic_pro.stream import SharedPublisher
from beamic_pro.validation import clean_case

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = Path('/private/tmp' if sys.platform != 'win32' else __import__('tempfile').gettempdir()) / ('beamic-sdk2-' + uuid.uuid4().hex)
    wheel = ROOT / 'build/local-wheels/beamic_pro_sdk-2.1.0-py3-none-any.whl'
    store = PluginStore(directory, Path(sys.executable), wheel)
    started = time.monotonic()
    manifest = store.install(ROOT / 'dist/plugins/basic_analysis.zip')
    state = store.read()
    events = queue.Queue()
    worker = PluginProcess(store.current_python(), directory / state['plugins'][manifest['id']]['path'], manifest,
                           lambda identity, event: events.put(event))
    publisher = SharedPublisher()
    try:
        if not worker.ready.wait(10):
            raise RuntimeError('Installed worker did not start')
        case = clean_case()
        fit = fit_beam(case.frame)
        descriptor = publisher.publish(case.frame)
        descriptor['frame']['metadata']['quantitative_ready'] = True
        worker.publish(descriptor, fit)
        text = ''
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            event = events.get(timeout=10)
            if event['type'] == 'error':
                raise RuntimeError(event['message'])
            if event['type'] == 'panel' and '13.5%' in event.get('text', ''):
                text = event['text']
                break
        if not text:
            raise RuntimeError('SDK v2 fit was not processed')
        report = {'simulated': True, 'sdk_version': state['sdk_version'], 'installed': True, 'worker_started': True,
                  'geometry_fit_processed': True, 'duration_seconds': time.monotonic() - started, 'panel': text,
                  'main_environment_modified': False, 'hardware_acceptance': 'pending'}
        (ROOT / 'docs/plugin-install-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        worker.close()
        publisher.close()


if __name__ == '__main__':
    main()
