"""Exercise manual exposure and a connected pair using synthetic numeric frames."""
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QApplication

from beamic_pro.app import Window, configure_application
from beamic_pro.measurement import Capture
from beamic_pro.validation import showcase_cases

ROOT=Path(__file__).resolve().parents[1]


def main():
    app=QApplication([])
    configure_application(app)
    def pump(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            app.processEvents()
            time.sleep(.003)
    with tempfile.TemporaryDirectory(prefix='beamic-manual-bimodal-') as data_dir:
        window=Window(data_dir,demo=True)
        window.show()
        try:
            pump(1.3)
            window.auto_exposure_input.setChecked(False)
            window.source.close()
            value=window.source.set_exposure(750)
            case=next(c for c in showcase_cases() if c.name=='necked_pair')
            identity=window.last_frame.frame_id
            def feed():
                nonlocal identity
                identity+=1
                frame=replace(case.frame,frame_id=identity,received_ns=time.monotonic_ns(),
                              exposure_us=750,exposure_epoch=value['epoch'],
                              coordinate_key=window.last_frame.coordinate_key)
                window.frames.put((window.source_generation,frame))
                pump(.1)
            for _ in range(3):
                feed()
            assert window.last_fit.valid and '连通双峰' in window.last_fit.reason
            assert window.exposure.state.stable
            assert window.source.exposure_us==750
            window.capture=Capture('connected_pair',4,value['epoch'],window.last_frame.coordinate_key,identity)
            for _ in range(4):
                feed()
            assert window.capture is None and '已记录 4 帧' in window.capture_label.text()
            window.source_label.setText('模拟双峰 · 手动曝光 · 合成数值输入')
            window.grab().save(str(ROOT/'docs/figures/ui-manual-connected.png'))
            report={'version':'0.3.0','simulated':True,'manual_exposure_us':750,
                    'auto_exposure':window.exposure.enabled,'connected_bimodal_valid':window.last_fit.valid,
                    'measurement_completed':True,'boundary_fraction':window.last_fit.boundary_fraction,
                    'fit_reason':window.last_fit.reason,'hardware_acceptance':'pending'}
            (ROOT/'docs/manual-bimodal-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))
        finally:
            window.close()
            app.processEvents()


if __name__=='__main__':
    main()
