import time

from beamic_pro.app import Window


def wait(application, window, condition, seconds=8):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        application.processEvents()
        if condition():
            return
        time.sleep(.004)
    raise AssertionError(window.plugin_states.get('waist', {}).__repr__()[:1800]+window.log.toPlainText())


def test_real_worker_waist_form_five_points_and_supplement(application, tmp_path):
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        wait(application, window, lambda: window.capture_ready and window.exposure.state.stable)
        window.waist_panel.buttons['activate'].click()
        wait(application, window, lambda: 'configure' in window.waist_panel.buttons)
        # Source change must settle before configuration; otherwise the host
        # correctly invalidates the old coordinate system.
        wait(application, window, lambda: window.last_frame.coordinate_key.startswith('sim-waist') and window.exposure.state.stable)
        assert window.waist_panel.controls['tolerance_mm'].value() == 10
        window.waist_panel.buttons['configure'].click()
        wait(application, window, lambda: 'record' in window.waist_panel.buttons)
        for index, z in enumerate((0, 150, 350, 170, 530)):
            if index == 3:
                wait(application, window, lambda: 'continue' in window.waist_panel.buttons)
                window.waist_panel.buttons['continue'].click()
                wait(application, window, lambda: 'record' in window.waist_panel.buttons)
            window.waist_panel.controls['z_mm'].setValue(z)
            window.waist_panel.buttons['record'].click()
            wait(application, window, lambda n=index+1: window.plugin_states['waist']['data']['count'] >= n)
        state = window.plugin_states['waist']['data']
        assert state['stage'] == 'final'
        assert state['outcome'] == '预测一致，需要补测'
        assert abs(state['axes'][0]['z_mm']-160) < 2
        assert abs(state['axes'][1]['diameter_um']-140) < 2
        assert window.waist_panel.plot.isVisible()
        assert window.waist_panel.table.rowCount() == 5
        window.boundary_input.setValue(25)
        wait(application, window, lambda: window.last_fit is not None and window.last_fit.boundary_fraction == .25)
        assert window.plugin_states['waist']['data']['count'] == 5
        window.waist_panel.buttons['add'].click()
        wait(application, window, lambda: 'record' in window.waist_panel.buttons)
        assert '第6点' in window.waist_panel.buttons['record'].text()
        window.waist_panel.controls['z_mm'].setValue(260)
        window.waist_panel.buttons['record'].click()
        wait(application, window, lambda: window.plugin_states['waist']['data']['count'] == 6)
        assert window.plugin_states['waist']['data']['base_count'] == 3
        assert 'held_out_pair' in [p['event'] for p in window.plugin_states['waist']['data']['audit']]
    finally:
        window.close()
        application.processEvents()
