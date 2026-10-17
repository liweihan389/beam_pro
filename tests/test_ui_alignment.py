import time

from PySide6.QtWidgets import QApplication

from beamic_pro.app import Window


def until(application, predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("操作流程未在期限内完成")


def test_entire_two_mirror_workflow_and_two_plane_verification(tmp_path):
    application = QApplication.instance() or QApplication([])
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        assert "DM固定" in window.alignment_notice.text()
        assert "先M2，再M1" in window.alignment_notice.text()
        assert [window.demo_mirror.itemText(i) for i in range(window.demo_mirror.count())] == ["调M2近反射镜", "调M1远反射镜"]
        until(application, lambda: window.last_fit and window.last_fit.valid and window.exposure.state.stable)
        window.start_alignment()
        until(application, lambda: "先遮蓝光" in window.guidance_text.text())
        assert "DM和红光光路保持固定" in window.guidance_text.text()
        def record_four():
            for index in range(4):
                if index == 1:
                    # A software fitting ROI must not destroy the already recorded red reference.
                    window.view.selection = (65, 45, 245, 200)
                    window.select_roi(window.view.selection)
                window.demo_plane.setCurrentIndex(index // 2)
                window.demo_color.setCurrentIndex(index % 2)
                deadline = time.monotonic() + .3
                while time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.005)
                until(application, lambda: window.exposure.state.stable and window.last_fit and window.last_fit.valid)
                window.send_action("record")
                until(application, lambda: window.capture is not None)
                until(application, lambda: window.capture is None, timeout=8)
                deadline = time.monotonic() + .2
                while time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.005)
        record_four()
        window.clear_selection()
        until(application, lambda: "只调M2近反射镜" in window.guidance_text.text() or "M2近反射镜目标" in window.guidance_text.text())
        source, geometry = window.source, window.source.geometry
        source.near_kick = (source.offset - geometry.mirror_far * source.slope) / (geometry.mirror_far - geometry.mirror_near)
        until(application, lambda: "M2近反射镜目标已到达" in window.guidance_text.text())
        assert "DM固定" in window.guidance_title.text()
        window.send_action("confirm_near")
        until(application, lambda: "只调M1远反射镜" in window.guidance_text.text() or "M1远反射镜目标" in window.guidance_text.text())
        source.far_kick = (-source.offset + geometry.mirror_near * source.slope) / (geometry.mirror_far - geometry.mirror_near)
        until(application, lambda: "M1远反射镜目标已到达" in window.guidance_text.text())
        window.send_action("confirm_far")
        until(application, lambda: "移回近处" in window.guidance_text.text())
        record_four()
        until(application, lambda: "合束通过" in window.guidance_text.text())
        assert "DM和红光光路保持固定" in window.guidance_text.text()
        assert "近处误差" in window.guidance_text.text()
        assert "远处误差" in window.guidance_text.text()
        assert not list(tmp_path.rglob("*.png")) and not list(tmp_path.rglob("*.npy"))
    finally:
        window.close()
        application.processEvents()

