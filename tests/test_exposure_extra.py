from conftest import image_frame

from beamic_pro.exposure import ExposureController


def test_dark_search_is_bounded_and_never_marks_dark_frames_stable():
    controller = ExposureController()
    exposure = 1000
    for identity in range(1, 8):
        frame = image_frame(amplitude=0, frame_id=identity, epoch=identity, exposure=exposure)
        request = controller.inspect(frame)
        assert not controller.state.stable
        if request:
            exposure = request
            controller.acknowledge(request, identity + 1)
        else:
            assert "无信号" in controller.state.reason
            break
    else:
        raise AssertionError("无信号曝光搜索未停止")


def test_failed_write_clears_pending_exposure_and_allows_retry():
    controller = ExposureController()
    assert controller.inspect(image_frame(amplitude=1000))
    controller.reject_request("设备写入失败")
    assert controller.pending_us is None
    assert controller.inspect(image_frame(amplitude=1000, frame_id=2))

