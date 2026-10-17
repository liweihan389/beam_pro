"""Exercise real Qt controls + plugin process and capture an illustrated tutorial."""

from __future__ import annotations

import base64
import html
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from PySide6.QtWidgets import QApplication, QPushButton

from beamic_pro import __version__
from beamic_pro.app import Window, configure_application

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/level"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    steps = []
    checks = {}

    def state(w):
        return w.plugin_states.get("level", {}).get("data", {})

    def wait(w, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.004)
        raise AssertionError(
            str(w.plugin_states.get("level", {})) + w.log.toPlainText()
        )

    def settle(seconds=0.15):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(0.004)

    def shot(w, title, demo, real, expected, scroll=None):
        if scroll == "top":
            w.level_panel.verticalScrollBar().setValue(0)
        elif scroll == "bottom":
            w.level_panel.verticalScrollBar().setValue(
                w.level_panel.verticalScrollBar().maximum()
            )
        settle()
        name = f"step-{len(steps) + 1:02d}.png"
        assert w.grab().save(str(FIGURES / name))
        steps.append(
            {
                "number": len(steps) + 1,
                "title": title,
                "demo_action": demo,
                "real_action": real,
                "expected": expected,
                "image": "figures/level/" + name,
                "state": state(w),
            }
        )

    def action(w, name):
        wait(w, lambda: name in w.level_panel.buttons)
        w.level_panel.buttons[name].click()

    def record(w, predicate, z=None, progress=False):
        wait(w, lambda: "record" in w.level_panel.buttons)
        if z is not None:
            w.level_panel.controls["z_mm"].setValue(z)
        action(w, "record")
        if progress:
            wait(w, lambda: w.capture is not None and len(w.capture.fits) >= 5)
            shot(
                w,
                "近处几何中心记录中",
                "点击记录后观察有效帧累计。",
                "CCD留在近处，不动镜子，等待20帧记录完成。",
                "旧帧、重复帧和未稳定曝光帧不会进入正式记录。",
            )
        wait(w, predicate)

    def activate(w):
        wait(w, lambda: w.capture_ready and w.exposure.state.stable)
        w.tabs.setCurrentWidget(w.level_panel)
        action(w, "activate")
        wait(
            w,
            lambda: (
                "configure" in w.level_panel.buttons
                and w.last_frame.coordinate_key.startswith("sim-level")
                and w.exposure.state.stable
            ),
        )
        wait(w, lambda: w.level_panel.controls["pixel_pitch_um"].value() == 20)

    def adjust_mirror(w):
        wait(
            w,
            lambda: (
                state(w).get("stage") == "level_guide"
                and w.last_fit
                and w.last_fit.valid
            ),
        )
        target = state(w)["target"]["y_px"]
        delta = target - w.last_fit.center_y
        step = w.source.level_position_mm * 0.00002 * 1000 / 20
        direction = "光斑下移" if delta > 0 else "光斑上移"
        button = next(
            b for b in w.level_demo.findChildren(QPushButton) if b.text() == direction
        )
        for _ in range(abs(round(delta / step))):
            button.click()
        wait(w, lambda: state(w).get("reached"))

    def level(w, illustrate=True, expected_pass=True):
        record(w, lambda: state(w).get("level_near_recorded"), progress=illustrate)
        w.level_position.setValue(state(w)["settings"]["far_mm"])
        settle(0.4)
        if illustrate:
            shot(
                w,
                "移动到远处，保持CCD基准高度",
                "模拟CCD位置改为900 mm。",
                "沿同高导轨移动CCD；保持方向、倍率和滤片不变。",
                "这一阶段只移动CCD，不调反射镜；横向坐标不作为等高约束。",
            )
        record(w, lambda: state(w)["stage"] == "level_guide")
        if illustrate:
            shot(
                w,
                "读取计算得到的水平目标带",
                "观察黄色目标线与容差带。",
                "CCD留在远处，只调单镜竖直指向，让光斑中心进入目标带。",
                "目标通常不同于第一次近处中心；软件已考虑调单镜也会移动近处光斑。",
            )
        adjust_mirror(w)
        if illustrate:
            shot(
                w,
                "远处到达目标，重新记录远处中心",
                "点击“远处已到目标，记录后回近处”。",
                "停止调镜，保持CCD在远处完成新的20帧记录。",
                "连续有效帧达到目标后才出现确认按钮；到达目标不等于最终等高验收。",
            )
        action(w, "confirm_far")
        wait(w, lambda: state(w)["stage"] == "level_return")
        if illustrate:
            shot(
                w,
                "返回近处校验",
                "再次点击记录；模拟器会移回近处。",
                "把CCD返回原近测量面，保持同高；镜子不再调整，记录20帧。",
                "将新的近处中心与调镜后的远处中心比较，不使用调镜前的近处中心验收。",
            )
        record(w, lambda: state(w)["stage"] == "level_done")
        assert state(w)["height_result"]["passed"] == expected_pass
        return state(w)["height_result"]

    with tempfile.TemporaryDirectory(prefix="beamic-level-guide-") as folder:
        w = Window(Path(folder) / "height", demo=True)
        w.resize(1480, 1120)
        w.show()
        try:
            activate(w)
            shot(
                w,
                "开启插件，默认仅选光高",
                "启动模拟光路，打开“等高／准直”，点击启用。",
                "连接并完成相机预检，打开插件。使用固定高度支架，CCD竖轴对应平台法线。",
                "“光高”默认勾选，“准直”默认关闭；距离从反射点量到感光面。",
                scroll="top",
            )
            shot(
                w,
                "核对误差预算与验收门槛",
                "向下滚动查看完整设置。",
                "填写有效像元、倍率和实际支架重复定位公差。SP932U原始像元3.45 μm；此模拟器为20 μm合成像元。",
                "默认近远距离100/900 mm，位置公差±10 mm，高度公差预算±0.010 mm，等高高差门槛0.05 mm；参数均可修改。",
                scroll="bottom",
            )
            action(w, "configure")
            wait(w, lambda: state(w)["stage"] == "level_capture")
            shot(
                w,
                "把CCD放在近处",
                "模拟近处位置100 mm，准备记录。",
                "近处位置、CCD高度和相机朝向确认后，记录几何中心。",
                "仅等高使用几何中心；不要求波长，不调coupler。",
                scroll="top",
            )
            checks["height_only"] = level(w)
            shot(
                w,
                "仅光高模式验收结果",
                "查看近远高差和绝对高差95%上限。",
                "核对按实际预算计算的上限小于0.05 mm；不足时重新测量调整。",
                "这里只承诺所选近远区间等高；不要求达到指定绝对平台光高。",
            )
            action(w, "reset")
            wait(w, lambda: state(w)["stage"] == "setup")
            w.level_panel.controls["height_tol_mm"].setValue(0.1)
            action(w, "configure")
            wait(w, lambda: state(w)["stage"] == "level_capture")
            checks["height_precision_insufficient"] = level(w, False, False)
            shot(
                w,
                "反例：中心接近，摆放精度仍不足",
                "把高度重复定位公差改为±0.100 mm，重新完成近远调整及复测。",
                "按真实支架能力填写预算；若预算大于目标，应改善定位能力或调整门槛。",
                "即使测得中心几乎等高，计入误差后95%上限仍超过0.05 mm，软件拒绝通过。",
            )
        finally:
            w.close()
            app.processEvents()

        w = Window(Path(folder) / "combined", demo=True)
        w.resize(1480, 1120)
        w.show()
        try:
            activate(w)
            w.level_panel.controls["collimate_enabled"].setChecked(True)
            w.level_panel.controls["coupler_confirmed"].setChecked(True)
            shot(
                w,
                "同时选择准直与光高",
                "勾选准直并确认coupler有独立调焦功能。",
                "平面镜负责指向，独立调焦机构负责准直。填写波长；本例780 nm。",
                "两项都选时，必须先完成准直，再以新的中心记录调等高。",
                scroll="top",
            )
            action(w, "configure")
            wait(w, lambda: state(w)["stage"] == "coarse_capture")
            shot(
                w,
                "准直近处粗测",
                "记录相对近处0 mm的20帧二阶矩直径。",
                "把CCD放在近处并等待曝光稳定。",
                "直径来自真实数值帧的D4σ，不取显示圈的大小。",
                scroll="top",
            )
            record(w, lambda: state(w)["coarse_count"] == 1)
            shot(
                w,
                "准直远处粗测",
                "下一点为相对近处800 mm，即距反射点900 mm。",
                "移动CCD到远处再记录20帧。",
                "近远粗测只提供尺寸趋势；两端一样大也不能直接判准直。",
            )
            record(w, lambda: state(w)["stage"] == "focus")
            shot(
                w,
                "手动小步调coupler",
                "观察近远粗测结果及当前远处D4σ反馈。",
                "小步调轴向调焦机构；不要只追求远处最小光斑。每次调过需点清空本轮。",
                "不提供未校准的旋钮方向或圈数；调焦可能同时改变指向。",
            )
            w.level_focus.setValue(0)
            wait(w, lambda: state(w)["generation"] == 1)
            wait(w, lambda: state(w).get("live_width_um") is not None)
            shot(
                w,
                "调焦后旧数据全部失效",
                "将模拟调焦量改为0，旧记录会自动清空。",
                "实机调整后点击“已调coupler：清空本轮并重新确认”。",
                "旧宽度、中心、预测和等高目标都不能用于新调焦状态的验收。",
            )
            action(w, "start_validation")
            wait(w, lambda: state(w)["stage"] == "collimation_capture")
            for i, name in enumerate(["近端", "远端", "中间"], 1):
                shot(
                    w,
                    f"五点确认第{i}点：{name}",
                    f"确认相对近处 {state(w)['pending_z']:g} mm，点击记录。",
                    "填写实际位置，保持调焦量固定；记录20个稳定、未饱和的有效帧。",
                    "默认先近端0，再远端800，第三点中间400 mm；各点须充分分离。",
                )
                record(w, lambda i=i: state(w)["count"] == i)
            shot(
                w,
                "前三点冻结，尚未验证",
                "观察传播曲线，然后点击冻结预测并继续。",
                "前三点建立预测；暂不改变调焦量和光路，准备两个新位置。",
                "检查区间端点及内部极值，避免漏掉中间聚焦；不要求精确锁定束腰。",
                scroll="bottom",
            )
            action(w, "continue")
            for i in [4, 5]:
                wait(w, lambda: "record" in w.level_panel.buttons)
                shot(
                    w,
                    f"独立验证第{i}点",
                    f"默认相对近处 {state(w)['pending_z']:g} mm，点击记录。",
                    "在此前未测过的两个位置分别记录；不能重复近端或远端来代替空间验证。",
                    "预测保持冻结，两个新点未参与前三点计算；X、Y两个方向都要一致。",
                    scroll="top",
                )
                record(w, lambda i=i: state(w)["count"] == i)
            assert state(w)["collimation_passed"]
            checks["combined_collimation"] = state(w)["collimation_result"]
            checks["combined_validation"] = state(w)["joint_validation"]
            shot(
                w,
                "五点验证与区间准直结果",
                "查看五点表、X/Y变化、联合95%上限及验证状态。",
                "按实际工作距离和误差预算验收；不达标时继续调焦或补测。",
                "默认X/Y在整个区间的直径变化联合95%上限须≤5%；这是区间准直，不是ISO M²认证。",
                scroll="bottom",
            )
            action(w, "start_level")
            wait(w, lambda: state(w)["stage"] == "level_capture")
            shot(
                w,
                "准直通过后开始单镜等高",
                "点击“准直通过，开始单镜等高”。",
                "保持coupler调焦量不变，重新测量当前状态的近远几何中心。",
                "调镜只改变竖直指向；若再次调coupler，须重新确认准直及等高。",
                scroll="top",
            )
            checks["combined_height"] = level(w, False)
            shot(
                w,
                "组合模式完成",
                "查看等高通过以及此前的准直结果。",
                "最后一次近处复测时停止调镜，确认两项均通过。",
                "模拟结果不能替代实验室CCD、支架与coupler的实机验收。",
            )
            # Exercise collimation-only configuration, never reusing combined records.
            action(w, "reset")
            wait(w, lambda: state(w)["stage"] == "setup")
            w.level_panel.controls["level_enabled"].setChecked(False)
            shot(
                w,
                "仅准直模式的选择",
                "取消光高，保留准直。",
                "不需要等高时，仅完成准直流程；软件不会要求调单镜或达到水平目标线。",
                "三种模式均独立支持；禁用两项会被拒绝。",
                scroll="top",
            )
            action(w, "configure")
            wait(w, lambda: state(w)["stage"] == "coarse_capture")
            assert not state(w)["settings"]["level_enabled"]
            checks["collimation_only_configured"] = True
            w.level_focus.setValue(0.1)
            wait(w, lambda: state(w)["stage"] == "focus")
            shot(
                w,
                "再次调焦：此前结果立即失效",
                "改变模拟调焦量，观察计数归零。",
                "任何调焦后都要重新验证，不能用调焦前的近处测量配调焦后的远处测量。",
                "五点计数、准直通过状态和等高结果均已清空。",
            )
            assert (
                state(w)["count"] == 0
                and state(w)["height_result"] is None
                and not state(w)["collimation_passed"]
            )
            checks["focus_change_invalidates_all"] = True
        finally:
            w.close()
            app.processEvents()
    report = {
        "date": "2026-10-15",
        "version": __version__,
        "simulated": True,
        "hardware_acceptance": "pending",
        "checks": checks,
        "step_count": len(steps),
        "steps": steps,
    }
    (ROOT / "docs/level-workflow-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_guide(report)
    print(
        json.dumps(
            {"steps": len(steps), "checks": checks}, ensure_ascii=False, indent=2
        )
    )


def write_guide(report):
    intro = (
        "本教程使用真实程序窗口、真实插件独立进程和内存合成光路，逐步点击并截图。"
        "所有图像均为模拟数据，不连接相机，不是实机验收。默认仅光高；准直可选，或同时开启。"
    )
    conditions = (
        "距离从单镜有效反射点沿传播路径量到CCD感光面，默认100/900 mm；近远之间无透镜或有光焦度元件。"
        "CCD使用同一固定高度支架或导轨，竖轴对应平台法线，方向、倍率与采集设置不变。"
        "真实SP932U原始像元3.45 μm；本教程合成像元20 μm，勿照抄到实机。"
        "SP932U感光面位于本体前端参考面后4.5 mm，不能以滤片最前缘量距。"
        "默认位置公差±10 mm、支架高度重复定位公差±0.010 mm、共同尺度公差±2%都是可修改误差预算，须按实际能力核对。"
    )
    page = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单镜等高与准直：一步一图</title>',
        '<style>body{max-width:1380px;margin:30px auto;padding:0 24px;background:#f2f6fa;color:#17334b;font:16px/1.8 -apple-system,"Microsoft YaHei",sans-serif}section{padding:24px;background:white;margin:24px 0;border-radius:12px}img{width:100%;height:auto;border:1px solid #d5dfe8}h1,h2{line-height:1.4}.note{background:#fff0cf;padding:16px}nav a{display:block}</style>',
        '<h1>单镜等高与准直：一步一图</h1><p class="note">'
        + intro
        + "</p><p>"
        + conditions
        + "</p><nav>",
    ]
    page += [
        f'<a href="#s{s["number"]}">{s["number"]}. {html.escape(s["title"])}</a>'
        for s in report["steps"]
    ]
    page.append("</nav>")
    md = ["# 单镜等高与准直：一步一图", intro, conditions]
    for s in report["steps"]:
        title = f"第{s['number']}步：{s['title']}"
        image = ROOT / "docs" / s["image"]
        encoded = base64.b64encode(image.read_bytes()).decode()
        page.append(f'<section id="s{s["number"]}"><h2>{html.escape(title)}</h2>')
        md.append("## " + title)
        for key, label in [
            ("demo_action", "模拟操作"),
            ("real_action", "实机操作"),
            ("expected", "检查要点"),
        ]:
            page.append("<p><b>" + label + "：</b>" + html.escape(s[key]) + "</p>")
            md.append(label + "：" + s[key])
        page.append(
            f'<img alt="{html.escape(title)}，模拟数据截图" src="data:image/png;base64,{encoded}"></section>'
        )
        md.append(f"![{title}]({s['image']})")
    notes = (
        "等高使用所选几何中心，目标是一条水平线，不约束横向位置。远处重新记录后再回近处验收，绝对高差的单侧95%上限须≤0.05 mm。"
        "准直使用独立D4σ测量，先近远粗查，再重新取近、远、中三个建模点，冻结后取两个新位置验证；只有两轴联合变化95%上限≤5%、模型与独立验证均一致才通过。"
        "近平坦传播可以通过区间检查，即使束腰位置无法确定；不把负曲率误差抽样删掉来制造窄区间。"
        "软件不能自动发现真实支架高度变化或真实coupler被调过，用户须保持基准并点击清空旧状态。图像和测量只在内存中使用；此教程为主动生成的模拟测试截图。"
    )
    page.append(
        "<section><h2>验收与限制</h2><p>"
        + notes
        + "</p><p>完整状态记录：level-workflow-check.json；数值检查：level-numerical-check.json。</p></section></html>"
    )
    md += ["## 验收与限制", notes]
    (ROOT / "docs/单镜等高准直操作说明.html").write_text(
        "".join(page), encoding="utf-8"
    )
    (ROOT / "docs/单镜等高准直操作说明.md").write_text(
        "\n\n".join(md), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
