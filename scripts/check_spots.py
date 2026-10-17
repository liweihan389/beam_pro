"""Generate reproducible numerical checks and illustrated reports from synthetic data."""
from __future__ import annotations

import base64
import html
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/beamic-mpl-cache" if os.name != "nt" else str(Path.home() / ".beamic-mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from beamic_pro.fitting import FitSettings
from beamic_pro.validation import clean_case, evaluate, showcase_cases

ROOT = Path(__file__).resolve().parents[1]


def render_paragraph(paragraph):
    value = html.escape(paragraph)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"`(.+?)`", r"<code>\1</code>", value)
    if value.startswith("## "):
        return "<h2>" + value[3:] + "</h2>"
    if all(line.startswith("- ") for line in value.splitlines()):
        return "<ul>" + "".join("<li>" + line[2:] + "</li>" for line in value.splitlines()) + "</ul>"
    return "<p>" + value.replace("\n", "<br>") + "</p>"


def main():
    fonts = [Path("/System/Library/Fonts/PingFang.ttc"), Path("/System/Library/Fonts/STHeiti Light.ttc"),
             Path("/System/Library/Fonts/Supplemental/Songti.ttc"), Path("C:/Windows/Fonts/msyh.ttc")]
    for font in fonts:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    destination = ROOT / "docs"
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    cases, rows, fits = showcase_cases(), [], {}
    for case in cases:
        for mode in ("boundary", "area"):
            fit, result = evaluate(case, FitSettings(center_mode=mode))
            rows.append(result)
            if mode == "boundary":
                fits[case.name] = fit
    rng = np.random.default_rng(20261014)
    randomized = []
    for i in range(24):
        for kind in ("round", "ellipse", "airy"):
            major, minor = rng.uniform(24, 48), rng.uniform(14, 23)
            radii = (major, minor) if kind == "ellipse" else (major, major)
            case = clean_case("airy" if kind == "airy" else "gaussian", seed=100 + i,
                              center=tuple(rng.uniform([100, 80], [215, 155])), radii=radii,
                              angle=rng.uniform(-1.5, 1.5), amplitude=rng.uniform(1700, 3400),
                              noise=rng.uniform(2, 18), gradient=rng.uniform(0, 90))
            for mode in ("boundary", "area"):
                _, row = evaluate(case, FitSettings(center_mode=mode))
                row["kind"] = kind
                randomized.append(row)
    thresholds = []
    for kind in ("gaussian", "airy"):
        for fraction in (.03, .05, .135, .3, .5, .8):
            case = clean_case(kind, radii=(42, 30) if kind == "gaussian" else (42, 42), fraction=fraction, noise=2)
            for mode in ("boundary", "area"):
                _, row = evaluate(case, FitSettings(center_mode=mode, boundary_fraction=fraction))
                thresholds.append(row)
    all_rows = rows + randomized + thresholds
    numerical = randomized + thresholds + [r for r in rows if r["axes_relative_errors"] is not None]
    maximum_center = max(r["center_error_px"] for r in numerical)
    maximum_axis = max(max(r["axes_relative_errors"]) for r in numerical)
    summary = {"date": "2026-10-14", "version": "0.3.0", "simulated": True,
               "hardware_acceptance": "pending", "record_count": len(all_rows),
               "passed_count": sum(r["passed"] for r in all_rows),
               "max_clean_center_error_px": maximum_center,
               "max_clean_axis_relative_error": maximum_axis,
               "showcase": rows, "randomized": randomized, "thresholds": thresholds}
    (destination / "spot-check-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pictures = []
    for index, start in enumerate(range(0, len(cases), 4)):
        batch = cases[start:start + 4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
        fig.suptitle("光斑定位检查 · 13.5% 边界 · 合成图像", fontsize=17)
        for ax, case in zip(axes.ravel(), batch, strict=False):
            fit = fits[case.name]
            y, x = np.mgrid[:case.frame.shape[0], :case.frame.shape[1]]
            bg = fit.background_origin if fit.background_origin is not None else 30.
            bg = bg + fit.background_slope_x * x + fit.background_slope_y * y
            intensity = np.clip((case.frame.pixels - bg) / max(fit.amplitude, 1), 0, 1)
            t = .135
            levels = [0., t*.3, t*.8, t, t+.18*(1-t), t+.45*(1-t), t+.75*(1-t), 1.]
            colors = np.array([[8,10,32],[55,15,95],[170,30,145],[25,65,215],
                               [20,190,220],[90,210,70],[255,195,25],[255,255,255]]) / 255
            cmap = LinearSegmentedColormap.from_list("beamic", list(zip(levels, colors, strict=True)))
            ax.imshow(intensity, origin="upper", cmap=cmap, vmin=0, vmax=1)
            if fit.radius_minor > 0:
                ax.add_patch(Ellipse(fit.boundary_center, 2*fit.radius_major, 2*fit.radius_minor,
                                    angle=math.degrees(fit.angle_rad), fill=False,
                                    color="#63ffd3" if fit.valid else "#ffbe59", linewidth=1.7,
                                    linestyle="-" if fit.valid else "--"))
                ax.plot(fit.area_center_x, fit.area_center_y, "x", color="#ffbe59", markersize=8)
            if case.center is not None:
                ax.plot(*case.center, "+", color="white", markersize=10)
            if case.name == "clipped":
                ax.set_xlim(0, 100)
                ax.set_ylim(175, 65)
            else:
                ax.set_xlim(80, 230)
                ax.set_ylim(190, 40)
            row = next(r for r in rows if r["name"] == case.name and r["mode"] == "boundary")
            status = "可测量" if fit.valid else "暂停记录"
            ax.set_title(f"{case.label} · {status}\n{fit.reason}", fontsize=11)
            ax.set_xlabel("x / 像素")
            ax.set_ylabel("y / 像素")
        for ax in axes.ravel()[len(batch):]:
            ax.set_visible(False)
        path = figures / f"spot-check-{index + 1}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        pictures.append(path)
    table = ["| 示例 | 边界中心相对完整中心的偏移 / px | 面积中心相对完整中心的偏移 / px | 结果 | 提示 |",
             "|---|---:|---:|---|---|"]
    html_table = []
    for case in cases:
        a = next(r for r in rows if r["name"] == case.name and r["mode"] == "boundary")
        b = next(r for r in rows if r["name"] == case.name and r["mode"] == "area")
        ea = f"{a['center_error_px']:.4f}" if a["center_error_px"] is not None else "—"
        eb = f"{b['center_error_px']:.4f}" if b["center_error_px"] is not None else "—"
        state = "可测量" if a["valid"] else "暂停记录"
        table.append(f"| {case.label} | {ea} | {eb} | {state} | {a['reason']} |")
        html_table.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in (case.label, ea, eb, state, a["reason"])) + "</tr>")
    opening = f"""# 光斑定位最终检查

日期：2026-10-14；软件版本：0.3.0。全部图像为可复现的合成数值图像，没有使用或保存真实 CCD 图像。用户提供的屏幕照片仅用于明确视觉需求，没有把拍屏畸变当作定位真值。

## 定义与验收

- 默认使用边界拟合圆／椭圆的中心；设置中可切换实际区域的面积中心。面积中心不按亮度加权，内部暗洞不计入面积；图上的圈始终围绕边界拟合中心绘制。
- 扣除平面背景、去除孤立坏点并轻度平滑后，以主光斑峰值的 13.5% 提取外边界；设置可调范围为 3%–80%。蓝色外缘跟随这个相对强度定义，曝光缩放不会直接改变边界。白色代表显示峰值，不代表已经饱和。
- 圆孔衍射测试采用远场 Airy 强度模型，只框中央亮斑的所选强度轮廓，不把它等同于第一暗环或高斯 1/e² 半径。一般近场衍射和实际像差尚需实验数据验证。
- 完整光斑中心误差目标：≤ max(0.5 像素，真值短半轴的 1%)；有圆／椭圆真值的轮廓，长短半轴误差各 ≤3%。缺陷测试检查提示和暂停记录是否符合预期，并列出中心偏移。
- 合束标准为近远两处中心偏差均 ≤ 当地红光**当前所选边界的拟合短半轴**的 5%。中心模式和边界百分比在一轮中必须一致；更改会清空测量。

## 本次数值结果

共 {len(all_rows)} 个检查记录，通过 {summary['passed_count']} 个：{len(cases)} 类示例 × 两种中心模式，72 组随机光斑 × 两种中心模式，以及高斯／Airy 在六种边界百分比 × 两种中心模式。

在本次完整光斑样本中，最大中心误差 **{maximum_center:.4f} px**，最大半轴相对误差 **{maximum_axis:.3%}**。这是所列采样、强度和噪声条件下的合成结果，不能解释为真实相机精度保证。

随机样本：320×240 像素，Gaussian 1/e² 长半轴 24–48 px、短半轴 14–23 px；Airy 第一暗环半径 24–48 px；峰值 1700–3400 / 4095，独立高斯噪声标准差 2–18，加入平面背景梯度。边界扫描为 3%、5%、13.5%、30%、50%、80%。

缺陷示例中的数值是相对于**缺陷发生前完整光斑中心**的偏移；面积中心会随缺损移动，不应把这种移动直接解释为算法失败。暂停记录的虚线圈只用于提示估计位置，不能进入合束测量。

"""
    controls = """## 界面控制与限制

“自动定圈”默认开启，下方为“固定圈”。点击固定后，圈的位置、长短半轴和旋转角度保持不变，按钮变为“解除固定”；实时中心、曝光和测量继续更新。固定圈标为“固定参考圈”，不会成为合束参考数据。关闭自动定圈仅隐藏定位圈并解除固定，定位和测量继续；插件的目标标记独立显示。

“曝光稳定”默认勾选，自动调节曝光以接近既定峰值目标。取消勾选停止自动调节，保持当前曝光，可在原版BeamMic手动设置；已发出的指令仍需完成并核对新帧。关闭不跳过曝光／量程验证，也不取消旧帧、饱和、无信号、抖动和截断过滤。三个有效新帧确认当前曝光后才能记录，可靠峰值不必达到80%。开关切换时，正在进行的记录保留任务并重新累计。

双峰连通性在扣除背景、抑制孤立坏点及轻度平滑后，按**观测图像整体峰值的1/e²（13.5335%）**判断，使用八邻域；与可调的显示边界百分比分开。两峰在该强度处属于同一连通区域时，以外边界的凸包拟合整体椭圆包络，不再仅因双峰及其细颈形状而拒测。包络不是处处贴合真实等强度轮廓，残差仍按实际边界计算并显示；单峰的半轴误差标准不能套用到非椭圆双峰。面积模式仍计算实际区域面积中心，不把凸包填充区计入面积。

1/e²处分离的可辨光斑、超过两个主峰、饱和、截断、低信噪比或大暗洞仍拒测。如果显示百分比过高而未包含较弱峰，会提示降低百分比，避免误把较强峰当成整个双峰。新增界面示例展示750微秒手动曝光下的连通双峰，已完成多帧记录。

圈默认标出自身的几何中心。固定后，紫色＋表示固定圈中心，白色×表示当前选定的实时光斑中心。旁边出现两个独立选项：“追踪新光斑的几何中心”默认开启，只显示当前中心，不保留轨迹；“连接新光斑中心与光圈中心”默认关闭，开启后以黄色线连接两点。关闭追踪标记也可单独显示连线；两项只影响显示，不影响定位、曝光、记录或合束。

面积中心模式下，固定圈的中心仍是边界拟合中心，连线终点使用实时面积中心。定位无效或暂时丢失时，隐藏新中心和连线，固定圈及其中心保留；恢复可靠定位后自动重新显示。解除固定或关闭自动定圈会隐藏两个选项并清除连线；下一次固定重新使用默认开关状态。

更换相机或采集坐标、改变中心／边界定义或主光斑选区，会解除固定，防止保留失效参考。轻微暗洞、局部缺口和孤立亮点标注缺陷；明显遮挡、截断、饱和、分离光斑和超过两个主峰暂停记录。恢复估计依赖剩余轮廓，不是任意缺损都能准确恢复。

算法的可靠性门槛包括：所选边界高于5倍背景噪声；单峰要求36个方位区间至少覆盖80%、连续缺失不超过5个区间；连通双峰采用整体包络并报告真实边界残差；内部大暗洞、饱和和截断仍拒测。这些门槛已用所列示例检查，仍需用真实缺陷数据扩展验证。

完整界面与合束回归结果另见《验收记录》。Windows 启动、真实 CCD 有效帧率及真实红蓝合束仍待实验电脑验收。

## 复现

在开发环境运行 `python scripts/check_spots.py`；数值明细见 `spot-check-results.json`，测试断言见 `tests/test_geometry_fitting.py`。
"""
    markdown = opening + "\n".join(table) + "\n\n## 光斑与拟合圈示意图\n\n绿色实线：可测量的边界拟合圈；橙色虚线：拒测估计；白色 +：完整光斑的已知中心；橙色 ×：当前实际区域的面积中心。\n\n"
    markdown += "\n\n".join(f"![合成光斑与拟合圈 {i+1}](figures/{p.name})" for i,p in enumerate(pictures))
    ui_pictures = [p for p in (figures / "ui-manual-connected.png", figures / "ui-fixed-circle.png", figures / "ui-independent-link.png",
                             figures / "ui-tracking-lost.png", figures / "ui-location-settings.png") if p.exists()]
    if ui_pictures:
        markdown += "\n\n## 界面控制示意图\n\n依次展示：手动曝光与连通双峰、追踪与连线同时开启、仅显示连线、光斑丢失保留参考圈、定位设置。紫色＋为固定圈中心，白色×为实时中心，黄色为连线。\n\n"
        markdown += "\n\n".join(f"![合成数据界面控制](figures/{p.name})" for p in ui_pictures)
    markdown += "\n\n" + controls
    (destination / "光斑定位检查.md").write_text(markdown, encoding="utf-8")
    # Standalone HTML embeds every figure, so copying one document is sufficient.
    body = "<h1>光斑定位最终检查</h1>"
    for paragraph in opening.split("\n\n")[1:]:
        body += render_paragraph(paragraph)
    body += "<div class='table'><table><thead><tr><th>示例</th><th>边界中心偏移 / px</th><th>面积中心偏移 / px</th><th>结果</th><th>提示</th></tr></thead><tbody>" + "".join(html_table) + "</tbody></table></div>"
    body += "<h2>光斑与拟合圈示意图</h2><p>绿色实线：可测量拟合圈；橙色虚线：拒测估计；白色 +：完整光斑已知中心；橙色 ×：面积中心。</p>"
    for picture in pictures:
        body += '<img alt="光斑拟合检查示意图" src="data:image/png;base64,' + base64.b64encode(picture.read_bytes()).decode() + '">'
    if ui_pictures:
        body += '<h2>界面控制示意图</h2><p>依次展示：手动曝光与连通双峰、追踪与连线同时开启、仅显示连线、光斑丢失保留参考圈、定位设置。紫色＋为固定圈中心，白色×为实时中心，黄色为连线。</p>'
        for picture in ui_pictures:
            body += '<img alt="合成数据界面控制" src="data:image/png;base64,' + base64.b64encode(picture.read_bytes()).decode() + '">'
    for paragraph in controls.split("\n\n"):
        body += render_paragraph(paragraph)
    style = "body{font:16px/1.75 -apple-system,BlinkMacSystemFont,'Microsoft YaHei',sans-serif;max-width:1100px;margin:35px auto;padding:0 22px;color:#233342}h1{font-size:30px}img{width:100%;margin:15px 0}.table{overflow:auto}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #cbd6df;padding:9px;text-align:left}th{background:#e9f0f5}"
    (destination / "光斑定位检查.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>beamic pro 光斑定位检查</title><style>' + style + '</style><body>' + body + '</body></html>', encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("showcase", "randomized", "thresholds")}, ensure_ascii=False, indent=2))
    if not all(r["passed"] for r in all_rows):
        raise SystemExit("Some checks failed; report includes the failures.")


if __name__ == "__main__":
    main()
