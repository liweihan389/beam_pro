"""Create an illustrated offline evaluation report, with actual BeamView captures."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/beamic-real-spots-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beamic_sdk import BeamFit, Frame
from check_real_spots import DATA, OUTPUT, ROOT, read_image
from matplotlib import font_manager
from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from beamic_pro.view import BeamView

FIGURES = OUTPUT / "figures"
PDF = ROOT / "output/pdf/真实光斑算法测试报告.pdf"
SOURCES = [
    ("牛津实验光斑数据 · Hofer、Krstajić、Smith · CC BY", "https://doi.org/10.5287/bodleian:JbDXrnQN1"),
    ("牛津作者代码、数据读取与坐标定义", "https://github.com/Dipolar-Quantum-Gases/nn-beam-profiling"),
    ("牛津论文：标注来自二维高斯拟合及人工检查", "https://arxiv.org/pdf/2202.07801"),
    ("Scott Prahl · laserbeamsize · MIT许可", "https://github.com/scottprahl/laserbeamsize"),
    ("laserbeamsize困难实拍说明与12位文件编码", "https://laserbeamsize.readthedocs.io/en/latest/08-Harder-Images.html"),
    ("Davide Torielli · Laser Pointer Spot数据 · CC BY 4.0", "https://zenodo.org/records/15230870"),
    ("NIST：附件下载超时，未纳入成绩", "https://doi.org/10.18434/M32233"),
]


def selected_cases(data):
    rows = data["rows"]
    lookup = {r["id"]: r for r in rows}
    singles = sorted([r for r in rows if r["source"] == "oxford" and r["beam_count"] == 1 and r["fit"]["valid"]],
                     key=lambda r: (r["center_error_px"], r["id"]))
    median = singles[len(singles)//2]
    separated = next(r for r in rows if r["source"] == "oxford" and r["beam_count"] == 2
                     and "分离" in r["fit"]["reason"])
    rgb = next(r for r in rows if r["source"] == "rgb" and r["boxes"] and "饱和" in r["fit"]["reason"])
    specs = [
        (median["id"], "典型单光斑", "按已接受单光斑的中心偏差排序，选取中间样本。绿色圈为默认13.5%边界拟合，黄色圈为作者高斯参考换算后的同一强度边界。"),
        ("oxford-140", "细长椭圆光斑", "选取已接受单光斑中参考长短轴比最大的样本，检查旋转椭圆的适配；未按此图调整参数。"),
        ("oxford-440", "中心偏差最大的已接受样本", "刻意展示最差中心偏差。当前算法仍允许测量，但与作者参考中心差超过参考短半轴的5%；“有效”不能解释为每张图都达到该比较尺度。"),
        ("oxford-405", "尺寸偏差最大的已接受样本", "选取长短半轴中相对偏差最大的一张。参考来自整体高斯拟合，当前结果来自低强度几何边界；这里用于比较定义与采样的影响。"),
        ("oxford-1286", "单光斑拒测样本", "这是210张单光斑中唯一被拒测的图片。展示它是为了避免只挑成功结果；不能仅凭拒测就断言原始光斑不可测。"),
        (separated["id"], "分离光斑：要求选区", "从双光斑拒测样本按文件编号选取。当前默认处理一个主光斑，画面存在独立光斑时要求手动框选，不执行多目标逐一定位。"),
        ("oxford-727", "连通双峰：允许整体包络", "这是840张多光斑图中唯一被接受的图片，识别为1/e²连通双峰。整体包络与凹颈轮廓并不贴合：IoU约0.728、完整轮廓残差约27.4%，应展示为兼容包络，不能称为精密椭圆匹配。"),
        ("lbs-t-hene", "真实HeNe单光斑", "实拍样本，缺少独立几何中心标注。可观察定圈与轮廓一致程度，不能将另一算法的二阶矩中心直接视作本算法的几何真值。"),
        ("lbs-TEM01_100mm", "真实高阶双峰光斑", "按作者说明解码12位数据；未删除图像顶部异常像素，未提供选区。当前算法允许1/e²处连通的双峰，并保留包络不规则提示。"),
        ("lbs-TEM02_100mm", "真实多峰：暂停记录", "高阶模式包含多个主峰；当前算法暂停测量。虚线找光圈不等于获得了可靠的几何测量结果。"),
        ("lbs-k-200mm", "中心饱和：只有找光圈", "保留8位文件原值。算法依据文件量程检测到光斑区域饱和并拒测；静态图片无法通过降低曝光补救，实机应重新采集。"),
        ("lbs-FBWR_151", "强背景：13.5%边界不可靠", "背景起伏使13.5%边界低于可靠噪声水平，当前设置下暂停记录。没有为获得圆圈而提高阈值或改变参数。"),
        (rgb["id"], "复杂场景RGB照片", "找光圈没有落在标注激光位置，返回中心甚至超出画面；虽然最终拒测，参考圈仍可能误导。该图仅作RGB转灰度后的压力测试，不是直接拍摄的线性强度分布；此类441张均未得到可测结果。"),
    ]
    return [(lookup[identity], title, note) for identity, title, note in specs]


def qpixmap(array):
    array = np.ascontiguousarray(array, dtype=np.uint8)
    h, w = array.shape[:2]
    kind = QImage.Format.Format_RGB888 if array.ndim == 3 else QImage.Format.Format_Grayscale8
    return QPixmap.fromImage(QImage(array.data, w, h, array.strides[0], kind).copy())


def capture(app, row, number, title):
    pixels, full_scale, _ = read_image(DATA / row["path"], row["source"])
    frame = Frame(1, pixels, 0, 1, 0, "archived-offline", full_scale)
    fit = BeamFit.from_dict(row["fit"])
    window = QWidget()
    window.setStyleSheet("QWidget{background:#101d2d;color:#e4edf7;font-family:'PingFang SC';font-size:15px} QLabel{padding:4px} QGraphicsView{border:1px solid #3b5066}")
    layout = QVBoxLayout(window)
    headline = QLabel(f"beamic pro 0.4.1   |   真实数据离线测试   |   {number:02d} {title}")
    headline.setStyleSheet("font-size:22px;color:#7ce5cb;font-weight:600")
    layout.addWidget(headline)
    label = QLabel(f"文件：{Path(row['path']).name}   ·   原图 {pixels.shape[1]} × {pixels.shape[0]}   ·   单帧 / 默认13.5% / 无框选")
    layout.addWidget(label)
    row_layout = QHBoxLayout()
    raw_scene = QGraphicsScene()
    with Image.open(DATA / row["path"]) as im:
        raw = np.asarray(im.convert("RGB")) if im.mode in ("RGB", "RGBA") else np.uint8(np.clip(pixels / full_scale, 0, 1) * 255)
    raw_scene.addPixmap(qpixmap(raw))
    raw_view = QGraphicsView(raw_scene)
    raw_view.setBackgroundBrush(Qt.GlobalColor.black)
    view = BeamView()
    raw_title = "原图全画面 · RGB原色" if raw.ndim == 3 else "原始灰度全画面 · 仅显示按本图最大值拉伸"
    if raw.ndim == 2:
        raw = np.uint8(np.clip(pixels / max(1, float(pixels.max())), 0, 1) * 255)
        raw_scene.clear()
        raw_scene.addPixmap(qpixmap(raw))
    for name, widget in [(raw_title, raw_view), ("当前算法定圈 · 有圈时局部放大", view)]:
        column = QVBoxLayout()
        column.addWidget(QLabel(name))
        column.addWidget(widget)
        row_layout.addLayout(column)
    layout.addLayout(row_layout, 1)
    status = QLabel(("允许测量：" if fit.valid else "暂停记录：") + fit.reason)
    status.setWordWrap(True)
    status.setStyleSheet("color:" + ("#7ce5cb" if fit.valid else "#ffbd70"))
    layout.addWidget(status)
    legend = QLabel("绿圈/十字：算法边界与中心；橙色虚线：拒测时仅供找光；黄色圈/十字：作者高斯参考（若有）。")
    legend.setWordWrap(True)
    layout.addWidget(legend)
    window.resize(1440, 730)
    window.show()
    app.processEvents()
    raw_view.fitInView(raw_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    view.set_image(frame, fit)
    shapes = [{"kind": "ellipse", "x": r["x"], "y": r["y"], "radius": r["major"],
               "radius_y": r["minor"], "angle_rad": r["angle"], "label": "作者参考"} for r in row["references"]]
    view.overlays(fit, shapes)
    if fit.radius_minor > 0:
        margin = max(16, fit.radius_major * 1.7)
        cx, cy = fit.boundary_center
        view.fitInView(QRectF(cx-margin, cy-margin, 2*margin, 2*margin), Qt.AspectRatioMode.KeepAspectRatio)
    app.processEvents()
    output = FIGURES / f"case-{number:02d}.png"
    assert window.grab().save(str(output))
    window.close()
    app.processEvents()
    return output


def metric_text(row):
    fit = row["fit"]
    lines = [f"算法判定：{'允许测量' if fit['valid'] else '暂停记录'}；{fit['reason']}。"]
    if fit["radius_minor"] > 0:
        lines.append(f"边界中心 ({fit['center_x']:.3f}, {fit['center_y']:.3f}) px；长/短半轴 {fit['radius_major']:.3f} / {fit['radius_minor']:.3f} px。")
    if row.get("center_error_px") is not None:
        lines.append(f"与作者参考中心差 {row['center_error_px']:.3f} px（参考短半轴的{row['center_error_fraction']*100:.2f}%）；长/短半轴相对差 {row['major_relative_error']*100:.2f}% / {row['minor_relative_error']*100:.2f}%。")
    if row.get("contour_iou") is not None:
        lines.append(f"观测边界与拟合椭圆的区域IoU {row['contour_iou']:.3f}；完整轮廓归一化RMS {row['all_contour_rms_fraction']*100:.2f}%。")
    return " ".join(lines)


def chart(data):
    for font in [Path("/System/Library/Fonts/PingFang.ttc"), Path("/System/Library/Fonts/STHeiti Light.ttc"),
                 Path("/System/Library/Fonts/Supplemental/Songti.ttc")]:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    single = [r for r in data["rows"] if r["source"] == "oxford" and r["beam_count"] == 1 and r["fit"]["valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), layout="constrained")
    values = sorted(r["center_error_px"] for r in single)
    axes[0].plot(values, np.arange(1, len(values)+1)/len(values)*100, color="#176b9b", linewidth=2)
    axes[0].axvline(np.percentile(values, 95), color="#db9138", linestyle="--", label="95百分位")
    axes[0].set(xlabel="与作者参考中心的距离 / px", ylabel="累计比例 / %", title="牛津单光斑 · 仅209张已接受样本")
    axes[0].legend()
    names = ["单光斑", "双光斑", "三光斑", "四光斑", "五光斑"]
    valid = [data["summary"][f"oxford_{i}_beam"]["accepted"] for i in range(1, 6)]
    axes[1].bar(names, valid, color="#188a79", label="允许测量")
    axes[1].bar(names, np.array([210]*5)-valid, bottom=valid, color="#dfb183", label="暂停记录")
    axes[1].set(ylabel="图片数", title="默认全画面测试 · 每组210张")
    axes[1].legend()
    target = FIGURES / "summary.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    return target


def build_report(data, cases, summary_image):
    single = data["summary"]["oxford_1_beam"]
    opening = ("本次将beamic pro 0.4.1当前生产算法直接用于公开的真实实验图像，未修改算法、阈值或拒测条件。"
               "共测试1,988个图像文件：牛津1,500张（作者标注1,050张，另附未标注450张）、laserbeamsize实拍47张、RGB激光指示器照片441张。NIST附件下载超时，未计入测试。")
    findings = [
        "牛津210张单光斑中209张允许测量、1张拒测。209张已接受样本的中心参考差：中位数0.092 px、95百分位0.291 px、最大1.361 px。",
        "长/短半轴相对作者参考的差值，中位数分别2.42% / 3.06%，95百分位分别3.88% / 4.56%；最差长半轴差11.60%。",
        "203/210张单光斑同时满足“被接受且中心参考差不超过参考短半轴5%”；6张被接受但超过该比较尺度，另1张拒测。它不是实机合束验收结果。",
        "牛津840张多光斑中839张拒测；1张被识别为连通双峰并允许整体包络。多光斑拒测符合当前单主光斑流程，不能将840张全部当作定位失败或全部当作正确拒测。",
        "laserbeamsize实拍47张：20张接受、27张拒测。拒测中16张为光斑区域文件量程饱和、8张为分离光斑/多主峰、2张为边界低于噪声、1张拟合不可靠。缺少逐图独立真值，接受率不等于准确率。",
        "441张复杂场景RGB照片全部拒测：300张信号/信噪比不足、138张边界低于噪声、2张饱和、1张截断。其中393张有激光框标注，48张未标注；未标注不自动当作无激光负例。",
    ]
    methods = [
        "输入：读取灰度TIFF、PGM、PNG的数值，不读取牛津伪彩色PNG。RGB照片仅通过Pillow亮度转换进入压力测试；未将其视为线性辐照度数据。所有图片保持原始分辨率和朝向。",
        "设置：默认13.5%相对峰值边界、边界拟合几何中心、最低SNR 12、最多1800个拟合点；全画面自动处理，不用参考标注生成ROI、不人工指定中心、不删异常像素、不调曝光。",
        "编码：按作者说明将12位PGM的16位容器右移4位，量程设为4095；8位文件量程255。这里的饱和判定仅依据文件编码量程，缺少相机原始曝光和ADC状态，不能补验硬件饱和行为。",
        "坐标：牛津作者以0.5、1.5……定义像素中心，转换到当前程序0、1……时，参考x/y均减0.5。作者半径对应1/e²；比较13.5%边界时乘sqrt[-ln(0.135)/2]，并统一长短轴及角度。",
        "参考定义：牛津参考来自二维高斯/多高斯拟合及人工检查，不是独立计量真值。当前边界几何中心和面积中心都可能与强度高斯中心有合理差异；误差数字应读作参考差。面积中心结果也保存在全量JSON/CSV。",
        "轮廓指标：IoU是拟合椭圆与当前平滑/背景扣除后所选连通域（填洞）的交并比；完整轮廓RMS使用全部边界点，不使用生产算法的离群点剔除结果。二者衡量内部一致性，不是独立准确度。",
        "统计：单光斑中心/半轴统计的主表分母为209张已接受样本，拒测另计。多目标不给每个作者标注分配一个自动结果。接近圆形的参考长短轴比<1.15时不统计角度差。",
        "数据使用：牛津光斑由空间光调制器产生并由相机实拍。除作者标注的1050张外，压缩包还附450张无标注图，本次全测（84张接受、366张拒测），只作处理行为统计。训练/验证划分沿用作者，本次没有训练模型或调参；原验证集50张单光斑单列。截图复用BeamView，原灰度仅显示时按最大值拉伸，不改变计算数据，不代表新增了主程序文件导入功能。",
    ]
    limits = [
        "对普通相机直拍、足够信噪比的单光斑，当前默认定位与椭圆轮廓有较好一致性；6张接受但中心参考差>5%的案例说明“有效”状态不能单独承担精度承诺。",
        "连通双峰兼容已在真实TEM01及牛津样本中出现。整体包络允许凹颈偏离，应在后续UI考虑区分“整体包络可定位”和“轮廓贴合良好”，并展示轮廓质量指标。此次不改变现有规则。",
        "尺寸相对高斯参考存在约2%-3%的典型差异。平滑、离散采样、背景估计和中心/宽度定义可能影响差异；本次未做因果隔离，不能直接用一个固定比例校正。",
        "复杂背景RGB照片需要单独的候选检测、颜色或场景分割流程。当前数据结果不足以支持在这类场景使用自动几何测量；改进需另设保留测试集。",
        "本次不覆盖带可靠几何标注的真实圆孔衍射中央亮斑，也没有SP932U原始实时序列。因此不能验证本机物理尺寸精度、20有效帧/秒、自动曝光、帧同步、时间稳定性或真实合束。",
        "报告保留最差、拒测和低贴合案例。13张截图按典型中位数、极端长短轴比、最大中心/尺寸参考差及明确失败类型选取，不代表随机抽样通过率。",
    ]
    headers = ["数据分组", "图片数", "允许测量", "暂停记录", "说明"]
    table = [headers, ["牛津标注单光斑", "210", "209", "1", "中心/半轴参考比较"],
             ["牛津标注多光斑", "840", "1", "839", "其中1张为连通双峰包络"],
             ["牛津额外未标注图", "450", "84", "366", "不参与参考准确度统计"],
             ["laserbeamsize实拍", "47", "20", "27", "没有独立几何真值"],
             ["复杂场景RGB", "441", "0", "441", "393张带激光框标注"],
             ["NIST附件", "0", "-", "-", "下载超时，不计入"]]
    exact = [
        ["指标 / 已接受牛津单光斑 n=209", "中位数", "95百分位", "最大值"],
        ["中心参考差 / px", ".092", ".291", "1.361"],
        ["中心参考差 / 参考短半轴", "1.17%", "3.76%", "13.57%"],
        ["长半轴相对参考差", "2.42%", "3.88%", "11.60%"],
        ["短半轴相对参考差", "3.06%", "4.56%", "7.44%"],
        ["角度参考差 / 度（n=159）", ".559", "2.482", "4.614"],
    ]
    assert single["accepted"] == 209 and len(data["rows"]) == 1988
    assert sum(r.get("within_5pct_reference_minor", False) for r in data["rows"]) == 203
    html_parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>真实光斑算法测试报告</title>",
                  "<style>body{font:16px/1.8 -apple-system,'Microsoft YaHei',sans-serif;max-width:1380px;margin:32px auto;padding:0 24px;background:#f2f6fa;color:#183149}section{background:white;padding:24px;border-radius:12px;margin:24px 0}h1,h2{line-height:1.4}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:14px 0}td,th{padding:9px;border:1px solid #cfdae5;text-align:left}.note{padding:14px;background:#fff0cd}.meta{font-size:13px;word-break:break-all;color:#5b6b7a}</style>",
                  "<h1>真实光斑算法测试报告</h1><p>beamic pro 0.4.1 · 2026-10-15 · 真实静态数据 / 当前默认算法</p>",
                  "<p class='note'>"+opening+"</p>"]
    markdown = ["# 真实光斑算法测试报告", "beamic pro 0.4.1 · 2026-10-15", opening]
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    style = ParagraphStyle("body", fontName="STSong-Light", fontSize=10.3, leading=16, textColor=colors.HexColor("#183149"), wordWrap="CJK", spaceAfter=7, alignment=TA_LEFT)
    heading = ParagraphStyle("heading", parent=style, fontSize=21, leading=27, spaceAfter=14)
    small = ParagraphStyle("small", parent=style, fontSize=8, leading=11)
    story = []
    def p(text, kind=style):
        # The PDF CJK base font lacks these glyphs; keep the mathematics legible.
        text = text.replace("²", "^2").replace("·", " / ").replace("ć", "c").replace("→", " -> ")
        return Paragraph(html.escape(text), kind)
    def table_html(rows):
        return "<table>"+"".join("<tr>"+"".join(f"<{'th' if i==0 else 'td'}>{html.escape(v)}</{'th' if i==0 else 'td'}>" for v in row)+"</tr>" for i,row in enumerate(rows))+"</table>"
    def pdf_table(rows, widths):
        result = Table([[p(v, small) for v in row] for row in rows], colWidths=widths)
        result.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5f0f6")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd7e2")),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
        return result
    story += [p("真实光斑算法测试报告", heading), p("beamic pro 0.4.1 | 2026-10-15 | 真实静态数据"), p(opening), pdf_table(table, [49*mm,22*mm,26*mm,26*mm,143*mm]), Spacer(1,7*mm)]
    html_parts += [table_html(table), "<h2>主要发现</h2><ul>"]
    for text in findings:
        story.append(p(text))
        html_parts.append("<li>"+html.escape(text)+"</li>")
        markdown.append("- "+text)
    html_parts.append("</ul>")
    story += [PageBreak(), p("单光斑定量比较", heading), pdf_table(exact,[118*mm,49*mm,49*mm,50*mm]), Spacer(1,4*mm), PDFImage(str(summary_image), width=252*mm, height=82*mm)]
    note = ("原作者验证集的50张单光斑全部被接受；中心参考差中位数0.090 px、95百分位0.290 px、最大0.510 px，48/50同时满足5%参考短半轴尺度。"
            "上述95百分位描述样本分布，不是95%置信区间或测量不确定度。原作者训练集160张中159张接受。")
    story.append(p(note))
    encoded = base64.b64encode(summary_image.read_bytes()).decode()
    html_parts += ["<section><h2>单光斑定量比较</h2>",table_html(exact),f"<img src='data:image/png;base64,{encoded}' alt='中心参考差分布及分组接受数量'>", "<p>"+note+"</p></section>"]
    markdown += ["## 单光斑定量比较", note, "![分布](figures/summary.png)"]
    story += [PageBreak(), p("测试方法与比较边界", heading)]
    html_parts.append("<section><h2>测试方法与比较边界</h2>")
    for text in methods:
        story.append(p(text))
        html_parts.append("<p>"+html.escape(text)+"</p>")
        markdown.append(text)
    story.append(p("算法文件SHA256："+data["fitter_sha256"],small))
    html_parts.append("<p class='meta'>算法SHA256："+data["fitter_sha256"]+"</p></section>")
    selected = []
    for number,(row,title,note,image) in enumerate(cases,1):
        story += [PageBreak(),p(f"{number:02d}  {title}",heading),p(f"样本 {row['id']} | {'允许测量' if row['fit']['valid'] else '暂停记录'}"),
                  PDFImage(str(image),width=244*mm,height=244*mm*730/1440),Spacer(1,3*mm),p(note),p(metric_text(row),small),p("文件："+row["path"],small)]
        encoded=base64.b64encode(image.read_bytes()).decode()
        html_parts += [f"<section><h2>{number:02d} {title}</h2><p>{html.escape(note)}</p>",
                       f"<img src='data:image/png;base64,{encoded}' alt='{html.escape(title)}，真实数据离线测试视图截图'>",
                       "<p>"+html.escape(metric_text(row))+"</p><p class='meta'>"+html.escape(row["path"])+"</p></section>"]
        markdown += [f"## {number:02d} {title}",note,f"![{title}](figures/{image.name})",metric_text(row),"文件："+row["path"]]
        selected.append({"id":row["id"],"title":title,"note":note,"image":"figures/"+image.name})
    story += [PageBreak(),p("结论、限制与改进方向",heading)]
    html_parts.append("<section><h2>结论、限制与改进方向</h2>")
    for text in limits:
        story.append(p(text))
        html_parts.append("<p>"+html.escape(text)+"</p>")
        markdown.append(text)
    story += [PageBreak(),p("来源、许可与复现",heading)]
    html_parts.append("</section><section><h2>来源、许可与复现</h2>")
    for name,url in SOURCES:
        story += [p(name),Paragraph(f"<link href='{html.escape(url,quote=True)}' color='#176b9b'>{html.escape(url)}</link>",small)]
        html_parts.append(f"<p><a href='{html.escape(url,quote=True)}'>{html.escape(name)}</a></p>")
        markdown.append(f"[{name}]({url})")
    reproducible = ("数据缓存位于build/real-spots，不进入主程序安装包。公开下载URL、成功/失败状态、文件SHA256和仓库版本在results.json中保存；每图结果在results.csv中保存。"
                    "复现顺序：fetch_real_spots.py → check_real_spots.py → report_real_spots.py。Python、NumPy、SciPy和Pillow版本见results.json。"
                    "本次只新增离线验证与报告工具，没有修改生产拟合算法。截图为真实数据的Qt测试视图，未经AI生成。")
    story += [Spacer(1,5*mm),p(reproducible)]
    html_parts += ["<p>"+reproducible+"</p></section></html>"]
    markdown.append(reproducible)
    (OUTPUT/"真实光斑算法测试报告.html").write_text("".join(html_parts),encoding="utf-8")
    (OUTPUT/"真实光斑算法测试报告.md").write_text("\n\n".join(markdown),encoding="utf-8")
    (OUTPUT/"selected-cases.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")
    PDF.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas, document):
        canvas.setFont("STSong-Light",8)
        canvas.setFillColor(colors.HexColor("#63778b"))
        canvas.drawString(15*mm,9*mm,"beamic pro | 真实数据离线测试 | 实机验收待进行")
        canvas.drawRightString(282*mm,9*mm,str(document.page))
    document=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),leftMargin=15*mm,rightMargin=15*mm,topMargin=12*mm,bottomMargin=15*mm,
                               title="真实光斑算法测试报告",author="beamic pro",subject="真实公开数据上的默认光斑定位与边界拟合测试")
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({"pdf":str(PDF),"html":str(OUTPUT/"真实光斑算法测试报告.html"),"screenshots":len(cases)},ensure_ascii=False),flush=True)


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    data=json.loads((OUTPUT/"results.json").read_text())
    assert hashlib.sha256((ROOT/"src/beamic_pro/fitting.py").read_bytes()).hexdigest()==data["fitter_sha256"]
    app=QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    cases=[]
    for i,(row,title,note) in enumerate(selected_cases(data),1):
        cases.append((row,title,note,capture(app,row,i,title)))
    build_report(data,cases,chart(data))


if __name__=="__main__":
    main()
