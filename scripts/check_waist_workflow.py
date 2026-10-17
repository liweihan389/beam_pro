"""Drive actual Qt controls and actual plugin processes; screenshot every step."""
from __future__ import annotations

import base64
import html
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from PySide6.QtWidgets import QApplication

from beamic_pro.app import Window, configure_application

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'plugins/waist'))
from waist_math import WidthPoint, fit_caustic


def main():
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    steps = []
    figures = ROOT/'docs/figures/waist'
    figures.mkdir(parents=True, exist_ok=True)

    def state(window):
        return window.plugin_states.get('waist', {}).get('data', {})

    def wait(window, condition, seconds=12):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            app.processEvents()
            if condition():
                app.processEvents()
                return
            time.sleep(.004)
        raise AssertionError(str(state(window))+window.plugin_states.get('waist', {}).get('text', '')+window.log.toPlainText())

    def shot(window, title, action, expected):
        if getattr(window.source, 'waist_mode', False):
            wait(window, lambda: window.last_fit is not None and window.last_fit.valid and window.last_fit.center_x > 200)
        app.processEvents()
        panel = window.waist_panel
        panel.verticalScrollBar().setValue(0)
        # Show the start of the instructions, rather than a previous text scroll.
        panel.text.verticalScrollBar().setValue(0)
        app.processEvents()
        name = f'step-{len(steps)+1:02d}.png'
        window.grab().save(str(figures/name))
        steps.append({'number': len(steps)+1, 'title': title, 'action': action, 'expected': expected,
                      'image': 'figures/waist/'+name, 'state': state(window)})

    def press(window, action):
        wait(window, lambda: action in window.waist_panel.buttons)
        window.waist_panel.buttons[action].click()

    def record(window, z):
        wait(window, lambda: 'record' in window.waist_panel.buttons)
        count = state(window).get('count', 0)
        window.waist_panel.controls['z_mm'].setValue(z)
        press(window, 'record')
        wait(window, lambda: state(window).get('count', 0) > count)

    with tempfile.TemporaryDirectory(prefix='beamic-waist-guide-') as data_dir:
        window = Window(data_dir, demo=True)
        window.show()
        try:
            wait(window, lambda: window.capture_ready and window.exposure.state.stable)
            window.tabs.setCurrentWidget(window.waist_panel)
            shot(window, '启动模拟光路，打开束腰页', '点击“模拟光路”，再打开“束腰测量”标签。',
                 '顶部明确显示模拟数据；点击“启用束腰测量插件”。实机测试应改为连接 BeamMic 并通过曝光预检。')
            press(window, 'activate')
            wait(window, lambda: 'configure' in window.waist_panel.buttons and window.last_frame.coordinate_key.startswith('sim-waist') and window.exposure.state.stable)
            shot(window, '检查测量设置', '保留默认摆放公差±10 mm、SP932U有效像元3.45 μm、直测倍率1。默认直径门槛5%，位置门槛5 mm。',
                 '共同宽度系统公差默认±2%，属于保守误差预算假设。保持绝对位置显示关闭；确认后开始第1点。')
            press(window, 'configure')
            wait(window, lambda: 'record' in window.waist_panel.buttons)
            shot(window, '第1点设为零', '将第一位置作为感光面相对零点，l1固定为0。点击记录第1点。',
                 '需要20个新的、未饱和、曝光稳定的有效帧；模拟相机将自动放到输入位置。')
            record(window, 0)
            shot(window, '第1点已记录，输入 l2', '相机移到第二位置，填写实际相对距离。本模拟输入l2=150 mm，然后记录。',
                 '表格有第1点的两个二阶矩直径；实际相机需要人工移动，不由程序控制。')
            record(window, 150)
            shot(window, '第2点已记录，输入 l3', '相机移到第三位置。本模拟填写l3=350 mm，然后记录。',
                 '输入有正负方向；每点可覆盖公差，数值表示±范围而非标准差。')
            record(window, 350)
            wait(window, lambda: state(window).get('stage') == 'initial')
            shot(window, '三点初算，保留预测', '检查X/Y方向束腰位置、最小直径及95%不确定区间。点击“保留初算预测，开始独立验证”。',
                 '此时尚未验证。初算模型保持不变，接下来的两点独立检查；若初测不足，会要求增加建模点。')
            press(window, 'continue')
            wait(window, lambda: 'record' in window.waist_panel.buttons)
            window.waist_panel.controls['z_mm'].setValue(170)
            press(window, 'preview')
            wait(window, lambda: '预测范围' in window.waist_panel.title.text())
            shot(window, '第4点：先看预测范围', '本模拟将l4设为170 mm，点击“查看此位置的预测范围”，再点击记录第4点。',
                 '预测含摆放公差和初算模型的不确定度。新测量进入检验时还会加入实测误差。')
            record(window, 170)
            shot(window, '第4点独立检查', '查看第4点的实测直径及“一致／不一致”标记。保持原预测，继续移动到第5点。',
                 '一处一致尚不算完成。原三点模型未利用第4点重新预测第5点。')
            window.waist_panel.controls['z_mm'].setValue(530)
            press(window, 'preview')
            wait(window, lambda: '预测范围' in window.waist_panel.title.text())
            shot(window, '第5点：检查扩张区域', '本模拟填写l5=530 mm，先查看预测，再记录第5点。',
                 '软件建议距离为理论建议，实际位置须可摆放、位于透镜之后且光斑完整。填写实际读数。')
            record(window, 530)
            shot(window, '五点重算：预测一致，仍需补测', '检查两验证点标记和五点拟合结果，阅读位置与直径区间。',
                 '默认±10 mm下，此模拟一般显示“预测一致，需要补测”：区间略超5 mm位置门槛，不会仅因两点一致而宣称成功。')
            assert state(window)['outcome'] == '预测一致，需要补测'
            for z in (260, 80, 430):
                if state(window)['outcome'] == '验证通过，精度达标':
                    break
                press(window, 'add')
                wait(window, lambda: 'record' in window.waist_panel.buttons)
                window.waist_panel.controls['z_mm'].setValue(z)
                shot(window, f'补测第{state(window)["count"]+1}点', f'保留既有数据，本模拟补测位置{z} mm。检查并填写实际位置后记录。',
                     '补测不会篡改原来的独立验证结果；同一个位置仅增加帧数不能改善摆放误差。')
                record(window, z)
            assert state(window)['outcome'] == '验证通过，精度达标', state(window)
            shot(window, '补测后达到设定精度', '检查最终区间以及完整测点表。需要查看较后测点时滚动表格；测量说明可在文字区滚动。',
                 '模拟真值：X束腰位置160 mm、最小直径110 μm；Y位置180 mm、直径140 μm。实际输出接近真值。')
            completed = state(window)
            press(window, 'reset')
            wait(window, lambda: 'configure' in window.waist_panel.buttons)
            window.waist_panel.controls['tolerance_mm'].setValue(1)
            press(window, 'configure')
            wait(window, lambda: 'record' in window.waist_panel.buttons)
            for z in (0, 150, 350):
                record(window, z)
            press(window, 'continue')
            wait(window, lambda: 'record' in window.waist_panel.buttons)
            original = window.source.waist_truth
            window.source.waist_truth = {**original, 'radius_um': (77., 70.)}
            record(window, 170)
            window.source.waist_truth = original
            shot(window, '异常示例：第4点不符合预测', '这是额外的故障测试：公差改为±1 mm，仅第4点人为改变模拟光束X宽度。',
                 '第4点标记“验证不一致”；不要把这一点当成原来同一束稳定光的有效验证。')
            record(window, 530)
            assert state(window)['outcome'] == '验证不一致'
            shot(window, '验证失败，选择重录或重新预测', '检查距离、曝光、选区和光束是否改变。可指定点编号重录，或重新开始；重新建模后仍需两个新验证点。',
                 '失败数据仍可查看。重录前三个建模点会撤销旧预测并清除旧验证点；重录首点需要重建全部相对距离。')
            failed = state(window)
        finally:
            window.close()
            app.processEvents()

    # Repeated synthetic experiments check that coarse placement is reflected
    # in intervals. Coverage is measured, not assumed to be universally 95%.
    rng = np.random.default_rng(334)
    coverage = np.zeros((2, 2), dtype=int)
    accepted = 0
    for k in range(200):
        z = np.array([0., 150, 350, 170, 530])
        displacement = rng.uniform(-10, 10, 5)
        true = np.sqrt(np.array([110., 140.])**2+(np.array([.7, .8])*(z[:, None]+displacement[:, None]-np.array([160., 180.])))**2)
        observed = true+rng.normal(0, .3, true.shape)
        points = [WidthPoint(zi, 10, tuple(di), (.3, .3)) for zi, di in zip(z, observed, strict=True)]
        try:
            result = fit_caustic(points, draws=1800, seed=k)
        except ValueError:
            continue
        accepted += 1
        truth = np.array([[160-displacement[0], 110], [180-displacement[0], 140]])
        coverage += (result.intervals[0, :, :2] <= truth) & (result.intervals[1, :, :2] >= truth)
    report = {'version': '0.4.0', 'simulated': True, 'hardware_acceptance': 'pending',
              'step_count': len(steps), 'completed': completed, 'failure_detected': failed['outcome'],
              'coverage_trials': 200, 'physical_trials': accepted,
              'empirical_coverage_z_diameter_xy': (coverage/accepted).tolist(), 'steps': steps}
    (ROOT/'docs/waist-workflow-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    intro = ('本说明由真实程序窗口、真实插件子进程和合成数值图像生成。不是界面设计稿，也不是实机验收。'
             '默认±10 mm未被暗中改小；达到门槛的示例通过增加测点完成。'
             '所有图片都为模拟数据。正式测试保持同一相机方向、有效像元、倍率、滤片及积分选区不变。')
    markdown = '# 束腰测量插件：一步一图\n\n'+intro+'\n\n'
    sections = []
    for step in steps:
        markdown += f"## 第{step['number']}步：{step['title']}\n\n操作：{step['action']}\n\n检查：{step['expected']}\n\n![第{step['number']}步]({step['image']})\n\n"
        data = base64.b64encode((ROOT/'docs'/step['image']).read_bytes()).decode()
        sections.append(f'<section id="step-{step["number"]}"><h2>第{step["number"]}步：{html.escape(step["title"])}</h2>'
                        f'<p><b>操作：</b>{html.escape(step["action"])}</p><p><b>检查：</b>{html.escape(step["expected"])}</p>'
                        f'<img alt="第{step["number"]}步实际程序截图，模拟数据" src="data:image/png;base64,{data}"></section>')
    notes = ('三点初算的不确定度来自外部误差预算，而不是三点残差。公差按均匀分布，位置读数共享零点误差，'
             '20帧统计考虑正相关并保留0.2%局部误差下限，共同宽度系统公差默认±2%。'
             '验证使用冻结模型和新点的位置／尺寸误差，两个点×两个固定方向采用保守联合检验。'
             '扩展五点后仍检验整体传播一致性。区间为误差模型下的近似传播区间，非普遍准确度承诺。'
             '200次设定条件的误差实验中，位置覆盖率X=95.0%、Y=93.5%，直径两方向均100%；'
             '仅适用于该模拟条件。未知系统偏差、背景模型错误、旋转像散和成像倍率未校准需实机检查。')
    markdown += '## 数值验证与实机边界\n\n'+notes+'\n\n真实Windows／BeamMic／SP932U验证待进行。\n'
    (ROOT/'docs/束腰测量操作说明.md').write_text(markdown, encoding='utf-8')
    contents = '<nav>'+''.join(f'<a href="#step-{s["number"]}">{s["number"]}. {html.escape(s["title"])}</a>' for s in steps)+'</nav>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>束腰测量：一步一图</title><style>'
    page += 'body{max-width:1280px;margin:30px auto;padding:0 24px;font:16px/1.75 -apple-system,"Microsoft YaHei",sans-serif;color:#172b40;background:#f4f7fa}section{background:white;padding:22px;margin:24px 0;border-radius:12px}img{width:100%;height:auto;border:1px solid #cbd7e2}nav a{display:block}h1,h2{line-height:1.4}.note{background:#fff0cd;padding:15px;border-radius:8px}</style>'
    page += '<h1>束腰测量插件：一步一图</h1><p class="note">'+html.escape(intro)+'</p>'+contents+''.join(sections)
    page += '<section><h2>数值验证与实机边界</h2><p>'+html.escape(notes)+'</p><p>真实Windows／BeamMic／SP932U验证待进行。</p></section></html>'
    (ROOT/'docs/束腰测量操作说明.html').write_text(page, encoding='utf-8')
    print(json.dumps({'steps': len(steps), 'complete_outcome': completed['outcome'],
                      'final_count': completed['count'], 'failure_detected': failed['outcome'],
                      'coverage': (coverage/accepted).tolist()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
