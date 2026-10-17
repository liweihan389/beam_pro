"""Illustrate alignment using real controls, frames, and a real plugin worker."""
from __future__ import annotations

import base64
import html
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from beamic_sdk import Measurement
from beamic_sdk.alignment import verify
from PySide6.QtWidgets import QApplication, QPushButton

from beamic_pro import __version__
from beamic_pro.app import Window, configure_application

ROOT = Path(__file__).resolve().parents[1]


def main():
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    steps, measurements, moves = [], [], []
    figures = ROOT/'docs/figures/alignment'
    figures.mkdir(parents=True, exist_ok=True)

    def wait(window, condition, timeout=12):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if condition():
                return
            time.sleep(.004)
        raise AssertionError(window.guidance_text.text()+' / '+window.capture_label.text()+'\n'+window.log.toPlainText())

    def settle(window, seconds=.15):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.004)

    def shot(window, title, demo, real, check):
        settle(window)
        assert 'DM固定' in window.alignment_notice.text()
        name = f'step-{len(steps)+1:02d}.png'
        assert window.grab().save(str(figures/name))
        steps.append({'number': len(steps)+1, 'title': title, 'demo_action': demo,
                      'real_action': real, 'expected': check, 'image': 'figures/alignment/'+name,
                      'guidance': window.guidance_text.text(), 'capture': window.capture_label.text(),
                      'source': {'plane': window.source.plane, 'color': window.source.color},
                      'fixed_elements': ['DM', 'red_reference_path'],
                      'target': window.overlays.get('alignment', [])})

    def action(window, identity):
        def find():
            actions = window.plugin_states.get('alignment', {}).get('actions', [])
            label = next((a['label'] for a in actions if a['id'] == identity), None)
            for i in range(window.action_layout.count()):
                widget = window.action_layout.itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.text() == label:
                    return widget
            return None
        wait(window, lambda: find() is not None)
        find().click()

    def select(window, plane, color):
        frame_id = window.source.frame_id
        window.demo_plane.setCurrentIndex(0 if plane == 'near' else 1)
        window.demo_color.setCurrentIndex(0 if color == 'red' else 1)
        wait(window, lambda: window.last_frame.metadata.get('plane') == plane
             and window.last_frame.metadata.get('color') == color
             and window.last_fit is not None and window.last_fit.valid
             and window.last_fit.frame_id > frame_id and window.exposure.state.stable)
        settle(window)

    def record(window, progress_shot=False):
        count = len(measurements)
        action(window, 'record')
        if progress_shot:
            wait(window, lambda: window.capture is not None and len(window.capture.fits) >= 5)
            shot(window, '记录近处红光：累计20个有效帧',
                 '点击“记录：近处红光（遮蓝光）”，观察正在记录的帧数。',
                 'CCD和红光保持不动，蓝光继续遮挡，等待记录完成。',
                 '只累计新的、曝光稳定且定位可靠的帧；不能把重复刷新当成新测量。')
        wait(window, lambda: len(measurements) > count and window.capture is None)
        wait(window, lambda: '等待曝光稳定并记录20个有效帧' not in window.guidance_text.text())
        assert measurements[-1]['sample_count'] == 20

    def move_to_target(window, mirror):
        geometry = window.source.geometry
        window.demo_mirror.setCurrentIndex(0 if mirror == 'near' else 1)
        label = 'M2中间目标' if mirror == 'near' else '红光参考中心'
        wait(window, lambda: any(s.get('label') == label for s in window.overlays.get('alignment', [])))
        target = next(s for s in window.overlays['alignment'] if s.get('label') == label)
        buttons = {b.text(): b for b in window.demo_group.findChildren(QPushButton)}
        for _ in range(3):
            delta = np.array([target['x'], target['y']])-window.last_fit.center
            distance = window.source.geometry.mirror_near if mirror == 'near' else window.source.geometry.mirror_far
            pixel_step = (window.source.geometry.z_far+distance)*.0003
            clicks = np.rint(delta/pixel_step).astype(int)
            for axis, count in enumerate(clicks):
                if count:
                    arrow = ('→' if count > 0 else '←') if axis == 0 else ('↓' if count > 0 else '↑')
                    for _ in range(abs(int(count))):
                        buttons[arrow].click()
            moves.append({'mirror': mirror, 'clicks_xy': clicks.tolist(), 'target': [target['x'], target['y']]})
            settle(window, .6)
            if np.linalg.norm(window.last_fit.center-[target['x'], target['y']]) <= target['radius']:
                break
        wait(window, lambda: ('M2近反射镜目标已到达' if mirror == 'near' else 'M1远反射镜目标已到达') in window.guidance_text.text())
        assert window.source.geometry is geometry

    with tempfile.TemporaryDirectory(prefix='beamic-alignment-guide-') as directory:
        window = Window(directory, demo=True)
        window.resize(1360, 980)
        window.show()
        try:
            wait(window, lambda: window.capture_ready and window.last_fit and window.last_fit.valid and window.exposure.state.stable)
            shot(window, '启动模拟光路，打开合束引导',
                 '启动模拟光路，打开“合束引导”标签。',
                 '实验电脑连接原版BeamMic，完成设备预检后打开“合束引导”。',
                 '顶部明确显示“模拟光路·合成数据”；实际操作中的伪彩色不能辨别红蓝光。')
            shot(window, '输入四个沿光路累计的距离',
                 '本示例使用：DM→近测量100 mm、DM→远测量900 mm、近镜→DM 120 mm、远镜→DM 400 mm。',
                 'DM位置和角度保持固定。将四个输入框替换为实测距离；M1、M2镜距沿蓝光实际传播路径累计到DM，测量距离到CCD感光面。',
                 '近测量<远测量，近镜距<远镜距。本例两镜间距是400−120=280 mm。')
            next(b for b in window.findChildren(QPushButton) if b.text() == '开始双镜合束（DM固定）').click()
            wait(window, lambda: '先遮蓝光' in window.guidance_text.text())
            worker = window.plugins['alignment']
            original_send = worker.send
            def observe(message):
                if message.get('type') == 'measurement':
                    measurements.append(dict(message['measurement']))
                original_send(message)
            worker.send = observe
            select(window, 'near', 'red')
            shot(window, '近处：先记录红光参考',
                 '模拟操作选择“相机在近处”和“显示红光（遮蓝）”。',
                 '将CCD放在近测量面，遮住蓝光，仅让红光进入CCD。',
                 '默认保持“曝光稳定”开启、使用未饱和测量模式；中心和外圈应可靠。')
            record(window, progress_shot=True)
            select(window, 'near', 'blue')
            shot(window, '近处：保持CCD不动，切换蓝光',
                 '选择“显示蓝光（遮红）”，然后点击记录近处蓝光。',
                 'CCD留在原位，遮红光、放开蓝光；只切换遮光，不移动CCD。',
                 '等待蓝光对应的曝光稳定后记录；近处红光参考已经保留。')
            record(window)
            select(window, 'far', 'red')
            shot(window, '远处：记录红光参考',
                 '选择“相机在远处”和“显示红光（遮蓝）”，点击记录远处红光。',
                 '移动同一台CCD到远测量面，保持方向、倍率及采集设置；遮蓝光，只看红光。',
                 '近处红蓝两项已经记录，当前距离与远处输入一致。')
            record(window)
            select(window, 'far', 'blue')
            shot(window, '远处：保持CCD不动，记录蓝光',
                 '选择“显示蓝光（遮红）”，点击记录远处蓝光。',
                 'CCD留在远处原位，遮红光、放开蓝光并记录。',
                 '四次记录完成后，软件计算中间目标；此时还没有完成合束。')
            record(window)
            wait(window, lambda: any(s.get('label') == 'M2中间目标' for s in window.overlays.get('alignment', [])))
            shot(window, '识别蓝光M2的中间目标',
                 '查看图中的黄色“M2中间目标”，模拟调镜选择“调M2近反射镜”。',
                 '相机继续留在远处，红光保持遮挡；只调蓝光M2近反射镜。DM和M1保持不动。',
                 '第一步的目标是计算出的中间位置；第二步才移动到红光参考中心。')
            window.demo_mirror.setCurrentIndex(0)
            shot(window, '只调M2近反射镜（蓝光）',
                 '使用模拟方向按钮，将当前蓝光中心移向黄色中间目标。',
                 '手动调节蓝光M2的两个角度，使蓝光中心进入目标圈；DM和蓝光M1都保持不动。',
                 '目标圈表示允许偏差；画面中的光斑外圈和目标圈用途不同。')
            move_to_target(window, 'near')
            shot(window, 'M2近反射镜目标到达，确认进入第二步',
                 '看到“M2近反射镜目标已到达”后，点击“M2已到目标，进入M1调节”。',
                 '确认蓝光中心稳定进入中间目标后停止调M2近反射镜，并确认下一步。',
                 '程序需要连续有效帧满足目标才显示确认按钮。CCD仍在远处，红光仍遮挡。')
            action(window, 'confirm_near')
            wait(window, lambda: any(s.get('label') == '红光参考中心' for s in window.overlays.get('alignment', [])))
            window.demo_mirror.setCurrentIndex(1)
            shot(window, '只调M1远反射镜（蓝光），目标切换为红光中心',
                 '模拟选择“调M1远反射镜”，使用方向按钮靠近黄色“红光参考中心”。',
                 'DM和蓝光M2保持不动，只调蓝光M1的两个角度，使蓝光中心到达之前记录的远处红光中心。',
                 '无需同时打开两束光；黄色标记来自已保存的红光参考。相机始终不动。')
            move_to_target(window, 'far')
            shot(window, 'M1远反射镜目标到达，准备两处复测',
                 '看到“M1远反射镜目标已到达”后，点击“M1已到目标，开始近远复测”。',
                 '停止调镜，确认进入复测；随后在近、远处分别重新记录红光和蓝光。',
                 '远处到达目标只是调镜步骤完成，还需要两处实测验收。')
            action(window, 'confirm_far')
            wait(window, lambda: '移回近处' in window.guidance_text.text())
            select(window, 'near', 'red')
            shot(window, '复测近处红光', '模拟切回“相机在近处”和红光，点击记录。',
                 '将CCD移回近测量面，遮蓝光，重新记录红光。',
                 '使用新红光参考验收；两面镜子在整个复测期间保持不动。')
            record(window)
            select(window, 'near', 'blue')
            shot(window, '复测近处蓝光', '保持近处，切换蓝光并点击记录。',
                 'CCD在近处保持不动，遮红光、放开蓝光并记录。',
                 '近处红蓝中心差应不超过当地红光所选边界短半轴的5%。')
            record(window)
            select(window, 'far', 'red')
            shot(window, '复测远处红光', '模拟切回远处和红光，点击记录。',
                 '将CCD移到远测量面，遮蓝光，重新记录红光。',
                 '方向、倍率和定位定义保持一致；远处有独立的红光尺寸和容差。')
            record(window)
            select(window, 'far', 'blue')
            shot(window, '复测远处蓝光', '保持远处，切换蓝光并点击记录。',
                 'CCD不动，遮红光、放开蓝光并记录。',
                 '完成第八次记录后，软件同时检查近处和远处。')
            record(window)
            wait(window, lambda: '合束通过' in window.guidance_text.text())
            result = verify({m['role']: Measurement(**m) for m in measurements[4:8]})
            assert result['passed']
            shot(window, '查看近、远两处验收结果', '阅读“合束通过”以及近处、远处误差和容差。',
                 '核对两处均通过；若复测未通过，DM仍保持固定，用新测量计算下一轮，再按M2、M1的顺序调整蓝光反射镜。',
                 '当前成功是模拟结果；判据只验证轴线位置与方向，不代表束腰、偏振或耦合效率已经匹配。')

            # Deliberately ill-conditioned mirror spacing: real geometry inputs,
            # valid images and four real captures; no injected plugin messages.
            window.geometry_inputs['mirror_near'].setValue(399.95)
            window.geometry_inputs['mirror_far'].setValue(400)
            shot(window, '异常演示：镜间距过小',
                 '额外故障测试把近镜距改成399.95 mm、远镜距400 mm，并开始新一轮。',
                 '正常实验保持真实镜距；本设置仅用于演示计算目标超出可调范围的提示。',
                 '这不是推荐光路；镜间距过小会使中间目标对偏差非常敏感。')
            next(b for b in window.findChildren(QPushButton) if b.text() == '开始双镜合束（DM固定）').click()
            wait(window, lambda: '先遮蓝光' in window.guidance_text.text())
            for plane, color in [('near', 'red'), ('near', 'blue'), ('far', 'red'), ('far', 'blue')]:
                select(window, plane, color)
                record(window)
            wait(window, lambda: '超出画面' in window.guidance_text.text() or '太靠近画面边缘' in window.guidance_text.text())
            outside = window.guidance_text.text()
            shot(window, '异常提示：目标超出画面', '查看“中间目标超出画面，请先粗调光路再重新测量”。',
                 '核对距离和镜子顺序，必要时先粗调；若改变相机或光路，再重新完成四次初测。',
                 '软件暂停这一轮，不会把超出视野的目标视为已到达，也不会给出合束通过。')
        finally:
            window.close()
            app.processEvents()

    report = {'date': '2026-10-15', 'version': __version__, 'simulated': True,
              'hardware_acceptance': 'pending', 'fixed_dm': True,
              'adjustable_elements': ['M2_blue_near_mirror', 'M1_blue_far_mirror'],
              'steps': steps, 'step_count': len(steps),
              'measurements': measurements, 'mirror_moves': moves, 'verification': result,
              'outside_example': outside, 'actions_used': 'Actual Qt controls and actual plugin process; no physical hardware.'}
    (ROOT/'docs/alignment-workflow-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_guide(steps, result)
    print(json.dumps({'steps': len(steps), 'records': len(measurements), 'samples_per_record': 20,
                      'verification': result, 'outside_example': outside}, ensure_ascii=False, indent=2))


def write_guide(steps, result):
    intro = ('本说明使用beamic pro实际程序窗口、真实插件进程和合成数值帧，逐步截图。'
             '模拟菜单代替人工移动CCD、遮光和调镜，不控制实验硬件；所有截图均为模拟数据，实机验收待进行。'
             '普通流程为第1–17步，最后两步为额外异常演示。')
    conditions = ('蓝光光路：M1远反射镜 → M2近反射镜 → DM（固定） → CCD。红光通过既有固定光路到达DM后，与蓝光共用出射光路。'
                  'DM的位置和角度全程保持固定，红光光路保持固定；只调蓝光M1、M2两面反射镜，先M2、再M1。'
                  'M1、M2是普通转向反射镜，DM是独立的二向色合束镜。仅支持该模型下的平面镜与自由传播。'
                  '使用同一台CCD在两个测量面之间移动，保持方向、倍率、采集格式和定位定义一致。'
                  '每个位置切换红蓝光时CCD保持不动；两次调镜时CCD都留在远处。'
                  '红光保持为参考，只调整蓝光路径上的镜子。伪彩色不能辨认红蓝光身份。')
    geometry = '<table><tr><th>输入项</th><th>本例</th><th>含义</th></tr>'
    geometry += '<tr><td>DM → 近处测量</td><td>100 mm</td><td>DM到近处感光面</td></tr>'
    geometry += '<tr><td>DM → 远处测量</td><td>900 mm</td><td>DM到远处感光面</td></tr>'
    geometry += '<tr><td>M2近反射镜 → DM</td><td>120 mm</td><td>沿蓝光传播路径累计；DM固定</td></tr>'
    geometry += '<tr><td>M1远反射镜 → DM</td><td>400 mm</td><td>包含M1→M2及M2→DM两段；DM固定</td></tr></table>'
    markdown = '# 双镜合束：一步一图\n\n'+intro+'\n\n'+conditions+'\n\n'
    markdown += '| 输入项 | 本例 |\n|---|---:|\n| DM→近处感光面 | 100 mm |\n| DM→远处感光面 | 900 mm |\n| M2近反射镜→DM | 120 mm |\n| M1远反射镜→DM | 400 mm |\n\n'
    sections = []
    for step in steps:
        title = f"第{step['number']}步：{step['title']}"
        markdown += f"## {title}\n\n模拟操作：{step['demo_action']}\n\n实机对应操作：{step['real_action']}\n\n检查：{step['expected']}\n\n![{title}]({step['image']})\n\n"
        data = base64.b64encode((ROOT/'docs'/step['image']).read_bytes()).decode()
        sections.append(f'<section id="step-{step["number"]}"><h2>{html.escape(title)}</h2>'
                        f'<p><b>模拟操作：</b>{html.escape(step["demo_action"])}</p>'
                        f'<p><b>实机对应操作：</b>{html.escape(step["real_action"])}</p>'
                        f'<p><b>检查：</b>{html.escape(step["expected"])}</p>'
                        f'<img alt="{html.escape(title)}，实际程序截图、模拟数据" src="data:image/png;base64,{data}"></section>')
    notes = ('本次正常流程完成八次记录，每次20个有效帧；两面镜子的调节通过实际模拟方向按钮完成。'
             f'近处中心差{result["near"]["error_px"]:.3f} px，容差{result["near"]["tolerance_px"]:.3f} px；'
             f'远处中心差{result["far"]["error_px"]:.3f} px，容差{result["far"]["tolerance_px"]:.3f} px。'
             '容差为各处红光所选边界拟合短半轴的5%，不是束腰插件的二阶矩精度门槛。'
             '目标超出画面的额外示例另执行四次记录。全部数据只证明软件模拟流程，不能证明实际合束精度。')
    camera = ('SP932U量距应以感光面为准。若从相机本体前端平面读数，感光面位于其后4.5 mm，'
              '前端平面不包含滤片最前缘。该偏移不能替代每次摆放距离的实测。'
              '曝光稳定默认开启；关闭仅停止自动调节，仍需有效曝光和可靠图像。'
              '固定圈只提供显示参考；本教程不使用固定圈替代记录的红光参考和最终复测。')
    markdown += '## 本次模拟检查结果\n\n'+notes+'\n\n'+camera+'\n'
    (ROOT/'docs/合束操作说明.md').write_text(markdown, encoding='utf-8')
    links = ''.join(f'<a href="#step-{s["number"]}">{s["number"]}. {html.escape(s["title"])}</a>' for s in steps)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双镜合束：一步一图</title><style>'
    page += 'body{max-width:1280px;margin:30px auto;padding:0 24px;font:16px/1.75 -apple-system,"Microsoft YaHei",sans-serif;color:#172b40;background:#f4f7fa}section{background:white;padding:22px;margin:24px 0;border-radius:12px}img{width:100%;height:auto;border:1px solid #cbd7e2}nav a{display:block}h1,h2{line-height:1.4}.note{background:#fff0cd;padding:15px;border-radius:8px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cbd7e2;padding:8px;text-align:left}</style>'
    page += '<h1>双镜合束：一步一图</h1><p class="note">'+html.escape(intro)+'</p><p>'+html.escape(conditions)+'</p>'+geometry+'<nav>'+links+'</nav>'+''.join(sections)
    page += '<section><h2>本次模拟检查结果</h2><p>'+html.escape(notes)+'</p><p>'+html.escape(camera)+'</p></section></html>'
    (ROOT/'docs/合束操作说明.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
