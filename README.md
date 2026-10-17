# beamic pro

通过 Ophir BeamMic 连接相机的中文光学实验助手，当前版本 **0.5.0**。

## Windows 下载与运行

从 [Releases](https://github.com/liweihan389/beam_pro/releases) 下载：

- `beamic-pro-0.5.0-setup.exe`：安装到当前用户目录，并创建开始菜单入口。
- `beamic-pro-0.5.0-windows-portable.zip`：免安装版，解压后运行 `beamic-pro.exe`。
- `SHA256SUMS.txt`：下载文件的SHA256校验值。

使用 Windows x64；.NET Framework 4.8 是桥接程序的运行要求。真实相机连接需要已授权的原版 BeamMic 和相机驱动。第一次建立独立的共享插件环境需要联网安装依赖。

可以从开始菜单选择“模拟光路”，或运行 `beamic-pro.exe --demo`，在没有相机时体验功能。模拟结果不能代替真实设备或光学实验验收。

## 功能

- 实时数值图像、几何中心与相对强度边界圆／椭圆定位，支持主光斑选区。
- 自动曝光“曝光稳定”默认开启；关闭后保持当前曝光，仍排除旧曝光及无效帧。
- 自动定圈默认开启，可固定参考圈、标记当前中心并连接两个中心。
- 固定DM、调节蓝光两面反射镜的双镜合束引导。
- 使用独立D4σ宽度的束腰测量：三点建模、两个新点验证，允许补测。
- 单镜等高默认开启，可选coupler手动调焦及区间准直验证；组合模式先准直、再等高。
- Python插件安装、启停和独立进程运行，包含三个实验引导插件及基础分析示例。

目标为20个新的有效帧/秒，支持设置与自动降档；每次正式位置记录默认20个有效新帧。真实设备吞吐和端到端延迟尚需实测。

公开版本仅包含源码及程序，不包含本地操作教程截图、图文测试报告、原始数据集或实验记录。图像与测量会话默认只在内存中；当前版本尚无完整的实测记录包导出功能。

## 源码运行

需要CPython3.12 x64：

```sh
python -m venv .venv
# Windows使用 .venv\Scripts\python.exe；macOS/Linux使用 .venv/bin/python。
python -m pip install -r requirements-core.lock -e sdk -e ".[dev]"
python -m beamic_pro.app --demo
```

上述安装和启动应使用虚拟环境中的Python，或先激活虚拟环境。

## 测试、打包与插件

```sh
python -m pytest -q
python scripts/package_plugins.py
```

在Windows上运行 `scripts/build_windows.ps1`，构建程序、桥接、插件、便携ZIP及Inno Setup安装包。需要.NET SDK与Inno Setup 6；`-SkipInstaller`仅构建便携版。默认排除本地图文资料。

GitHub Actions的Release流程在Windows上测试、构建并发布安装包，创建版本标签和校验文件；真实BeamMic／SP932U接口、光高精度与实验表现仍需设备预检和独立参考核查。

插件实现位于 `plugins/alignment`、`plugins/waist`、`plugins/level` 和 `plugins/basic_analysis`；SDK位于 `sdk`，当前版本2.1.0、协议版本2。安装ZIP到“插件库”，引导任务一次运行一个，分析插件可并行读取最新数值帧。
