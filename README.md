# 2026 电赛拼图装置上位机

面向拼图装置的计算机视觉与运动指令生成程序。项目从摄像头画面中定位 A4 工作区、分割拼图碎片、计算拼接方案，并通过 UART 向下位机发送每块碎片的吸取点、放置点和旋转角。

> 这是比赛期间持续迭代的工程代码，视觉阈值、标定数据和串口设备名都与实际机械结构、相机和光照有关。使用前请先完成自己的相机标定，并在机械臂断电或急停可用的条件下验证坐标方向。

## 功能

- A4 工作区定位与透视矫正
- 白纸、彩色纸和扑克牌碎片分割
- 碎片轮廓简化、边长/角度分析与吸取点计算
- 保留基于模板的拼图片识别与配准模块
- 基于 DFS 的几何拼接，支持复合缝与纹理重排序
- UART READY / 坐标 / ACK 通信
- 相机标定、模板采集、分割调参和实机联调工具

核心处理链路：

```text
摄像头帧
  -> 工作区定位与透视矫正
  -> 分界线与碎片分割
  -> 轮廓和几何特征提取
  -> 模板匹配 / DFS 几何拼接
  -> 吸取点、放置点、旋转角
  -> UART 下发至运动控制器
```

## 仓库结构

```text
.
├─ AIPro/                 # 比赛部署版：视觉、求解、UART 与主入口
│  ├─ main.py             # 生产入口
│  ├─ protocol.py         # 串口协议编解码
│  ├─ camera.py           # 相机封装
│  ├─ detection.py        # 部署版目标检测
│  ├─ solver.py           # 模板匹配求解
│  ├─ b_solver.py         # DFS 几何拼接
│  └─ calib_data/         # 部署时使用的标定与模板数据
└─ saiti/                 # 模块化算法开发与验证目录
   ├─ vision/             # 视觉/求解模块及算法说明
   ├─ scripts/            # 标定、调参、离线测试与实机测试工具
   └─ calib_data/         # 开发环境标定与模板数据
```

`AIPro/main.py` 会复用 `saiti` 中的部分检测逻辑，因此部署完整项目时请保留两个目录的相对位置。本仓库当前公开的是上位机部分；运动控制器、机械结构和 FPGA/下位机工程不包含在本目录中。

当前比赛入口对 A/B 路白纸与 C 路扑克牌统一使用 DFS 几何拼接；模板匹配代码与模板采集工具作为实验路线保留。

更详细的视觉算法见 [`saiti/vision/VISION_ALGO.md`](saiti/vision/VISION_ALGO.md)，开发脚本和已知问题见 [`saiti/scripts/PROGRESS.md`](saiti/scripts/PROGRESS.md)。

## 环境要求

- Python 3.10 或更高版本
- 支持 OpenCV 的 USB 摄像头
- UART 串口设备（运行生产入口时需要）
- 下位机需实现本文后述通信协议

安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

项目使用 `cv2.aruco` 完成标定，因此依赖完整版 `opencv-contrib-python`。如果环境中同时安装了其他 OpenCV wheel，建议先卸载以避免 `cv2` 冲突。

## 快速开始

### 1. 检查离线求解

该测试不需要摄像头或串口：

```bash
cd saiti
python scripts/test_b_route.py
```

### 2. 相机标定

在 `saiti` 目录运行：

```bash
python scripts/calibrate_aruco.py --camera 0
```

按脚本界面提示采集 ArUco 点位，结果写入 `saiti/calib_data/homography.json`。部署前请把确认后的标定结果同步到 `AIPro/calib_data/homography.json`。

### 3. 构建模板

```bash
python scripts/build_templates.py --camera 0
```

模板保存到 `saiti/calib_data/templates.json`。若使用模板路线，同样需要把结果同步到 `AIPro/calib_data/templates.json`。

### 4. 启动比赛程序

生产入口中的默认标定路径是相对于当前工作目录解析的，因此请从 `AIPro` 目录启动：

```bash
cd ../AIPro
python main.py --port /dev/ttyACM0 --camera 0
```

Windows 串口示例：

```powershell
python main.py --port COM3 --camera 0
```

常用参数：

```text
-m, --mode {A,B,C}  强制指定路线；默认由 MCU READY 帧决定
-p, --port PORT     串口设备
-b, --baudrate N    波特率，默认 115200
-c, --camera N      摄像头索引；配置值小于 0 时自动检测
--dry-run            完成视觉求解但不发送坐标（仍需连接 UART）
--simulate           使用内置坐标测试 UART，不打开摄像头
```

查看完整参数：

```bash
python main.py --help
```

## UART 协议

帧头为 `0xAA`，帧尾为 `0xBB`，多字节有符号数使用大端序；坐标和角度均放大 100 倍后以 `int16` 传输。

| 方向 | 帧 | 含义 |
|---|---|---|
| MCU -> 上位机 | `AA 00 mode BB` | READY；`mode` 为 `01` / `02` / `03` |
| 上位机 -> MCU | `AA 01 'X' xg 'Y' yg 'X' xp 'Y' yp yaw BB` | 17 字节坐标指令 |
| MCU -> 上位机 | `AA 02 BB` | 当前动作完成 ACK |

完整实现与调试示例位于 [`AIPro/protocol.py`](AIPro/protocol.py)。

## 标定与参数说明

- `homography.json` 保存像素坐标到纸面毫米坐标的单应矩阵，换相机、改变相机位置或工作平面后必须重新标定。
- `templates.json` 保存模板拼图的几何特征，改变拼图材料或模板布局后应重新采集。
- 视觉阈值、拼接容差、放置偏移、相机参数和串口默认值集中在 `AIPro/config.py` 与 `saiti/vision/config.py`。
- 运行时生成的调试图位于 `debug_output/` 或 `calib_data/debug_*.png`，已通过 `.gitignore` 排除。

不要直接把示例标定数据用于新的机械装置。建议先运行 `saiti/scripts` 中的实时诊断工具确认纸面、分界线和碎片轮廓，再连接运动机构。

## 开发与验证工具

| 命令（在 `saiti` 目录执行） | 用途 |
|---|---|
| `python scripts/capture.py` | 相机预览与抓帧 |
| `python scripts/live_dividing_line.py` | 实时检查工作区与分界线 |
| `python scripts/live_segmentation.py` | 实时检查碎片分割 |
| `python scripts/tune_white.py` | 白纸分割参数调优 |
| `python scripts/tune_green_bg.py` | 绿底检测参数调优 |
| `python scripts/test_b_live.py` | B 路实机端到端联调 |
| `python scripts/test_solver.py` | A/B 路综合实机测试 |

## 安全提示

本项目会向运动控制器发送真实坐标。首次运行、重新标定或修改坐标系后，请先禁用执行机构或使用低速单步模式，核对原点、轴方向、单位、旋转正方向和运动范围。作者不对设备损坏或人身伤害承担责任。

## License

本项目采用 [MIT License](LICENSE)。
