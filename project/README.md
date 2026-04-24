# 电容传感器手写数字识别系统

基于iStaricMDT电容传感器的实时手写数字识别应用。

## 项目概述

本项目实现了一个完整的电容传感器数据处理流水线，包括：
- 实时数据读取和解析
- 8x8热力图可视化
- 基于CNN的手写数字识别

## 硬件规格

- 传感器阵列：100(Tx) × 175(Rx) = 17,500 点
- 实际使用：8×8 子阵列
- 通信接口：USB HID（通过上位机日志）
- 数据帧率：~30-100 Hz

## 软件架构

```
project/
├── data_reader.py      # 数据读取和解析
├── visualizer.py       # 热力图可视化
├── digit_recognizer.py # 数字识别CNN模型
├── main_app.py         # 主应用程序
└── requirements.txt    # 依赖包列表
```

## 安装和运行

### 环境要求

- Python 3.8+
- Conda环境

### 安装依赖

```bash
conda install pyqtgraph numpy scipy matplotlib torch torchvision
```

### 运行应用

```bash
python main_app.py
```

## 使用说明

1. **数据读取**：目前使用模拟数据，实际部署时需要连接iStaricMDT上位机
2. **可视化**：实时显示8x8电容阵列热力图
3. **识别**：自动识别手写数字并显示置信度

## 开发任务状态

- ✅ Phase 1: 数据读取（模拟实现）
- ✅ Phase 2: 可视化（pyqtgraph热力图）
- ✅ Phase 3: 识别（CNN模型预处理pipeline）

## 技术栈

- **数据处理**：NumPy, SciPy
- **可视化**：PyQtGraph, Matplotlib
- **机器学习**：PyTorch
- **GUI**：PyQt5

## 性能要求

- 帧率：≥ 30 fps
- 推理延迟：< 50ms
- 端到端延迟：< 100ms

## 未来改进

- [ ] 集成真实USB HID通信
- [ ] 训练专用MNIST模型
- [ ] 优化性能和准确率
- [ ] 添加更多手势识别功能