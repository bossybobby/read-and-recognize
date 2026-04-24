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

# 屏幕手写数字识别器 (Screen Digit Recognizer)

本项目提供了一种简单高效的方法来识别屏幕上显示的手写数字。它通过截取指定区域的屏幕图像，将其转换为8x8的强度矩阵，并利用在MNIST数据集上训练的卷积神经网络(CNN)模型进行实时预测。

## 核心功能

- **屏幕区域选择**：用户可以自由框选包含手写数字的屏幕区域。
- **实时热力图显示**：将截取的图像转换为8x8的热力图，直观展示识别输入。
- **实时数字预测**：基于预训练的CNN模型，实时显示预测结果及其置信度。

## 安装

1. **克隆或下载本项目**。
2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 运行主程序：
   ```bash
   python screen_capture_app.py
   ```
2. 在打开的应用窗口中，点击 **“选择区域”** 按钮。
3. 在屏幕上拖拽以框选包含手写数字的区域（例如，iStaricMDT软件的热力图区域，或任何显示黑白数字的窗口）。
4. 应用将自动开始处理所选区域，并在右侧显示8x8热力图及预测结果。

## 技术细节

- **模型**：使用自定义的`SimpleCNN`架构，在标准MNIST数据集上训练。首次运行时会自动下载数据集并训练模型，之后会加载已保存的权重文件(`mnist_cnn.pth`)。
- **输入处理**：截屏区域会被转换为灰度图，并缩放/归一化为8x8的浮点数矩阵作为模型输入。
- **可视化**：使用`PyQtGraph`进行高效的热力图渲染。

## 依赖库

- `opencv-python`
- `numpy`
- `pyqtgraph`
- `mss`
- `torch`
- `torchvision`
