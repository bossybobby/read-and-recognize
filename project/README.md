# Screen Capture Digit Recognizer

一个用于屏幕手写数字识别的小工具。它可以框选屏幕上的数字区域，把截图实时转换成阵列热力图，并用 MNIST 训练的小型 CNN 模型输出数字预测。

项目最初面向 iStaricMDT 电容阵列热力图调试，也可以直接识别浏览器、画图工具或其他窗口里的黑白手写数字。

## 当前功能

- 屏幕区域框选：点击“选择区域”，拖拽出要识别的区域。
- 实时预览：左侧显示原始截图缩放预览。
- 热力图显示：右侧显示采样后的阵列数据。
- 阵列切换：支持 `8x8`、`32x32`、`100x100`。
- 最大帧率切换：支持 `10 FPS`、`30 FPS`、`60 FPS`、`无限`。
- 通道切换：支持 `gray`、`saturation`、`value`。
- CNN 识别：使用 `cnn_mnist_model_v2.pth`，输入统一规范成 MNIST 风格的 `28x28`。
- 调试输出：每次预测会保存模型实际看到的图到 `debug_last_preprocessed.png`。

## 文件结构

```text
project/
├── screen_capture_app.py      # PyQt 主界面
├── screen_capture_reader.py   # 屏幕捕获和阵列采样
├── digit_recognizer.py        # CNN 模型、训练和预测预处理
├── cnn_mnist_model_v2.pth     # 当前使用的 CNN 权重
├── requirements.txt           # Python 依赖
└── README.md                  # 本说明
```

仓库根目录还保留：

- `istaricMDT-V20241130/`：iStaricMDT 上位机程序。
- `IS5816_v1.0_ch.pdf`：芯片/硬件资料。
- `ISTAA1_444 板卡容值数据录制说明.pdf`：板卡数据录制说明。

## 安装依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果你的环境里没有可用的 Qt 后端，请确认 `PyQt5` 安装成功。

## 运行

在仓库根目录执行：

```bash
python project/screen_capture_app.py
```

也可以进入 `project` 后运行：

```bash
python screen_capture_app.py
```

## 使用建议

1. 打开要识别的手写数字页面或 iStaricMDT 热力图页面。
2. 点击“选择区域”，只框住数字主体，尽量不要包含窗口边框、坐标轴、色条或多余文字。
3. 黑白数字优先使用 `gray` 通道。
4. 彩色热力图可以尝试 `saturation` 或 `value` 通道。
5. 阵列优先用 `32x32` 或 `100x100`，`8x8` 信息量太少，笔画细节容易丢。
6. 如果预测结果异常，查看 `project/debug_last_preprocessed.png`：
   - 应该是黑底白字、居中的 `28x28` 数字。
   - 如果这张图不像目标数字，问题在截图区域或预处理。
   - 如果这张图很像目标数字但预测错，问题才更可能在模型。

## 识别原理

实时流程如下：

```text
屏幕截图
-> RGB/HSV/灰度通道提取
-> resize 成 8x8 / 32x32 / 100x100 阵列
-> 自动转成黑底亮字
-> Otsu 阈值提取数字主体
-> 裁剪、缩放、质心居中到 28x28
-> CNN 输出数字和置信度
```

界面里的阵列尺寸用于显示和采样精度；识别模型最终固定吃 MNIST 标准的 `28x28`，这样比直接训练 `100x100` 线性模型稳定得多。

## 重新训练模型

默认模型文件已经放在仓库里。如果需要重新训练：

```bash
python project/digit_recognizer.py --sizes cnn --force
```

训练数据通过 `sklearn.datasets.fetch_openml("mnist_784")` 获取。首次训练需要网络和一些时间。

## 常见问题

**为什么浏览器里很清楚的数字也会识别错？**

通常是模型实际看到的输入和你肉眼看到的不一样。优先检查 `debug_last_preprocessed.png`。如果截图区域包含太多背景、边框或颜色条，预处理后的 28x28 图就会变形。

**彩色热力图怎么识别？**

先试 `saturation` 或 `value` 通道。如果背景和笔画亮度接近，后续可以加入专门的 `color-diff` 通道：用边缘颜色估计背景，再按颜色差异提取数字主体。

**为什么保留 iStaricMDT 和 PDF？**

它们是硬件调试和数据来源相关资料，不属于 Python 应用运行缓存，因此保留。
