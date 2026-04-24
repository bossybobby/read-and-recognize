import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from data_reader import CapacitiveDataReader
from digit_recognizer import DigitRecognizer

class CapacitiveSensorApp:
    def __init__(self):
        # 创建主窗口
        self.win = pg.GraphicsLayoutWidget(show=True, title="电容传感器数字识别系统")
        self.win.resize(1200, 800)

        # 创建布局
        self.layout = pg.GraphicsLayout()
        self.win.setCentralItem(self.layout)

        # 热力图
        self.heatmap_plot = self.layout.addPlot(title="8x8 电容阵列热力图", row=0, col=0)
        self.heatmap_img = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_img)

        # 设置颜色映射
        self.colormap = pg.colormap.get('plasma')
        self.heatmap_img.setLookupTable(self.colormap.getLookupTable())

        # 颜色条
        self.colorbar = pg.ColorBarItem(values=(0, 1000), colorMap=self.colormap)
        self.colorbar.setImageItem(self.heatmap_img)
        self.layout.addItem(self.colorbar, row=0, col=1)

        # 状态信息
        self.info_layout = self.layout.addLayout(row=1, col=0, colspan=2)
        self.fps_label = pg.LabelItem("FPS: --", justify='left')
        self.frame_label = pg.LabelItem("帧号: --", justify='center')
        self.recognition_label = pg.LabelItem("识别结果: --", justify='right')

        self.info_layout.addItem(self.fps_label, row=0, col=0)
        self.info_layout.addItem(self.frame_label, row=0, col=1)
        self.info_layout.addItem(self.recognition_label, row=0, col=2)

        # 初始化组件
        self.data_reader = CapacitiveDataReader("dummy")
        self.recognizer = DigitRecognizer()

        # 定时器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # 10 FPS

        # 状态变量
        self.frame_count = 0
        self.last_time = QtCore.QTime.currentTime()

    def update_frame(self):
        """更新一帧数据"""
        # 获取数据
        data = self.data_reader.get_simulated_data()

        # 更新热力图
        self.heatmap_img.setImage(data.T)

        # 识别数字
        digit, confidence = self.recognizer.predict(data)

        # 更新状态
        self.frame_count += 1
        self.frame_label.setText(f"帧号: {self.frame_count}")
        self.recognition_label.setText(f"识别结果: {digit} (置信度: {confidence:.3f})")

        # 计算FPS
        current_time = QtCore.QTime.currentTime()
        dt = self.last_time.msecsTo(current_time)
        if dt > 0:
            fps = 1000.0 / dt
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.last_time = current_time

def main():
    app = QtWidgets.QApplication(sys.argv)
    sensor_app = CapacitiveSensorApp()

    if __name__ == '__main__':
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()
