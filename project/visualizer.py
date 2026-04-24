import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
from data_reader import CapacitiveDataReader

class CapacitiveVisualizer:
    def __init__(self):
        # 创建窗口
        self.win = pg.GraphicsLayoutWidget(show=True, title="电容传感器热力图")
        self.win.resize(800, 600)

        # 创建热力图
        self.plot = self.win.addPlot(title="8x8 电容阵列")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)

        # 设置颜色映射
        self.colormap = pg.colormap.get('viridis')
        self.img.setLookupTable(self.colormap.getLookupTable())

        # 添加颜色条
        self.colorbar = pg.ColorBarItem(values=(0, 1000), colorMap=self.colormap)
        self.colorbar.setImageItem(self.img)
        self.win.addItem(self.colorbar)

        # FPS显示
        self.fps_label = pg.LabelItem(justify='right')
        self.win.addItem(self.fps_label, row=1, col=0)

        # 帧号显示
        self.frame_label = pg.LabelItem(justify='left')
        self.win.addItem(self.frame_label, row=1, col=1)

        # 识别结果显示
        self.recognition_label = pg.LabelItem(justify='center')
        self.win.addItem(self.recognition_label, row=2, col=0, colspan=2)

        self.frame_count = 0
        self.last_time = QtCore.QTime.currentTime()

    def update_heatmap(self, data):
        """更新热力图数据"""
        if data is not None:
            # 转置数据以正确显示（如果需要）
            self.img.setImage(data.T)  # 或 data 根据方向调整

            # 更新帧号
            self.frame_count += 1
            self.frame_label.setText(f"帧号: {self.frame_count}")

            # 计算FPS
            current_time = QtCore.QTime.currentTime()
            dt = self.last_time.msecsTo(current_time)
            if dt > 0:
                fps = 1000.0 / dt
                self.fps_label.setText(".1f")
            self.last_time = current_time

    def update_recognition(self, result):
        """更新识别结果"""
        self.recognition_label.setText(f"识别结果: {result}")

def main():
    # 创建应用
    app = QtGui.QApplication(sys.argv)

    # 创建可视化器
    visualizer = CapacitiveVisualizer()

    # 创建数据读取器（模拟）
    reader = CapacitiveDataReader("dummy")

    # 定时器用于更新数据
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: visualizer.update_heatmap(reader.get_simulated_data()))
    timer.start(100)  # 100ms = 10 FPS

    # 启动应用
    if __name__ == '__main__':
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()