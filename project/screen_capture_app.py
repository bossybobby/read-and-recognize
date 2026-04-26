import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from screen_capture_reader import CaptureRegion, ScreenCaptureReader
from digit_recognizer import DigitRecognizer  # 假设这是你的识别器类


class RegionSelector(QtWidgets.QWidget):
    region_selected = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.start_pos: Optional[QtCore.QPoint] = None
        self.end_pos: Optional[QtCore.QPoint] = None

        desktop = QtWidgets.QApplication.desktop()
        geometry = desktop.screenGeometry(desktop.primaryScreen())
        self.setGeometry(geometry)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.start_pos = event.globalPos()
            self.end_pos = event.globalPos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_pos is not None:
            self.end_pos = event.globalPos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.start_pos and self.end_pos:
            rect = QtCore.QRect(self.start_pos, self.end_pos).normalized()
            if rect.width() > 8 and rect.height() > 8:
                self.region_selected.emit(
                    CaptureRegion(rect.left(), rect.top(), rect.width(), rect.height())
                )
            self.close()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 90))

        if self.start_pos is None or self.end_pos is None:
            return

        local_start = self.mapFromGlobal(self.start_pos)
        local_end = self.mapFromGlobal(self.end_pos)
        rect = QtCore.QRect(local_start, local_end).normalized()
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 2))
        painter.setBrush(QtGui.QColor(0, 220, 255, 35))
        painter.drawRect(rect)


class ScreenCaptureApp(QtWidgets.QMainWindow):
    GRID_SHAPES = {
        "8x8": (8, 8),
        "32x32": (32, 32),
        "100x100": (100, 100),
    }
    FPS_LIMITS = {
        "10 FPS": 100,
        "30 FPS": 33,
        "60 FPS": 16,
        "无限": 0,
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Capture Bridge")
        self.resize(1100, 720)

        self.grid_shape = (8, 8)
        self.reader = ScreenCaptureReader(grid_shape=self.grid_shape, value_mode="gray")
        self.region: Optional[CaptureRegion] = None
        self.latest_preview: Optional[np.ndarray] = None
        self.frame_count = 0
        self.last_time = QtCore.QTime.currentTime()

        # 初始化数字识别器
        self.recognizer = DigitRecognizer(grid_shape=self.grid_shape, auto_train=False)

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.FPS_LIMITS["30 FPS"])

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.btn_select = QtWidgets.QPushButton("选择区域")
        self.btn_pause = QtWidgets.QPushButton("暂停")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["gray", "saturation", "value", "color-diff"])
        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.addItems(self.GRID_SHAPES.keys())
        self.fps_combo = QtWidgets.QComboBox()
        self.fps_combo.addItems(self.FPS_LIMITS.keys())
        self.fps_combo.setCurrentText("30 FPS")
        self.region_label = QtWidgets.QLabel("区域: --")
        self.fps_label = QtWidgets.QLabel("FPS: --")
        self.frame_label = QtWidgets.QLabel("帧: 0")
        # 新增识别结果显示标签
        self.prediction_label = QtWidgets.QLabel("预测: --")

        controls.addWidget(self.btn_select)
        controls.addWidget(self.btn_pause)
        controls.addWidget(QtWidgets.QLabel("通道:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(QtWidgets.QLabel("阵列:"))
        controls.addWidget(self.grid_combo)
        controls.addWidget(QtWidgets.QLabel("最大帧率:"))
        controls.addWidget(self.fps_combo)
        controls.addStretch(1)
        controls.addWidget(self.region_label)
        controls.addWidget(self.fps_label)
        controls.addWidget(self.frame_label)
        controls.addWidget(self.prediction_label)  # 添加到控制栏
        layout.addLayout(controls)

        content = QtWidgets.QHBoxLayout()
        layout.addLayout(content, stretch=1)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumSize(360, 240)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #111; border: 1px solid #333;")
        content.addWidget(self.preview_label, stretch=1)

        self.plot = pg.GraphicsLayoutWidget()
        self.heatmap_plot = self.plot.addPlot(title="8x8")
        self.heatmap_plot.setAspectLocked(True)
        self.heatmap_img = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_img)
        self.colormap = pg.colormap.get("plasma")
        self.heatmap_img.setLookupTable(self.colormap.getLookupTable())
        self.colorbar = pg.ColorBarItem(values=(0, 1000), colorMap=self.colormap)
        self.colorbar.setImageItem(self.heatmap_img)
        self.plot.addItem(self.colorbar, row=0, col=1)
        content.addWidget(self.plot, stretch=1)

        self.btn_select.clicked.connect(self.select_region)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.mode_combo.currentTextChanged.connect(self.change_value_mode)
        self.grid_combo.currentTextChanged.connect(self.change_grid_shape)
        self.fps_combo.currentTextChanged.connect(self.change_max_fps)

    def select_region(self):
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self.set_region)
        self.selector.show()

    def set_region(self, region: CaptureRegion):
        self.region = region.normalized()
        self.region_label.setText(
            f"区域: {self.region.left},{self.region.top} {self.region.width}x{self.region.height}"
        )

    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("继续")
        else:
            self.last_time = QtCore.QTime.currentTime()
            self.timer.start(self.FPS_LIMITS[self.fps_combo.currentText()])
            self.btn_pause.setText("暂停")

    def change_value_mode(self, mode: str):
        self.reader.value_mode = mode

    def change_grid_shape(self, grid_name: str):
        self.grid_shape = self.GRID_SHAPES[grid_name]
        self.reader.grid_shape = self.grid_shape
        self.recognizer.set_grid_shape(self.grid_shape, auto_train=False)
        self.heatmap_plot.setTitle(grid_name)
        self.heatmap_img.clear()
        if not self.recognizer.is_ready:
            self.prediction_label.setText(f"预测: 缺少 {grid_name} 模型")
        else:
            self.prediction_label.setText("预测: --")

    def change_max_fps(self, fps_name: str):
        if self.timer.isActive():
            self.last_time = QtCore.QTime.currentTime()
            self.timer.start(self.FPS_LIMITS[fps_name])

    def update_frame(self):
        if self.region is None:
            return

        frame = self.reader.capture(self.region)
        if frame is None:
            return

        grid = self.reader.frame_to_grid(frame)
        # --- 修复1: 翻转热力图 ---
        flipped_grid = np.flipud(grid)
        self.heatmap_img.setImage(flipped_grid.T, levels=(0, 1000))
        # --- 修复1结束 ---

        self.update_preview(frame)

        if not self.recognizer.is_ready:
            rows, cols = grid.shape
            self.prediction_label.setText(f"预测: 缺少 {cols}x{rows} 模型")
        elif grid.shape == self.recognizer.grid_shape:
            try:
                input_array = np.array(grid, dtype=np.float32)
                prediction, confidence = self.recognizer.predict(input_array)
                debug_path = Path(__file__).resolve().parent / "debug_last_preprocessed.png"
                self.recognizer.save_last_preprocessed(debug_path)
                self.prediction_label.setText(f"预测: {prediction} (置信度: {confidence:.2f})")
            except Exception as e:
                self.prediction_label.setText(f"预测错误: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            rows, cols = grid.shape
            self.prediction_label.setText(f"预测: 模型尺寸不匹配 ({cols}x{rows})")

        self.frame_count += 1
        self.frame_label.setText(f"帧: {self.frame_count}")
        current_time = QtCore.QTime.currentTime()
        dt = self.last_time.msecsTo(current_time)
        if dt > 0:
            self.fps_label.setText(f"FPS: {1000.0 / dt:.1f}")
        self.last_time = current_time

    def update_preview(self, frame: np.ndarray):
        h, w, _ = frame.shape
        target_w = max(1, self.preview_label.width())
        target_h = max(1, self.preview_label.height())
        scale = min(target_w / w, target_h / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        qimage = QtGui.QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            preview.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimage.copy()))


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ScreenCaptureApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

