from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mss
except ImportError:  # pragma: no cover - fallback for minimal environments
    mss = None
    from PIL import ImageGrab


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def normalized(self) -> "CaptureRegion":
        left = self.left
        top = self.top
        width = self.width
        height = self.height
        if width < 0:
            left += width
            width = -width
        if height < 0:
            top += height
            height = -height
        return CaptureRegion(left, top, width, height)


class ScreenCaptureReader:
    DIFF_NOISE_FLOOR = 18

    def __init__(self, grid_shape: Tuple[int, int] = (8, 8), value_mode: str = "gray"):
        self.grid_shape = grid_shape
        self.value_mode = value_mode
        self._mss = mss.mss() if mss else None

    def capture(self, region: CaptureRegion) -> Optional[np.ndarray]:
        region = region.normalized()
        if region.width < 2 or region.height < 2:
            return None

        if self._mss:
            monitor = {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
            shot = np.asarray(self._mss.grab(monitor))
            return cv2.cvtColor(shot, cv2.COLOR_BGRA2RGB)

        box = (region.left, region.top, region.left + region.width, region.top + region.height)
        return np.asarray(ImageGrab.grab(box))

    def frame_to_grid(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros(self.grid_shape, dtype=np.int32)

        plane = self._signal_plane(frame, self.value_mode)

        plane = cv2.GaussianBlur(plane, (3, 3), 0)
        grid = cv2.resize(
            plane,
            (self.grid_shape[1], self.grid_shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        grid -= grid.min()
        max_value = grid.max()
        if max_value > 0:
            grid = grid / max_value
        return np.rint(grid * 1000).astype(np.int32)

    def frame_difference_to_grid(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        noise_floor: int | None = None,
    ) -> np.ndarray:
        if (
            frame is None
            or background is None
            or frame.size == 0
            or background.size == 0
        ):
            return np.zeros(self.grid_shape, dtype=np.int32)

        if frame.shape != background.shape:
            background = cv2.resize(
                background,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        current = self._signal_plane(frame, self.value_mode).astype(np.float32)
        base = self._signal_plane(background, self.value_mode).astype(np.float32)
        diff = np.abs(current - base)
        floor = self.DIFF_NOISE_FLOOR if noise_floor is None else int(noise_floor)
        diff = np.maximum(diff - floor, 0)

        if diff.max() > 0:
            diff = cv2.GaussianBlur(diff, (3, 3), 0)

        grid = cv2.resize(
            diff,
            (self.grid_shape[1], self.grid_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        return np.rint(np.clip(grid / 255.0, 0.0, 1.0) * 1000).astype(np.int32)

    @classmethod
    def _signal_plane(cls, frame: np.ndarray, value_mode: str) -> np.ndarray:
        # Extract the selected signal plane before resizing into the sensor grid.
        if value_mode == "color-diff":
            return cls._color_difference_plane(frame)
        if value_mode == "red-force":
            return cls._red_force_plane(frame)
        if value_mode == "value":
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            return hsv[:, :, 2]
        if value_mode == "saturation":
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            return hsv[:, :, 1]
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _color_difference_plane(frame: np.ndarray) -> np.ndarray:
        frame_float = frame.astype(np.float32)
        height, width, _ = frame_float.shape
        border_size = max(1, min(height, width) // 20)

        border_pixels = np.concatenate(
            [
                frame_float[:border_size, :, :].reshape(-1, 3),
                frame_float[-border_size:, :, :].reshape(-1, 3),
                frame_float[:, :border_size, :].reshape(-1, 3),
                frame_float[:, -border_size:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border_pixels, axis=0)
        diff = np.linalg.norm(frame_float - background, axis=2)

        diff -= diff.min()
        max_value = diff.max()
        if max_value > 0:
            diff = diff / max_value * 255.0
        return diff.astype(np.uint8)

    @staticmethod
    def _red_force_plane(frame: np.ndarray) -> np.ndarray:
        frame_int = frame.astype(np.int16)
        red_advantage = frame_int[:, :, 0] - np.maximum(frame_int[:, :, 1], frame_int[:, :, 2])
        return np.clip(red_advantage, 0, 255).astype(np.uint8)
