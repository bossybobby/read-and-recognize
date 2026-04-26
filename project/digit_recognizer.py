import os
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader, TensorDataset


GridShape = Tuple[int, int]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DigitRecognizer:
    SUPPORTED_GRID_SHAPES = ((8, 8), (32, 32), (100, 100))
    MODEL_GRID_SHAPES = ((8, 8), (28, 28), (32, 32), (100, 100))
    CANONICAL_GRID_SHAPE = (28, 28)
    CNN_MODEL_FILENAME = "cnn_mnist_model_v2.pth"
    MODEL_FILENAMES = {
        (8, 8): "sklearn_mnist_model.joblib",
        (28, 28): "sklearn_mnist_model_28x28.joblib",
        (32, 32): "sklearn_mnist_model_32x32.joblib",
        (100, 100): "sklearn_mnist_model_100x100.joblib",
    }

    def __init__(self, grid_shape: GridShape = (8, 8), auto_train: bool = True):
        self.project_dir = Path(__file__).resolve().parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sklearn_model = None
        self.model_grid_shape = self.CANONICAL_GRID_SHAPE
        self.grid_shape = self._normalize_grid_shape(grid_shape)
        self.set_grid_shape(self.grid_shape, auto_train=auto_train)

    def set_grid_shape(self, grid_shape: GridShape, auto_train: bool = True, force_train: bool = False):
        self.grid_shape = self._normalize_grid_shape(grid_shape)
        self.model_grid_shape = self.CANONICAL_GRID_SHAPE
        self.model_path = self.project_dir / self.CNN_MODEL_FILENAME

        if self.model_path.exists() and not force_train:
            self.model = SimpleCNN().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"成功加载 CNN 模型: {self.model_path.name}")
            return

        fallback_path = self.project_dir / self.MODEL_FILENAMES[self.CANONICAL_GRID_SHAPE]
        if fallback_path.exists() and not force_train:
            self.sklearn_model = joblib.load(fallback_path)
            print(f"未找到 CNN，临时加载 28x28 线性模型: {fallback_path.name}")
            return

        if not auto_train:
            self.model = None
            self.sklearn_model = None
            print(f"未找到 CNN 模型: {self.model_path.name}")
            return

        print("未找到 CNN 模型，正在训练...")
        self.model = self.train_cnn_model()
        torch.save(self.model.state_dict(), self.model_path)
        print(f"CNN 模型训练完成并已保存: {self.model_path.name}")

    @property
    def grid_name(self) -> str:
        rows, cols = self.grid_shape
        return f"{cols}x{rows}"

    @property
    def is_ready(self) -> bool:
        return self.model is not None or self.sklearn_model is not None

    @classmethod
    def train_all_models(
        cls,
        grid_shapes: Iterable[GridShape] = SUPPORTED_GRID_SHAPES,
        force_train: bool = False,
    ):
        recognizers = []
        for grid_shape in grid_shapes:
            recognizer = cls(grid_shape=grid_shape, auto_train=False)
            recognizer.set_grid_shape(grid_shape, auto_train=True, force_train=force_train)
            recognizers.append(recognizer)
        return recognizers

    @classmethod
    def train_cnn_model(cls, epochs: int = 3, batch_size: int = 256):
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X.astype("float32") / 255.0
        y = y.astype(np.int64)

        X_train = torch.from_numpy(X.reshape(-1, 1, 28, 28))
        y_train = torch.from_numpy(y)
        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            correct = 0
            total = 0
            running_loss = 0.0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

            print(
                f"训练 CNN: epoch {epoch + 1}/{epochs}, "
                f"loss={running_loss / total:.4f}, acc={correct / total:.4f}"
            )

        model.eval()
        return model

    @classmethod
    def train_model(cls, grid_shape: GridShape, epochs: int = 5):
        grid_shape = cls._normalize_grid_shape(grid_shape)
        rows, cols = grid_shape

        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X.astype("float32") / 255.0
        y = y.astype(str)

        model = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            max_iter=1,
            random_state=42,
            tol=None,
            learning_rate="optimal",
        )

        classes = np.array([str(i) for i in range(10)])
        batch_size = 512
        rng = np.random.default_rng(42)
        for epoch in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_indices = order[start:end]
                X_batch = cls._resize_batch(X[batch_indices], (rows, cols))
                model.partial_fit(X_batch, y[batch_indices], classes=classes)
                print(f"训练 {cols}x{rows}: epoch {epoch + 1}/{epochs}, {end}/{len(X)}")

        return model

    @staticmethod
    def _resize_batch(X_batch: np.ndarray, grid_shape: GridShape) -> np.ndarray:
        rows, cols = grid_shape
        images = X_batch.reshape(-1, 28, 28)
        resized = np.empty((len(images), rows * cols), dtype=np.float32)
        for index, image in enumerate(images):
            resized[index] = cv2.resize(
                image,
                (cols, rows),
                interpolation=cv2.INTER_AREA if rows <= 28 and cols <= 28 else cv2.INTER_CUBIC,
            ).reshape(-1)
        return resized

    @staticmethod
    def _normalize_input(data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data_min = data.min()
        data_max = data.max()
        if data_max <= data_min:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)

    @classmethod
    def _preprocess_input(cls, data: np.ndarray) -> np.ndarray:
        image = cls._normalize_input(data)
        image = cls._auto_invert(image)
        image = cls._binarize_soft(image)
        return cls._center_like_mnist(image)

    @staticmethod
    def _auto_invert(image: np.ndarray) -> np.ndarray:
        if image.shape[0] < 3 or image.shape[1] < 3:
            return 1.0 - image if image.mean() > 0.5 else image

        border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
        border_level = float(np.median(border))
        if border_level > 0.5:
            return 1.0 - image
        return image

    @staticmethod
    def _binarize_soft(image: np.ndarray) -> np.ndarray:
        image_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        soft = image * (mask.astype(np.float32) / 255.0)
        if soft.max() > 0:
            soft = soft / soft.max()
        return soft.astype(np.float32)

    @staticmethod
    def _center_like_mnist(image: np.ndarray) -> np.ndarray:
        mask = image > 0.05
        if not mask.any():
            return np.zeros((28, 28), dtype=np.float32)

        ys, xs = np.where(mask)
        crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

        scale = 20.0 / max(crop.shape)
        new_w = max(1, int(round(crop.shape[1] * scale)))
        new_h = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.float32)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        moments = cv2.moments(canvas)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            shift_x = int(round(14 - cx))
            shift_y = int(round(14 - cy))
            transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, transform, (28, 28), borderValue=0)

        if canvas.max() > 0:
            canvas = canvas / canvas.max()
        return canvas.astype(np.float32)

    @classmethod
    def _normalize_grid_shape(cls, grid_shape: GridShape) -> GridShape:
        grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        if grid_shape not in cls.MODEL_GRID_SHAPES:
            supported = ", ".join(f"{cols}x{rows}" for rows, cols in cls.MODEL_GRID_SHAPES)
            rows, cols = grid_shape
            raise ValueError(f"不支持 {cols}x{rows} 阵列，仅支持: {supported}")
        return grid_shape

    def predict(self, capacitive_data):
        """
        预测手写数字。
        输入可以是 8x8、32x32 或 100x100，内部统一规范成 MNIST 风格 28x28。
        输出: (预测数字, 置信度)
        """
        try:
            if capacitive_data.shape != self.grid_shape:
                rows, cols = capacitive_data.shape
                raise ValueError(f"输入数据必须是 {self.grid_name}，当前是 {cols}x{rows}")

            normalized = self._preprocess_input(capacitive_data)
            self.last_preprocessed = normalized

            if self.model is not None:
                with torch.no_grad():
                    tensor = torch.from_numpy(normalized.reshape(1, 1, 28, 28)).to(self.device)
                    probabilities = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
                    prediction = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))
                    return prediction, confidence

            if self.sklearn_model is not None:
                flat_input = normalized.reshape(1, -1)
                prediction = self.sklearn_model.predict(flat_input)[0]
                probabilities = self.sklearn_model.predict_proba(flat_input)[0]
                return int(prediction), float(np.max(probabilities))

            return -1, 0.0
        except Exception as e:
            print(f"预测错误: {e}")
            return -1, 0.0

    def save_last_preprocessed(self, path: str | os.PathLike):
        if not hasattr(self, "last_preprocessed"):
            return False
        image = np.clip(self.last_preprocessed * 255, 0, 255).astype(np.uint8)
        return cv2.imwrite(str(path), image)


def _shape_from_arg(value: str) -> GridShape:
    value = value.lower()
    size = int(value.split("x", 1)[0] if "x" in value else value)
    return (size, size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST classifiers for sensor grids.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["cnn"],
        help="Use 'cnn' for the default neural network, or grid sizes such as 8 28 32 100.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite existing models.")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)

    if any(size.lower() == "cnn" for size in args.sizes):
        recognizer = DigitRecognizer(auto_train=False)
        if recognizer.model_path.exists() and not args.force:
            print(f"已存在 {recognizer.model_path.name}，跳过训练")
        else:
            recognizer.model = DigitRecognizer.train_cnn_model()
            torch.save(recognizer.model.state_dict(), recognizer.model_path)
            print(f"CNN 模型训练完成并已保存: {recognizer.model_path.name}")
    else:
        shapes = [_shape_from_arg(size) for size in args.sizes]
        DigitRecognizer.train_all_models(shapes, force_train=args.force)
