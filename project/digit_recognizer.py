import numpy as np
import joblib
import os
import cv2
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

class DigitRecognizer:
    def __init__(self):
        self.model_path = 'sklearn_mnist_model.joblib'
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("成功加载Scikit-learn预训练模型")
        else:
            print("未找到预训练模型，正在从OpenML下载并训练...")
            self._train_sklearn_model()
            joblib.dump(self.model, self.model_path)
            print("模型训练完成并已保存")

    def _train_sklearn_model(self):
        """使用Scikit-learn训练一个SGD分类器"""
        # 从OpenML获取MNIST数据集 (70,000 samples)
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype('float32') / 255.0  # 归一化到 [0, 1]

        # 由于我们的输入是8x8，我们需要将28x28的数据下采样
        # 将784维向量重塑为28x28，然后下采样到8x8，再展平回64维
        X_28x28 = X.reshape(-1, 28, 28)
        X_8x8 = np.array([self._downsample_28_to_8(img) for img in X_28x28])
        X_flat = X_8x8.reshape(-1, 64)

        # 训练SGD分类器
        self.model = SGDClassifier(
            loss='log_loss', # 使用对数损失（即逻辑回归）
            alpha=0.0001,
            max_iter=10,
            random_state=42,
            tol=1e-3,
            verbose=True
        )
        self.model.fit(X_flat, y)

    def _downsample_28_to_8(self, img_28x28):
        """将28x28的图像下采样到8x8"""
        return cv2.resize(img_28x28, (8, 8), interpolation=cv2.INTER_AREA)

    def predict(self, capacitive_data):
        """
        预测手写数字
        输入: 8x8 numpy array
        输出: (预测数字, 置信度)
        """
        try:
            # 确保输入是8x8
            if capacitive_data.shape != (8, 8):
                raise ValueError("输入数据必须是8x8")
            
            # 归一化并展平
            data_min = capacitive_data.min()
            data_max = capacitive_data.max()
            if data_max > data_min:
                normalized = (capacitive_data - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(capacitive_data)
            
            flat_input = normalized.reshape(1, -1) # shape: (1, 64)
            
            # 预测
            prediction = self.model.predict(flat_input)[0]
            probabilities = self.model.predict_proba(flat_input)[0]
            confidence = np.max(probabilities)
            
            return int(prediction), float(confidence)
        except Exception as e:
            print(f"预测错误: {e}")
            return -1, 0.0

def simulate_digit_patterns():
    """生成模拟的手写数字模式用于测试"""
    patterns = {
        0: np.array([
            [0,1,1,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0]
        ]),
        1: np.array([
            [0,0,1,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,1,1,1,0]
        ]),
        2: np.array([
            [0,1,1,1,0],
            [1,0,0,0,1],
            [0,0,1,1,0],
            [0,1,0,0,0],
            [1,1,1,1,1]
        ]),
    }

    # 扩展到8x8
    expanded_patterns = {}
    for digit, pattern in patterns.items():
        expanded = np.zeros((8, 8))
        start = (8 - 5) // 2
        expanded[start:start+5, start:start+5] = pattern
        # 缩放值到电容范围
        expanded_patterns[digit] = (expanded * 600 + 200).astype(int)

    return expanded_patterns

if __name__ == "__main__":
    # 测试识别器
    recognizer = DigitRecognizer()

    # 使用模拟模式
    patterns = simulate_digit_patterns()

    print("测试数字识别:")
    for digit, pattern in patterns.items():
        pred, conf = recognizer.predict(pattern)
        print(f"真实数字: {digit}, 预测: {pred}, 置信度: {conf:.3f}")

        # 显示模式
        plt.figure(figsize=(4, 4))
        plt.imshow(pattern, cmap='viridis')
        plt.title(f"数字 {digit} - 预测 {pred} (置信度: {conf:.3f})")
        plt.colorbar()
        plt.show()